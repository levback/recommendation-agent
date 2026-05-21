from __future__ import annotations

import logging
import os
import pickle
import random
from pathlib import Path
from typing import Any

from .schemas import ItemProfile, Rating, UserProfile

logger = logging.getLogger(__name__)

_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "Western",
]

# Default HuggingFace dataset: MovieLens ratings with title + genre metadata
# 891 k train ratings, ~30 MB download (parquet, no script required)
_HF_DATASET_NAME = "ashraq/movielens_ratings"

# Local cache directory (relative to project root, or override via DATA_DIR env)
_DEFAULT_CACHE_DIR = Path(
    os.environ.get("DATA_DIR", Path(__file__).resolve().parents[2] / "data")
) / "datasets"


class DatasetLoader:
    """Load recommendation datasets from HuggingFace or generate synthetic data.

    Downloaded datasets are cached to ``cache_dir`` as pickle files so
    subsequent runs are instant (no network call needed).
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def load_huggingface(
        self,
        dataset_name: str | None = None,
        max_ratings: int | None = None,
        split: str = "train",
        min_ratings_per_user: int = 10,
    ) -> tuple[list[Rating], list[ItemProfile], list[UserProfile]]:
        """Load a rating dataset from HuggingFace datasets library.

        Downloaded data is cached locally so repeat runs skip the download.
        Falls back to synthetic data if the download fails.

        Args:
            dataset_name: HuggingFace dataset ID (default: ashraq/movielens_ratings).
            max_ratings: If set, cap the number of ratings after filtering (None = all).
            split: Dataset split to use (default: "train").
            min_ratings_per_user: Drop users with fewer ratings than this threshold.
                Keeps the dataset dense enough for neighbourhood-based CF (default: 10).
        """
        name = dataset_name or _HF_DATASET_NAME
        cache_key = name.replace("/", "__") + f"__{split}__minU{min_ratings_per_user}"
        if max_ratings:
            cache_key += f"__max{max_ratings}"
        cache_path = self._cache_dir / f"{cache_key}.pkl"

        if cache_path.exists():
            logger.info("Loading dataset from cache: %s", cache_path)
            with cache_path.open("rb") as f:
                return pickle.load(f)  # noqa: S301 — loading our own cache

        try:
            from datasets import load_dataset  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Install the 'datasets' package: pip install datasets"
            ) from exc

        try:
            logger.info("Downloading HuggingFace dataset '%s' (split=%s)…", name, split)
            ds = load_dataset(name, split=split)
            result = self._parse_hf_dataset(
                ds,
                max_ratings=max_ratings,
                min_ratings_per_user=min_ratings_per_user,
            )
            with cache_path.open("wb") as f:
                pickle.dump(result, f)
            logger.info("Cached %d ratings to %s", len(result[0]), cache_path)
            return result
        except Exception as exc:
            logger.warning(
                "Could not load HuggingFace dataset '%s': %s — using synthetic data.",
                name,
                exc,
            )
            return self.generate_synthetic()

    def _parse_hf_dataset(
        self,
        ds: Any,
        max_ratings: int | None = None,
        min_ratings_per_user: int = 0,
    ) -> tuple[list[Rating], list[ItemProfile], list[UserProfile]]:
        """Parse a HuggingFace dataset into domain objects.

        Handles the ``ashraq/movielens_ratings`` schema (movie_id, user_id,
        rating, title, genres) as well as generic column name variants.

        When ``min_ratings_per_user > 0``, users with fewer ratings are dropped
        to ensure the dataset is dense enough for neighbourhood-based CF.
        """
        all_ratings: list[Rating] = []
        items_map: dict[str, ItemProfile] = {}

        for row in ds:
            try:
                uid = str(
                    row.get("user_id") or row.get("userId") or row.get("user") or ""
                )
                iid = str(
                    row.get("movie_id") or row.get("movieId")
                    or row.get("item_id") or row.get("asin")
                    or row.get("item") or ""
                )
                if not uid or not iid or iid == "0":
                    continue
                raw = float(row.get("rating") or row.get("score") or 3.0)
                ts = int(row.get("timestamp") or 0)
                all_ratings.append(
                    Rating(
                        user_id=uid,
                        item_id=iid,
                        rating=float(min(max(raw, 0.0), 5.0)),
                        timestamp=ts,
                    )
                )
                if iid not in items_map:
                    title = str(row.get("title") or f"Item {iid}")
                    raw_genres = row.get("genres") or ""
                    if isinstance(raw_genres, str):
                        genre_list = [
                            g.strip() for g in raw_genres.replace("|", ",").split(",")
                            if g.strip() and g.strip() != "(no genres listed)"
                        ]
                    else:
                        genre_list = [str(g) for g in raw_genres if g]
                    items_map[iid] = ItemProfile(
                        item_id=iid,
                        title=title,
                        genres=tuple(genre_list),
                    )
            except (TypeError, ValueError, KeyError):
                continue

        # Filter to active users
        if min_ratings_per_user > 1:
            from collections import Counter
            counts = Counter(r.user_id for r in all_ratings)
            active = {uid for uid, n in counts.items() if n >= min_ratings_per_user}
            all_ratings = [r for r in all_ratings if r.user_id in active]
            logger.info(
                "Kept %d/%d users (≥%d ratings each); %d ratings remaining",
                len(active), len(counts), min_ratings_per_user, len(all_ratings),
            )

        # Cap after filtering
        ratings = all_ratings[:max_ratings] if max_ratings else all_ratings

        # Rebuild items to only those present after filtering
        present_items = {r.item_id for r in ratings}
        items = [v for k, v in items_map.items() if k in present_items]
        users = [UserProfile(user_id=uid) for uid in sorted({r.user_id for r in ratings})]
        logger.info(
            "Final dataset: %d ratings, %d items, %d users",
            len(ratings), len(items), len(users),
        )
        return ratings, items, users

    def generate_synthetic(
        self,
        n_users: int = 100,
        n_items: int = 200,
        n_ratings: int = 2000,
        seed: int = 42,
    ) -> tuple[list[Rating], list[ItemProfile], list[UserProfile]]:
        """Generate a synthetic MovieLens-style dataset with preference-aware ratings."""
        if n_users < 1 or n_items < 1 or n_ratings < 1:
            raise ValueError("n_users, n_items, and n_ratings must be >= 1")

        rng = random.Random(seed)

        # Build items with random genre tags
        items: list[ItemProfile] = []
        for i in range(1, n_items + 1):
            n_g = rng.randint(1, 3)
            genres = tuple(rng.sample(_GENRES, n_g))
            title = f"Movie {i}: {' & '.join(genres[:2])}"
            items.append(
                ItemProfile(
                    item_id=str(i),
                    title=title,
                    genres=genres,
                    features={g: 1.0 for g in genres},
                )
            )

        # Build users with random genre preferences
        users: list[UserProfile] = []
        for u in range(1, n_users + 1):
            pref_genres = rng.sample(_GENRES, rng.randint(2, 5))
            users.append(
                UserProfile(
                    user_id=str(u),
                    features={g: rng.uniform(0.4, 1.0) for g in pref_genres},
                )
            )

        # Generate preference-aware ratings
        ratings: list[Rating] = []
        seen: set[tuple[str, str]] = set()
        attempts = 0
        while len(ratings) < n_ratings and attempts < n_ratings * 20:
            attempts += 1
            user = rng.choice(users)
            item = rng.choice(items)
            key = (user.user_id, item.item_id)
            if key in seen:
                continue
            seen.add(key)
            overlap = len(set(user.features) & set(item.features))
            base = 2.5 + overlap * 0.6
            raw = base + rng.gauss(0, 0.8)
            rating = round(float(min(5.0, max(1.0, raw))), 1)
            ratings.append(
                Rating(
                    user_id=user.user_id,
                    item_id=item.item_id,
                    rating=rating,
                    timestamp=rng.randint(1_000_000, 2_000_000),
                )
            )

        return ratings, items, users
