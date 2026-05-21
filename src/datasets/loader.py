from __future__ import annotations

import logging
import random
from typing import Any

from .schemas import ItemProfile, Rating, UserProfile

logger = logging.getLogger(__name__)

_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "Western",
]

# HuggingFace dataset — falls back to synthetic data if unavailable
_HF_DATASET_NAME = "nateraw/movie-lens-latest-small"


class DatasetLoader:
    """Load recommendation datasets from HuggingFace or generate synthetic data."""

    def load_huggingface(
        self,
        dataset_name: str | None = None,
    ) -> tuple[list[Rating], list[ItemProfile], list[UserProfile]]:
        """Load a rating dataset from HuggingFace datasets library.

        Falls back to a synthetic dataset if the download fails.
        """
        name = dataset_name or _HF_DATASET_NAME
        try:
            from datasets import load_dataset  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Install the 'datasets' package: pip install datasets"
            ) from exc

        try:
            ds = load_dataset(name, trust_remote_code=True)
            return self._parse_hf_dataset(ds)
        except Exception as exc:
            logger.warning(
                "Could not load HuggingFace dataset '%s': %s — using synthetic data.",
                name,
                exc,
            )
            return self.generate_synthetic()

    def _parse_hf_dataset(
        self, ds: Any
    ) -> tuple[list[Rating], list[ItemProfile], list[UserProfile]]:
        split = ds.get("train") or next(iter(ds.values()))
        ratings: list[Rating] = []
        for row in split:
            try:
                uid = str(
                    row.get("userId") or row.get("user_id") or row.get("user") or ""
                )
                iid = str(
                    row.get("movieId")
                    or row.get("item_id")
                    or row.get("asin")
                    or row.get("item")
                    or ""
                )
                if not uid or not iid:
                    continue
                raw = float(row.get("rating") or row.get("score") or 3.0)
                ts = int(row.get("timestamp") or 0)
                ratings.append(
                    Rating(
                        user_id=uid,
                        item_id=iid,
                        rating=float(min(max(raw, 0.0), 5.0)),
                        timestamp=ts,
                    )
                )
            except (TypeError, ValueError, KeyError):
                continue

        users = [UserProfile(user_id=uid) for uid in {r.user_id for r in ratings}]
        items = [
            ItemProfile(item_id=iid, title=f"Item {iid}")
            for iid in {r.item_id for r in ratings}
        ]
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
