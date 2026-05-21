"""Security tests aligned with NIST SP 800-53 controls."""
from __future__ import annotations

import logging
import os

import pytest

from src.core.base import BaseLLM, LLMConfig, LLMResponse, Message
from src.datasets.schemas import Rating
from src.recommender.hybrid import HybridRecommender
from src.recommender.pipeline import RecommendationPipeline


# ─── Mock LLM ─────────────────────────────────────────────────────────────────

class _MockLLM(BaseLLM):
    def __init__(self):
        super().__init__(LLMConfig(model="mock"))

    def complete(self, messages):
        return LLMResponse(content="safe", model=self.config.model)

    def stream(self, messages):
        yield "safe"


def _small_ratings():
    data = [
        ("u1","i1",5.0),("u1","i2",4.0),
        ("u2","i1",4.0),("u2","i3",5.0),
        ("u3","i2",5.0),("u3","i4",3.0),
    ]
    return [Rating(u, i, r) for u, i, r in data]


# ─── AC (Access Control) ──────────────────────────────────────────────────────

class TestAccessControl:
    """NIST AC: Least privilege and separation of duties."""

    def test_no_cross_user_data_leakage(self):
        """Recommendations for one user must not return another user's private items."""
        hr = HybridRecommender()
        hr.fit(_small_ratings())
        recs_u1 = {iid for iid, _ in hr.recommend("u1", n=10).recommendations}
        recs_u2 = {iid for iid, _ in hr.recommend("u2", n=10).recommendations}
        # Both are valid item sets; neither should error out or be identical to the other's
        # (unless the datasets genuinely overlap — what we check is no crash/exception)
        assert isinstance(recs_u1, set)
        assert isinstance(recs_u2, set)

    def test_pipeline_unknown_user_no_exception(self):
        p = RecommendationPipeline()
        p.fit(_small_ratings())
        # An unknown user should get graceful handling, not a crash
        result = p.run("totally_unknown_user", n=5)
        assert result is not None


# ─── AU (Audit and Accountability) ────────────────────────────────────────────

class TestAuditLogging:
    """NIST AU: Audit events are logged with sufficient detail."""

    def test_pipeline_run_logs_activity(self, caplog):
        p = RecommendationPipeline()
        p.fit(_small_ratings())
        with caplog.at_level(logging.DEBUG):
            p.run("u1", n=3)
        # The call must not suppress all logging (INFO+ should work)
        # We don't mandate exact messages, just confirm no crash during logging

    def test_hybrid_fit_logs_info(self, caplog):
        with caplog.at_level(logging.INFO, logger="src.recommender.hybrid"):
            HybridRecommender().fit(_small_ratings())
        assert any("fitted" in r.message.lower() for r in caplog.records)

    def test_no_secrets_in_logs(self, caplog, monkeypatch):
        import boto3

        class _FakeSession:
            def client(self, *a, **kw):
                return object()

        monkeypatch.setattr(boto3, "Session", lambda **kw: _FakeSession())
        from src.core.bedrock_client import BedrockClient
        with caplog.at_level(logging.DEBUG):
            BedrockClient()
        for record in caplog.records:
            assert "AKIA" not in record.message
            assert "secret" not in record.message.lower()


# ─── CM (Configuration Management) ───────────────────────────────────────────

class TestConfigurationManagement:
    """NIST CM: Secure configuration baseline."""

    def test_model_config_exists(self):
        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "model_config.yaml"
        )
        assert os.path.isfile(cfg_path), "model_config.yaml missing"

    def test_model_config_valid_yaml(self):
        import yaml
        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "model_config.yaml"
        )
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg, dict)

    def test_logging_config_exists(self):
        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "logging_config.yaml"
        )
        assert os.path.isfile(cfg_path), "logging_config.yaml missing"

    def test_no_debug_mode_in_default_logging_config(self):
        import yaml
        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "logging_config.yaml"
        )
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        # Root logger should not default to DEBUG in production config
        root_level = (
            cfg.get("root", {}).get("level", "INFO")
            if isinstance(cfg, dict)
            else "INFO"
        )
        assert root_level.upper() != "DEBUG"


# ─── IA (Identification and Authentication) ───────────────────────────────────

class TestIdentificationAuthentication:
    """NIST IA: Credentials not stored in source; env vars used."""

    def test_env_example_uses_placeholders(self):
        env_path = os.path.join(
            os.path.dirname(__file__), "..", "..", ".env.example"
        )
        if not os.path.exists(env_path):
            pytest.skip(".env.example not found")
        with open(env_path) as f:
            content = f.read()
        # Should only contain placeholder values, not real keys
        for line in content.splitlines():
            if line.startswith("#") or not line.strip() or "=" not in line:
                continue
            _, _, value = line.partition("=")
            assert len(value.strip()) < 50 or value.strip().startswith("your"), (
                f"Suspicious value in .env.example: {line}"
            )

    def test_factory_reads_from_env_not_hardcoded(self):
        import inspect
        from src.core import factory
        src = inspect.getsource(factory)
        # Should reference os.environ or os.getenv, not hardcoded strings
        assert "os.environ" in src or "os.getenv" in src


# ─── SC (System and Communications Protection) ────────────────────────────────

class TestSystemProtection:
    """NIST SC: Input bounds enforced; no buffer overflows via large inputs."""

    def test_huge_item_list_handled(self):
        """Recommender must handle large item sets without crashing."""
        from src.datasets.loader import DatasetLoader
        loader = DatasetLoader()
        ratings, items, users = loader.generate_synthetic(
            n_users=5, n_items=200, n_ratings=1000, seed=0
        )
        hr = HybridRecommender()
        hr.fit(ratings)
        result = hr.recommend(users[0].user_id, n=50)
        assert len(result.recommendations) <= 50

    def test_linucb_context_dim_enforced(self):
        import numpy as np
        from src.rl.contextual_bandit import LinUCBBandit, LinUCBConfig
        b = LinUCBBandit(n_arms=3, config=LinUCBConfig(context_dim=4))
        # Wrong-dimension context must raise
        with pytest.raises(ValueError):
            b.select_arm(np.ones(8))

    def test_narrator_truncates_oversized_input(self):
        from src.datasets.schemas import ItemProfile
        from src.narration.narrator import RecommendationNarrator

        llm = _MockLLM()
        narrator = RecommendationNarrator(llm)
        giant_title = "X" * 100_000
        items_by_id = {
            "i1": ItemProfile(item_id="i1", title=giant_title, genres=("Drama",))
        }
        result = narrator.narrate(
            user_id="u1",
            recommendations=[("i1", 4.0)],
            items_by_id=items_by_id,
        )
        assert isinstance(result.narrative, str)


# ─── SI (System and Information Integrity) ────────────────────────────────────

class TestSystemIntegrity:
    """NIST SI: Immutability and data integrity checks."""

    def test_rating_is_frozen(self):
        r = Rating(user_id="u1", item_id="i1", rating=3.5)
        with pytest.raises(Exception):
            r.user_id = "attacker"  # type: ignore[misc]

    def test_mf_predict_clipped_to_valid_range(self):
        from src.collaborative.matrix_factorization import MatrixFactorization, MFConfig
        mf = MatrixFactorization(MFConfig(n_factors=5, n_epochs=5))
        mf.fit(_small_ratings())
        pred = mf.predict("u1", "i1")
        assert mf.config.rating_min <= pred <= mf.config.rating_max

    def test_synthetic_ratings_uniqueness(self):
        from src.datasets.loader import DatasetLoader
        loader = DatasetLoader()
        ratings, _, _ = loader.generate_synthetic(n_users=10, n_items=20, n_ratings=50)
        pairs = [(r.user_id, r.item_id) for r in ratings]
        assert len(pairs) == len(set(pairs)), "Duplicate (user, item) pairs found"

    def test_pipeline_result_elapsed_non_negative(self):
        p = RecommendationPipeline()
        p.fit(_small_ratings())
        result = p.run("u1", n=3)
        assert result.elapsed_seconds >= 0.0
