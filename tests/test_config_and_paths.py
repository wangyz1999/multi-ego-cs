"""Config validation and the on-disk layout contract."""

import os

import pytest
import yaml

from multi_ego_cs.config import Config, load_config, resolve_workers
from multi_ego_cs.paths import Layout

MATCH = "1-0076bc6b-4ce9-45fa-8e0b-35fd140ddd60-1-1"
STEAM = "76561198064353169"


def test_layout_paths(tmp_path):
    lay = Layout(tmp_path)
    assert lay.video_file(MATCH, 5, STEAM) == (
        tmp_path / "video" / f"match={MATCH}" / "round=5" / f"{STEAM}.mp4"
    )
    assert lay.state_action_file(MATCH, 5, STEAM).suffix == ".parquet"
    assert lay.offsets_file(MATCH, 5).name == "offsets.json"
    assert lay.demo_file(MATCH).name == f"{MATCH}.dem"


def test_layout_expands_env_vars(monkeypatch, tmp_path):
    monkeypatch.setenv("MECS_TEST_ROOT", str(tmp_path))
    assert Layout("$MECS_TEST_ROOT/data").root == tmp_path / "data"


def test_layout_ensure_is_idempotent(tmp_path):
    lay = Layout(tmp_path).ensure()
    Layout(tmp_path).ensure()
    assert lay.demo.is_dir() and lay.state_action.is_dir()


def test_unknown_config_key_is_rejected(tmp_path):
    # A silently-ignored typo would leave the run looking configured while
    # doing something else entirely.
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml.safe_dump({"discover": {"taget_matches": 10}}))
    with pytest.raises(ValueError, match="Unknown key"):
        load_config(cfg)


def test_config_roundtrip(tmp_path):
    cfg = Config()
    cfg.discover.target_matches = 1000
    out = tmp_path / "c.yaml"
    cfg.dump(out)
    assert load_config(out).discover.target_matches == 1000


def test_env_override(tmp_path, monkeypatch):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"discover": {"target_matches": 10}}))
    monkeypatch.setenv("MECS_TARGET_MATCHES", "250")
    assert load_config(cfg).discover.target_matches == 250


def test_resolve_workers_respects_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    assert resolve_workers(0, 1.0) == 16
    assert resolve_workers(4, 1.0) == 4  # explicit wins


def test_resolve_workers_never_zero(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    assert resolve_workers(0, 0.1) >= 1


def test_secrets_are_env_only(monkeypatch):
    cfg = Config()
    monkeypatch.setenv("MECS_FACEIT_API_KEY", "abc")
    assert cfg.faceit_api_key == "abc"
    assert "faceit_api_key" not in cfg.to_dict()
    assert "hf_token" not in cfg.to_dict()
