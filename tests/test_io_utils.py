"""Atomic writes and materialization. Long runs get interrupted; partial files kill resume."""

import json
import os

from multi_ego_cs.util.io_utils import (
    append_jsonl,
    human_bytes,
    materialize,
    read_json,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)


def test_json_roundtrip(tmp_path):
    p = write_json_atomic(tmp_path / "a" / "b.json", {"x": 1})
    assert read_json(p) == {"x": 1}


def test_atomic_write_leaves_no_temp_files(tmp_path):
    write_json_atomic(tmp_path / "a.json", {"x": 1})
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_jsonl_append_and_read(tmp_path):
    p = tmp_path / "m.jsonl"
    append_jsonl(p, {"i": 1})
    append_jsonl(p, {"i": 2})
    assert [r["i"] for r in read_jsonl(p)] == [1, 2]


def test_read_jsonl_tolerates_truncated_final_line(tmp_path):
    # Exactly what a hard kill mid-write leaves behind.
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({"i": 1}) + "\n" + '{"i": 2')
    assert [r["i"] for r in read_jsonl(p)] == [1]


def test_read_jsonl_missing_file_is_empty(tmp_path):
    assert list(read_jsonl(tmp_path / "nope.jsonl")) == []


def test_write_jsonl_atomic(tmp_path):
    p = write_jsonl_atomic(tmp_path / "m.jsonl", [{"i": 1}, {"i": 2}])
    assert [r["i"] for r in read_jsonl(p)] == [1, 2]


def test_materialize_hardlinks_within_a_filesystem(tmp_path):
    src = tmp_path / "s.bin"
    src.write_bytes(b"data")
    dst = tmp_path / "out" / "d.bin"
    assert materialize(src, dst, "auto") == "hardlink"
    assert dst.read_bytes() == b"data"
    assert os.stat(src).st_ino == os.stat(dst).st_ino


def test_materialize_symlink_mode(tmp_path):
    src = tmp_path / "s.bin"
    src.write_bytes(b"data")
    dst = tmp_path / "d.bin"
    assert materialize(src, dst, "symlink") == "symlink"
    assert dst.is_symlink() and dst.read_bytes() == b"data"


def test_materialize_copy_is_independent(tmp_path):
    src = tmp_path / "s.bin"
    src.write_bytes(b"data")
    dst = tmp_path / "d.bin"
    assert materialize(src, dst, "copy") == "copy"
    src.unlink()
    assert dst.read_bytes() == b"data"


def test_materialize_is_idempotent(tmp_path):
    src = tmp_path / "s.bin"
    src.write_bytes(b"x")
    dst = tmp_path / "d.bin"
    materialize(src, dst, "copy")
    assert materialize(src, dst, "copy") == "exists"


def test_human_bytes():
    assert human_bytes(0) == "0.0 B"
    assert human_bytes(1536) == "1.5 KiB"
    assert human_bytes(1024 ** 3) == "1.0 GiB"
