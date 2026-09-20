"""Split assignment must be deterministic AND stable as the dataset grows."""

from collections import Counter

from multi_ego_cs.stages.package import assign_split

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def _ids(n):
    return [f"1-{i:08x}-4ce9-45fa-8e0b-35fd140ddd60-1-1" for i in range(n)]


def test_deterministic():
    mid = _ids(1)[0]
    assert assign_split(mid, RATIOS, 1999) == assign_split(mid, RATIOS, 1999)


def test_seed_changes_assignment():
    a = [assign_split(m, RATIOS, 1) for m in _ids(200)]
    b = [assign_split(m, RATIOS, 2) for m in _ids(200)]
    assert a != b


def test_stable_when_dataset_grows():
    # The property that makes "the test split" mean the same thing across
    # dataset versions: adding matches must not reshuffle existing ones.
    first = {m: assign_split(m, RATIOS, 1999) for m in _ids(100)}
    later = {m: assign_split(m, RATIOS, 1999) for m in _ids(1000)}
    for m, split in first.items():
        assert later[m] == split


def test_ratios_approximately_respected():
    counts = Counter(assign_split(m, RATIOS, 1999) for m in _ids(4000))
    assert counts["train"] / 4000 == __import__("pytest").approx(0.7, abs=0.03)
    assert counts["val"] / 4000 == __import__("pytest").approx(0.15, abs=0.03)
    assert counts["test"] / 4000 == __import__("pytest").approx(0.15, abs=0.03)


def test_only_declared_splits_are_produced():
    assert {assign_split(m, RATIOS, 7) for m in _ids(500)} <= set(RATIOS)


def test_empty_ratios_defaults_to_train():
    assert assign_split(_ids(1)[0], {}, 1999) == "train"
