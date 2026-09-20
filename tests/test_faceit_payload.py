"""Payload readers.

`voting.map.pick` is the only field that reliably holds the decision. On a real
101-match sample, reading `voting.map.entities` resolved a single map for only
40; the rest still listed 2-7 candidates.
"""

from multi_ego_cs.stages.faceit_api import (
    read_demo_urls,
    read_finished_at,
    read_picked_map,
    read_players,
)

PAYLOAD = {
    "payload": {
        "voting": {"map": {"pick": ["de_overpass"],
                           "entities": [{"game_map_id": "de_overpass"}]}},
        "demoURLs": ["https://cdn.example/1-abc-1-1.dem.zst"],
        "finishedAt": "2025-09-12T00:03:20Z",
        "teams": {
            "faction1": {"roster": [
                {"player_id": "p1", "nickname": "a", "game_player_id": "76561198000000001"}
            ]},
            "faction2": {"roster": [
                {"player_id": "p2", "nickname": "b", "game_player_id": "76561198000000002"}
            ]},
        },
    }
}


def test_reads_pick_not_pool():
    assert read_picked_map(PAYLOAD) == "de_overpass"


def test_pick_wins_over_ambiguous_entities():
    p = {"payload": {"voting": {"map": {
        "pick": ["de_nuke"],
        "entities": [{"game_map_id": "de_mirage"}, {"game_map_id": "de_nuke"}],
    }}}}
    assert read_picked_map(p) == "de_nuke"


def test_ambiguous_entities_without_pick_yield_nothing():
    p = {"payload": {"voting": {"map": {
        "entities": [{"game_map_id": "de_mirage"}, {"game_map_id": "de_nuke"}]
    }}}}
    assert read_picked_map(p) is None


def test_single_entity_is_an_acceptable_fallback():
    p = {"payload": {"voting": {"map": {"entities": [{"game_map_id": "de_dust2"}]}}}}
    assert read_picked_map(p) == "de_dust2"


def test_handles_top_level_and_nested_shapes():
    flat = {"voting": {"map": {"pick": ["de_train"]}}, "demo_url": "https://x/y.dem.zst"}
    assert read_picked_map(flat) == "de_train"
    assert read_demo_urls(flat) == ["https://x/y.dem.zst"]


def test_demo_urls_and_finished_at():
    assert read_demo_urls(PAYLOAD) == ["https://cdn.example/1-abc-1-1.dem.zst"]
    assert read_finished_at(PAYLOAD) == "2025-09-12T00:03:20Z"


def test_missing_fields_are_not_fatal():
    assert read_picked_map({}) is None
    assert read_demo_urls({}) == []
    assert read_finished_at({}) is None
    assert read_players({}) == []


def test_players_flattened_with_steamids():
    players = read_players(PAYLOAD)
    assert len(players) == 2
    assert {p["steamid"] for p in players} == {
        "76561198000000001", "76561198000000002"
    }
