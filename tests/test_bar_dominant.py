from datavideo.bar_dominant import (
    _frames_in_window,
    _load_search_anchor,
    _positive_reaches_right_boundary,
    _trim_item,
    merge_bar_candidates,
)


def _candidate(idx, start, end, identity, mark_types=None, scene_states=None):
    return {
        "clip_id": f"bar_candidate_{idx:03d}",
        "start": start,
        "end": end,
        "source_start": start + 0.5,
        "source_end": end - 0.5,
        "confidence": 0.9,
        "positive_frame_count": 4,
        "scene_states": scene_states or ["stable_chart"],
        "animation_cues": ["unknown"],
        "chart_identities": [identity],
        "chart_types": ["bar-chart"],
        "mark_types": mark_types or ["vertical-bar"],
    }


def test_merge_bar_candidates_blocks_different_chart_identity():
    merged = merge_bar_candidates(
        [
            _candidate(0, 10.0, 12.0, "sales by region"),
            _candidate(1, 13.0, 15.0, "market share by age"),
        ],
        max_gap=2.0,
    )

    assert len(merged) == 2
    assert merged[1]["merge_block_reasons"] == ["different_chart_identity"]


def test_merge_bar_candidates_blocks_orientation_change():
    merged = merge_bar_candidates(
        [
            _candidate(0, 10.0, 12.0, "same chart", mark_types=["vertical-bar"]),
            _candidate(1, 13.0, 15.0, "same chart", mark_types=["horizontal-bar"]),
        ],
        max_gap=2.0,
    )

    assert len(merged) == 2
    assert "bar_orientation_changed" in merged[1]["merge_block_reasons"]


def test_merge_bar_candidates_allows_same_identity_continuation():
    merged = merge_bar_candidates(
        [
            _candidate(0, 10.0, 12.0, "sales by region"),
            _candidate(1, 13.0, 15.0, "sales by region"),
        ],
        max_gap=2.0,
    )

    assert len(merged) == 1
    assert merged[0]["candidate_ids"] == ["bar_candidate_000", "bar_candidate_001"]


def test_load_search_anchor_uses_sample_id(tmp_path):
    start_times = tmp_path / "start_time.jsonl"
    start_times.write_text(
        '{"video_id":"bar_001","video_path":"other.mp4","clip_start_sec":65.0}\n',
        encoding="utf-8",
    )
    cfg = {
        "sample_id": "bar_001",
        "video_path": "expected.mp4",
        "target_search": {"start_times_path": str(start_times)},
    }

    anchor = _load_search_anchor(cfg)

    assert anchor["clip_start_sec"] == 65.0
    assert anchor["video_path_matches_config"] is False


def test_frames_in_window_uses_inclusive_bounds():
    frames = [{"timestamp": value} for value in [62.5, 63.0, 63.5, 85.0, 85.5]]

    selected = _frames_in_window(frames, 63.0, 85.0)

    assert [frame["timestamp"] for frame in selected] == [63.0, 63.5, 85.0]


def test_positive_at_right_boundary_triggers_extension():
    frames = [
        {"frame_id": "f0", "timestamp": 84.0},
        {"frame_id": "f1", "timestamp": 84.5},
        {"frame_id": "f2", "timestamp": 85.0},
    ]
    positive = {
        "is_bar_chart_dominant_candidate": True,
        "bar_marks_visible": True,
        "bar_marks_dominant": True,
        "has_data_encoding_evidence": True,
    }
    results = [
        {"frame_id": "f0", "result": positive},
        {"frame_id": "f1", "result": positive},
        {"frame_id": "f2", "result": {}},
    ]

    assert _positive_reaches_right_boundary(frames, results, max_gap=1.0)
    assert not _positive_reaches_right_boundary(frames, results, max_gap=0.25)


def test_trim_item_accepts_clip_relative_times():
    item = {
        "start": 178.5,
        "end": 203.5,
        "qwen_review": {"suggested_start": 0, "suggested_end": 15},
    }

    trimmed = _trim_item(item)

    assert trimmed["start"] == 178.5
    assert trimmed["end"] == 193.5
    assert trimmed["duration"] == 15.0
