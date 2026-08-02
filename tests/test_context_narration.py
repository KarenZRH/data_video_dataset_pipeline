from datavideo.context import build_intervals_payload, compute_context_source_interval
from datavideo.narration import build_sentence_boundaries, select_full_sentences_for_reference, slice_transcript
from datavideo.visual_provenance import assert_qwen_visual_inputs


def _cfg():
    return {
        "context": {"padding_before_seconds": 5.0, "padding_after_seconds": 5.0},
        "asr": {
            "sentence_safety_margin_seconds": 0.2,
            "max_boundary_expansion_seconds": 5.0,
            "pause_threshold_seconds": 0.45,
        },
    }


def _intervals(requires_context_redownload=False):
    return build_intervals_payload(
        {"start_seconds": 65.0, "end_seconds": 77.0},
        _cfg(),
        context_source={"start": 60.0, "end": 82.0},
        context_duration=22.0,
        requires_context_redownload=requires_context_redownload,
    )


def test_context_start_clamps_at_video_start():
    interval = compute_context_source_interval(3.0, 10.0, padding_before_seconds=5.0, padding_after_seconds=5.0)

    assert interval["start"] == 0.0
    assert interval["end"] == 15.0


def test_context_end_clamps_at_video_end():
    interval = compute_context_source_interval(90.0, 98.0, padding_before_seconds=5.0, padding_after_seconds=5.0, source_duration_seconds=100.0)

    assert interval["start"] == 85.0
    assert interval["end"] == 100.0


def test_visual_clip_context_strictly_matches_reference_inside_context():
    intervals = _intervals()

    assert intervals["reference_source"] == {"start": 65.0, "end": 77.0}
    assert intervals["visual_clip_context"] == {"start": 5.0, "end": 17.0}
    assert intervals["visual_clip_relative"] == {"start": 0.0, "end": 12.0}
    assert "chart_context" not in intervals
    assert "proposed_clip_context" not in intervals


def test_full_sentence_overlap_keeps_complete_cross_boundary_sentence():
    intervals = _intervals()
    boundaries = [
        {"sentence_index": 1, "start_context_seconds": 3.8, "end_context_seconds": 18.2, "start_source_seconds": 63.8, "end_source_seconds": 78.2, "text": "complete sentence", "confidence": 0.93, "needs_review": False},
    ]
    words = [
        {"word": "before", "start": 3.8, "end": 4.2, "probability": 0.9},
        {"word": "inside", "start": 6.0, "end": 6.5, "probability": 0.9},
        {"word": "after", "start": 18.0, "end": 18.2, "probability": 0.9},
    ]

    selected = select_full_sentences_for_reference(boundaries, words, intervals)

    assert selected["selected_full_sentences"][0]["start_source"] == 63.8
    assert selected["selected_full_sentences"][0]["end_source"] == 78.2
    assert selected["selected_full_sentences"][0]["starts_before_visual_clip"] is True
    assert selected["selected_full_sentences"][0]["ends_after_visual_clip"] is True
    assert [row["word"] for row in selected["overlap_words"]] == ["inside"]


def test_non_overlapping_neighbor_sentence_is_not_selected():
    intervals = _intervals()
    boundaries = [
        {"sentence_index": 1, "start_context_seconds": 1.0, "end_context_seconds": 4.0, "start_source_seconds": 61.0, "end_source_seconds": 64.0, "text": "previous", "confidence": 0.9, "needs_review": False},
        {"sentence_index": 2, "start_context_seconds": 18.0, "end_context_seconds": 20.0, "start_source_seconds": 78.0, "end_source_seconds": 80.0, "text": "next", "confidence": 0.9, "needs_review": False},
    ]

    selected = select_full_sentences_for_reference(boundaries, [], intervals)

    assert selected["selected_full_sentences"] == []


def test_visual_subtitle_times_are_clamped_to_visual_duration():
    intervals = _intervals()
    words = [
        {"word": "start", "start": 4.8, "end": 5.2, "probability": 0.9},
        {"word": "end", "start": 16.8, "end": 17.4, "probability": 0.9},
    ]

    selected = select_full_sentences_for_reference([], words, intervals)
    subtitle = selected["visual_subtitle_segments"][0]

    assert subtitle["start"] == 0.0
    assert subtitle["end"] == 12.0


def test_missing_context_marks_narration_incomplete_without_forging_sentences():
    intervals = _intervals(requires_context_redownload=True)
    selected = select_full_sentences_for_reference([], [], intervals)

    assert selected["status"] == "incomplete_context"
    assert selected["selected_full_sentences"] == []


def test_qwen_rejects_context_inputs():
    try:
        assert_qwen_visual_inputs(["data/processed/bar_1/context/frame_000001.jpg"])
    except RuntimeError as exc:
        assert "visual_clip" in str(exc)
    else:
        raise AssertionError("context-derived Qwen input was accepted")


def test_qwen_accepts_visual_frame_inputs():
    assert_qwen_visual_inputs(["data/processed/bar_1/visual_frames/keyframe_candidates/keyframe_candidate_000001.jpg"])


def test_final_transcript_times_are_clip_relative():
    raw = {
        "segments": [{"start": 3.0, "end": 8.0, "text": "hello world"}],
        "words": [
            {"word": "hello", "start": 3.0, "end": 3.5},
            {"word": "world", "start": 7.5, "end": 8.0},
        ],
    }

    sliced = slice_transcript(raw, 2.0, 7.75)

    assert sliced["segments"][0]["start"] == 1.0
    assert sliced["segments"][0]["end"] == 5.75
    assert sliced["words"][0]["start"] == 1.0
    assert sliced["words"][-1]["end"] == 5.75


def test_build_sentence_boundaries_splits_on_pause():
    cfg = _cfg()
    words = [
        {"word": "a", "start": 0.2, "end": 0.4, "probability": 0.9},
        {"word": "b", "start": 1.2, "end": 1.4, "probability": 0.9},
    ]

    rows = build_sentence_boundaries([], words, cfg, context_duration_seconds=3.0)

    assert len(rows) == 2
    assert rows[0]["boundary_reason"] == "pause"
