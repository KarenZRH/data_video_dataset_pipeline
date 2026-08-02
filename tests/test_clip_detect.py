from datavideo.clip_detect import merge_positive_frames


def _frame(i, timestamp):
    return {"frame_id": f"f{i}", "timestamp": timestamp, "fps": 2, "path": f"f{i}.jpg"}


def _result(i, is_chart=True, confidence=0.9):
    return {"frame_id": f"f{i}", "result": {"is_chart": is_chart, "confidence": confidence}}


def test_merge_positive_frames_keeps_multiple_clips(tmp_path):
    frames = [_frame(0, 0.0), _frame(1, 0.5), _frame(2, 10.0), _frame(3, 10.5)]
    results = [_result(0), _result(1), _result(2), _result(3)]
    clips = merge_positive_frames(
        frames,
        results,
        expand=0.0,
        min_duration=0.1,
        duration=20.0,
        out_path=tmp_path / "clips.jsonl",
        min_positive_frames=2,
        min_confidence=0.5,
        max_gap_seconds=1.0,
    )
    assert len(clips) == 2
    assert clips[0]["start"] == 0.0
    assert clips[1]["start"] == 10.0


def test_merge_positive_frames_filters_singletons(tmp_path):
    frames = [_frame(0, 0.0), _frame(1, 5.0)]
    results = [_result(0), _result(1)]
    clips = merge_positive_frames(
        frames,
        results,
        expand=0.0,
        min_duration=0.1,
        duration=10.0,
        out_path=tmp_path / "clips.jsonl",
        min_positive_frames=2,
        min_confidence=0.5,
        max_gap_seconds=1.0,
    )
    assert clips == []
