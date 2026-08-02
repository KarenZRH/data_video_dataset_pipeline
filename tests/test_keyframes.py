from datavideo.keyframes import extract_scored_keyframe
from datavideo.schemas import write_json


class FakeClient:
    def __init__(self, scores):
        self.scores = scores

    def score_keyframe_candidate(self, image_path, chart_identity):
        name = image_path.rsplit("/", 1)[-1]
        return {"result": self.scores[name], "raw_response": None, "model_status": "qwen", "failure_reason": None}


def test_scored_keyframe_reuses_complete_cached_manifest(monkeypatch, tmp_path):
    initial = tmp_path / "initial.png"
    initial.write_bytes(b"png")
    cached = {
        "clip_id": "bar_final_000",
        "assets": {"initial": str(initial)},
        "timestamps": {"initial": 10.0},
    }
    write_json(tmp_path / "keyframe_manifest.json", cached)
    monkeypatch.setattr(
        "datavideo.keyframes.extract_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should use cache")),
    )

    result = extract_scored_keyframe(
        "video.mp4",
        {"clip_id": "bar_final_000", "start": 10.0, "end": 11.0},
        tmp_path,
        {"sampling": {"short_side": 768}, "model": {"env_var": "MODEL_PATH"}},
        client=FakeClient({}),
    )

    assert result == cached


def _score(
    *,
    same_chart=True,
    scene_change=False,
    pre_change=True,
    post_change_state=False,
    complete_initial_chart=True,
    all_target_categories_visible=True,
    completeness=0.9,
    staticness=0.9,
    chart_identity_consistency=1.0,
    initial_state_representative=0.9,
    motion_score=0.05,
):
    return {
        "same_chart": same_chart,
        "scene_change": scene_change,
        "pre_change": pre_change,
        "post_change_state": post_change_state,
        "complete_initial_chart": complete_initial_chart,
        "all_target_categories_visible": all_target_categories_visible,
        "completeness": completeness,
        "staticness": staticness,
        "chart_identity_consistency": chart_identity_consistency,
        "initial_state_representative": initial_state_representative,
        "motion_score": motion_score,
    }


def test_scored_keyframe_prefers_complete_initial_same_chart(monkeypatch, tmp_path):
    calls = []
    frames = [
        {"frame_id": "f0", "path": "frames/f0.jpg", "timestamp": 10.0, "fps": 4, "sample_type": "keyframe_candidate"},
        {"frame_id": "f1", "path": "frames/f1.jpg", "timestamp": 10.25, "fps": 4, "sample_type": "keyframe_candidate"},
        {"frame_id": "f2", "path": "frames/f2.jpg", "timestamp": 10.5, "fps": 4, "sample_type": "keyframe_candidate"},
    ]
    scores = {
        "f0.jpg": _score(complete_initial_chart=False, all_target_categories_visible=False, completeness=0.4, initial_state_representative=0.4, motion_score=0.1),
        "f1.jpg": _score(),
        "f2.jpg": _score(same_chart=False, scene_change=True, pre_change=False, chart_identity_consistency=0.0, motion_score=0.0),
    }

    def fake_extract_still(video, timestamp, out, force=False):
        calls.append(timestamp)
        return out

    monkeypatch.setattr("datavideo.keyframes.extract_frames", lambda *args, **kwargs: frames)
    monkeypatch.setattr("datavideo.keyframes._image_motion_scores", lambda rows: {row["frame_id"]: 0.0 for row in rows})
    monkeypatch.setattr("datavideo.keyframes.extract_still", fake_extract_still)

    manifest = extract_scored_keyframe(
        "video.mp4",
        {
            "clip_id": "bar_final_000",
            "start": 10.0,
            "end": 11.0,
            "chart_identities": ["sales by region"],
        },
        tmp_path,
        {"sampling": {"short_side": 768}, "keyframes": {"sample_fps": 4}, "model": {"env_var": "MODEL_PATH"}},
        client=FakeClient(scores),
    )

    assert calls == [10.25]
    assert manifest["assets"]["initial"].endswith("initial.png")
    assert manifest["source_frame_id"] == "f1"
    assert manifest["selection_method"] == "sampled_frame_priority_initial_seed_keyframe"


def test_scored_keyframe_uses_earliest_equivalent_initial_frame(monkeypatch, tmp_path):
    calls = []
    frames = [
        {"frame_id": "f0", "path": "frames/f0.jpg", "timestamp": 10.0, "fps": 4, "sample_type": "keyframe_candidate"},
        {"frame_id": "f1", "path": "frames/f1.jpg", "timestamp": 10.25, "fps": 4, "sample_type": "keyframe_candidate"},
    ]
    complete_score = _score()

    def fake_extract_still(video, timestamp, out, force=False):
        calls.append(timestamp)
        return out

    monkeypatch.setattr("datavideo.keyframes.extract_frames", lambda *args, **kwargs: frames)
    monkeypatch.setattr("datavideo.keyframes._image_motion_scores", lambda rows: {row["frame_id"]: 0.0 for row in rows})
    monkeypatch.setattr("datavideo.keyframes.extract_still", fake_extract_still)

    manifest = extract_scored_keyframe(
        "video.mp4",
        {"clip_id": "bar_final_001", "start": 10.0, "end": 11.0},
        tmp_path,
        {"sampling": {"short_side": 768}, "keyframes": {"sample_fps": 4}, "model": {"env_var": "MODEL_PATH"}},
        client=FakeClient({"f0.jpg": complete_score, "f1.jpg": complete_score}),
    )

    assert calls == [10.0]
    assert manifest["source_frame_id"] == "f0"


def test_scored_keyframe_prefers_pre_change_over_stable_post_change(monkeypatch, tmp_path):
    calls = []
    frames = [
        {"frame_id": "f0", "path": "frames/f0.jpg", "timestamp": 10.0, "fps": 4, "sample_type": "keyframe_candidate"},
        {"frame_id": "f1", "path": "frames/f1.jpg", "timestamp": 10.25, "fps": 4, "sample_type": "keyframe_candidate"},
    ]
    scores = {
        "f0.jpg": _score(staticness=0.7, motion_score=0.2),
        "f1.jpg": _score(
            pre_change=False,
            post_change_state=True,
            completeness=1.0,
            staticness=1.0,
            initial_state_representative=0.2,
            motion_score=0.0,
        ),
    }

    def fake_extract_still(video, timestamp, out, force=False):
        calls.append(timestamp)
        return out

    monkeypatch.setattr("datavideo.keyframes.extract_frames", lambda *args, **kwargs: frames)
    monkeypatch.setattr("datavideo.keyframes._image_motion_scores", lambda rows: {row["frame_id"]: 0.0 for row in rows})
    monkeypatch.setattr("datavideo.keyframes.extract_still", fake_extract_still)

    manifest = extract_scored_keyframe(
        "video.mp4",
        {"clip_id": "bar_final_002", "start": 10.0, "end": 11.0},
        tmp_path,
        {"sampling": {"short_side": 768}, "keyframes": {"sample_fps": 4}, "model": {"env_var": "MODEL_PATH"}},
        client=FakeClient(scores),
    )

    assert calls == [10.0]
    assert manifest["source_frame_id"] == "f0"
