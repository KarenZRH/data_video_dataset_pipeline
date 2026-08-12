from datavideo.chart_processors import detect_chart_type, normalize_chart_type, resolve_processor


def test_detect_chart_type_prefers_line_family():
    processor, declared, consistent = detect_chart_type("line", "line")
    assert processor == "line"
    assert declared == "line"
    assert consistent is True


def test_resolve_processor_maps_timeline_to_line():
    assert resolve_processor("timeline") == "line"
    assert normalize_chart_type(" Line ") == "line"
