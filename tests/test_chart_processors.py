from datavideo.chart_processors import (
    SUPPORTED_PROCESSORS,
    detect_chart_type,
    normalize_chart_type,
    resolve_processor,
)


def test_resolve_processor_maps_families():
    assert resolve_processor("bar") == "bar"
    assert resolve_processor("combined") == "bar"
    assert resolve_processor("line") == "line"
    assert resolve_processor("area") == "line"
    assert resolve_processor("timeline") == "line"
    assert resolve_processor("pie") == "pie"
    assert resolve_processor("donut") == "pie"
    assert resolve_processor("map") == "map"
    assert resolve_processor("sankey") == "sankey"
    assert resolve_processor("something-else") == "unknown"
    assert resolve_processor(None) == "unknown"


def test_bar_and_line_processors_are_supported():
    assert SUPPORTED_PROCESSORS == {"bar", "line"}


def test_detect_chart_type_prefers_recovered_and_flags_mismatch():
    processor, declared, consistent = detect_chart_type("bar", "bar")
    assert processor == "bar"
    assert consistent is True

    # Recovered type is authoritative for dispatch.
    processor, declared, consistent = detect_chart_type("bar", "line")
    assert processor == "line"
    assert consistent is False

    # Unknown recovered type falls back to the declaration.
    processor, declared, consistent = detect_chart_type("line", None)
    assert processor == "line"
    assert consistent is True


def test_normalize_chart_type():
    assert normalize_chart_type(" Bar ") == "bar"
    assert normalize_chart_type("") == ""
    assert normalize_chart_type(None) == ""
