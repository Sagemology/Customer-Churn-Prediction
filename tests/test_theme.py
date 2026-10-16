import plotly.graph_objects as go
import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest

from src.theme import (PALETTES, THEME_OPTIONS, initialize_theme_state, normalize_theme,
                       plotly_mode, style_plotly_figure, theme_css)


def _luminance(hex_color):
    channels = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4 for value in channels]
    return .2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2]


def _contrast(first, second):
    light, dark = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (light + .05) / (dark + .05)


def test_theme_selector_values_and_validation():
    assert THEME_OPTIONS == ("Light", "Dark", "System")
    assert normalize_theme("Dark") == "dark"
    assert plotly_mode("System") == "light"
    with pytest.raises(ValueError):
        normalize_theme("Blue")


def test_theme_selection_survives_streamlit_reruns():
    session = {}
    assert initialize_theme_state(session) == "System"
    session["theme_mode"] = "Dark"
    assert initialize_theme_state(session) == "Dark"


@pytest.mark.parametrize("selection", THEME_OPTIONS)
def test_dashboard_switches_theme_without_script_errors(selection):
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30).run()
    app.selectbox(key="theme_mode").set_value(selection).run()
    assert app.selectbox(key="theme_mode").value == selection
    assert not app.exception


def test_system_stylesheet_tracks_os_preference_and_documents_plot_fallback():
    css = theme_css("System")
    assert "prefers-color-scheme: dark" in css
    assert "--page-bg: #171c22" in css
    assert "--chart-frame:#ffffff" in css


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_theme_text_tokens_have_wcag_aa_contrast(mode):
    palette = PALETTES[mode]
    for text_token in ("text", "muted"):
        assert _contrast(palette[text_token], palette["surface"]) >= 4.5
        assert _contrast(palette[text_token], palette["page"]) >= 4.5


@pytest.mark.parametrize("selection,mode", [("Light", "light"), ("Dark", "dark"), ("System", "light")])
def test_plotly_style_sets_explicit_chart_text_and_surfaces(selection, mode):
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], name="Example"))
    style_plotly_figure(fig, selection)
    palette = PALETTES[mode]
    assert fig.layout.paper_bgcolor == palette["plot"]
    assert fig.layout.plot_bgcolor == palette["plot"]
    assert fig.layout.font.color == palette["text"]
    assert fig.layout.title.font.color == palette["text"]
    assert fig.layout.legend.font.color == palette["text"]
    assert fig.layout.xaxis.tickfont.color == palette["axis"]
    assert fig.layout.yaxis.gridcolor == palette["grid"]
    assert fig.layout.hoverlabel.font.color == palette["hover_text"]
