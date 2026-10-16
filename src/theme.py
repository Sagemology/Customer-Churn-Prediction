"""Theme tokens and CSS for the Streamlit dashboard.

System mode uses CSS ``prefers-color-scheme`` for Streamlit-owned UI. Plotly
figures are built on the server and use the light palette in System mode; see
README for this Streamlit boundary and fallback.
"""

THEME_OPTIONS = ("Light", "Dark", "System")

PALETTES = {
    "light": {
        "page": "#f2f3f1", "surface": "#ffffff", "surface_alt": "#f7f9f8",
        "text": "#252d32", "muted": "#536068", "subtle": "#68747b",
        "border": "#d8dfdc", "grid": "#e6ebe9", "axis": "#617078",
        "plot": "#ffffff", "hover": "#ffffff", "hover_text": "#252d32",
        "focus": "#126b7a", "accent": "#126b7a", "heat_low": "#e7f0f2",
        "heat_high": "#286a7c", "heat_text": "#ffffff", "heat_text_dark": "#243139",
        "chart_frame": "#ffffff",
    },
    "dark": {
        "page": "#171c22", "surface": "#222a32", "surface_alt": "#29333d",
        "text": "#f2f5f6", "muted": "#c3cdd2", "subtle": "#aab6bd",
        "border": "#3a4650", "grid": "#3a4650", "axis": "#c0cbd1",
        "plot": "#222a32", "hover": "#303b45", "hover_text": "#f5f7f8",
        "focus": "#73c7d2", "accent": "#73c7d2", "heat_low": "#26333d",
        "heat_high": "#327b8a", "heat_text": "#ffffff", "heat_text_dark": "#ffffff",
        "chart_frame": "#222a32",
    },
}


def initialize_theme_state(session_state) -> str:
    """Set a first-run default without replacing a user's session choice."""
    session_state.setdefault("theme_mode", "System")
    return session_state["theme_mode"]


def normalize_theme(selection: str) -> str:
    """Validate a selector value and return its lowercase representation."""
    if selection not in THEME_OPTIONS:
        raise ValueError(f"Theme must be one of: {', '.join(THEME_OPTIONS)}")
    return selection.lower()


def plotly_mode(selection: str) -> str:
    """Return the explicit Plotly palette; System uses its documented light fallback."""
    mode = normalize_theme(selection)
    return "dark" if mode == "dark" else "light"


def style_plotly_figure(fig, selection: str):
    """Apply explicit, contrast-aware Plotly styling to every chart type."""
    mode = plotly_mode(selection)
    palette = PALETTES[mode]
    fig.update_layout(
        margin=dict(l=18, r=18, t=58, b=22),
        paper_bgcolor=palette["plot"], plot_bgcolor=palette["plot"],
        font=dict(family="Inter, system-ui, sans-serif", color=palette["text"], size=12),
        title=dict(font=dict(color=palette["text"], size=16)),
        legend=dict(font=dict(color=palette["text"], size=11),
                    title=dict(font=dict(color=palette["text"])), bgcolor=palette["plot"],
                    bordercolor=palette["border"], borderwidth=0),
        hoverlabel=dict(bgcolor=palette["hover"], bordercolor=palette["border"],
                        font=dict(color=palette["hover_text"], size=12)),
    )
    fig.update_xaxes(showgrid=False, linecolor=palette["border"],
                     tickfont=dict(color=palette["axis"]), title_font=dict(color=palette["text"]),
                     zerolinecolor=palette["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=palette["grid"], zeroline=False,
                     linecolor=palette["border"], tickfont=dict(color=palette["axis"]),
                     title_font=dict(color=palette["text"]))
    return fig


def _css_vars(palette: dict[str, str]) -> str:
    mapping = {
        "page": "page-bg", "surface": "surface", "surface_alt": "surface-alt",
        "text": "text", "muted": "muted", "subtle": "subtle", "border": "border",
        "grid": "grid", "focus": "focus", "accent": "accent", "chart_frame": "chart-frame",
    }
    return "\n".join(f"--{token}: {palette[key]};" for key, token in mapping.items())


def theme_css(selection: str) -> str:
    """Build CSS using shared tokens; System follows OS changes without JS."""
    mode = normalize_theme(selection)
    if mode == "dark":
        root = f":root {{\n{_css_vars(PALETTES['dark'])}\n}}"
        system_note = ""
    elif mode == "system":
        root = f":root {{\n{_css_vars(PALETTES['light'])}\n}}"
        system_note = f"@media (prefers-color-scheme: dark) {{ :root {{\n{_css_vars(PALETTES['dark'])}\n--chart-frame:#ffffff;\n}} }}"
    else:
        root = f":root {{\n{_css_vars(PALETTES['light'])}\n}}"
        system_note = ""
    return f"""
    <style>
    :root {{ color-scheme:{'light dark' if mode == 'system' else mode}; }}
    {root}
    {system_note}
    html, body, .stApp, [data-testid="stAppViewContainer"] {{
        background:var(--page-bg) !important; color:var(--text) !important;
    }}
    [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background:var(--page-bg) !important; color:var(--text) !important;
    }}
    [data-testid="stHeader"] button, [data-testid="stToolbar"] button {{ color:var(--text) !important; }}
    [data-testid="stSidebar"] {{
        background:var(--surface) !important; border-right:1px solid var(--border);
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] small {{ color:var(--text) !important; }}
    .block-container {{ max-width:1400px; padding:1.55rem 2rem 2.6rem; }}
    .dash-title {{ margin:0 0 1rem; }}
    .dash-title h1 {{ color:var(--text); font-size:1.85rem; line-height:1.22; margin:0 0 .25rem; letter-spacing:-.025em; }}
    .dash-title p {{ color:var(--muted); font-size:.96rem; margin:0; }}
    [data-testid="stMetric"] {{ background:var(--surface) !important; border:1px solid var(--border) !important; border-radius:9px; padding:13px 16px; min-height:99px; }}
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{ color:var(--muted) !important; font-size:.83rem !important; font-weight:550 !important; }}
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] div {{ color:var(--text) !important; font-size:1.55rem !important; font-weight:650 !important; }}
    [data-testid="stTabs"] [role="tab"] {{ color:var(--muted) !important; border-bottom-color:transparent !important; }}
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{ color:var(--accent) !important; border-bottom-color:var(--accent) !important; }}
    [data-testid="stTabs"] [role="tab"]:focus-visible, button:focus-visible, input:focus-visible,
    textarea:focus-visible, [role="combobox"]:focus-visible {{ outline:3px solid var(--focus) !important; outline-offset:2px !important; }}
    [data-testid="stForm"], [data-testid="stExpander"] {{ background:var(--surface); border:1px solid var(--border); border-radius:9px; }}
    [data-testid="stForm"] {{ padding:1rem; }}
    [data-testid="stPlotlyChart"] {{ background:var(--chart-frame); border:1px solid var(--border); border-radius:9px; padding:.2rem; overflow:hidden; }}
    [data-testid="stDataFrame"] {{ background:var(--surface); border:1px solid var(--border); }}
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"] {{ color:var(--muted); }}
    code, pre {{ color:var(--text) !important; background:var(--surface-alt) !important; border-color:var(--border) !important; }}
    label, [data-testid="stWidgetLabel"] p {{ color:var(--text) !important; }}
    [data-baseweb="input"] > div, [data-baseweb="select"] > div,
    [data-baseweb="textarea"] > div, textarea, input {{
        background:var(--surface-alt) !important; color:var(--text) !important;
        border-color:var(--border) !important;
    }}
    [data-baseweb="select"] *, [data-baseweb="input"] *, [data-baseweb="textarea"] * {{ color:var(--text) !important; }}
    [data-baseweb="popover"], [role="listbox"], [role="option"], [role="menu"],
    [data-testid="stTooltipContent"], [data-testid="stPopover"] {{
        background:var(--surface) !important; color:var(--text) !important; border-color:var(--border) !important;
    }}
    [role="listbox"] *, [role="option"] *, [role="menu"] *, [data-testid="stTooltipContent"] * {{ color:var(--text) !important; }}
    [role="option"]:hover, [role="option"][aria-selected="true"] {{ background:var(--surface-alt) !important; }}
    [data-testid="stButton"] button {{ color:var(--text); border-color:var(--border); background:var(--surface); }}
    [data-testid="stButton"] button[kind="primary"] {{ color:#ffffff; background:#126b7a; border-color:#126b7a; }}
    [data-testid="stAlert"] {{ background:var(--surface-alt); color:var(--text); border-color:var(--border); }}
    [data-testid="stAlert"] p, [data-testid="stAlert"] span {{ color:var(--text) !important; }}
    .small-note {{ color:var(--muted); font-size:.86rem; margin:.05rem 0 .5rem; }}
    @media (max-width:700px) {{
        .block-container {{ padding:1.1rem 1rem 2rem; }}
        .dash-title h1 {{ font-size:1.55rem; }}
        [data-testid="stMetric"] {{ min-height:88px; padding:10px 12px; }}
        [data-testid="stMetricValue"] {{ font-size:1.25rem !important; }}
    }}
    </style>
    """
