"""
stab_views.py - interactive views for the TVM4173 stabilisation notebook.

Each function shows one interactive element of the notebook (figure,
sliders, self-check). The chemistry is in carbonate.py; this file only
draws and connects widgets.

Two drawing back-ends:
* Plotly (default when plotly and anywidget are installed): hover over
  points and lines to read values, drag to zoom, double-click to reset,
  camera icon to save a PNG.
* matplotlib: static figures that are redrawn when a slider is released.
Switch in the notebook with  sv.USE_PLOTLY = False.
"""
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML
import carbonate as cb


def _plotly_available():
    try:
        import plotly.graph_objects  # noqa: F401
        import anywidget             # noqa: F401  (needed for FigureWidget)
        return True
    except ImportError:
        return False


USE_PLOTLY = _plotly_available()

WATERS = cb.example_waters()
STYLE = {"description_width": "200px"}
WIDE = widgets.Layout(width="520px")
PH_VALUES = (5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 10.0)
STEP_COLOR = "#1f4e9a"
GREEN, RED = "#43a047", "#e53935"          # readable on light and dark backgrounds

# Theme. Boxes and cards use transparent colours that work in light and dark
# JupyterLab themes. Figures cannot detect the theme: set sv.DARK = True in the
# first cell of a chapter for dark figures.
DARK = False
BORDER = "rgba(128,128,128,0.45)"
ACCENT = "#1f77b4"
_THEME = {"template": ("simple_white", "plotly_dark"), "ink": ("black", "#f2f2f2"),
          "muted": ("#666666", "#cccccc"), "faint": ("#bbbbbb", "#8a8a8a"),
          "grid": ("#c4c4c4", "#8a8a8a"), "shade": ("rgba(0,0,0,0.06)", "rgba(255,255,255,0.12)"),
          "legend": ("rgba(255,255,255,0.85)", "rgba(43,43,43,0.9)"), "step": ("#1f4e9a", "#64b5f6"),
          "target": ("rgba(46,125,50,0.28)", "rgba(102,187,106,0.65)")}
DARK_BG, DARK_TEXT, DARK_AXIS, DARK_GRIDLINE = "#2b2b2b", "#e6e6e6", "#bbbbbb", "#444444"


def _style(fig):
    """Dark mode: dark-grey figure background, light text and axes (light mode unchanged)."""
    if not DARK:
        return
    ax = dict(linecolor=DARK_AXIS, tickcolor=DARK_AXIS, gridcolor=DARK_GRIDLINE, zerolinecolor=DARK_GRIDLINE)
    fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG, font=dict(color=DARK_TEXT),
                      xaxis=ax, yaxis=ax)


def T(key):
    """Figure colour for the current theme (light or dark)."""
    return _THEME[key][1 if DARK else 0]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def nz(x, digits):
    """Round and avoid '-0.0' in tables."""
    return round(float(x), digits) + 0.0


def slider(value, vmin, vmax, step, text):
    return widgets.FloatSlider(value=value, min=vmin, max=vmax, step=step,
                               description=text, style=STYLE, layout=WIDE,
                               continuous_update=False)


def table(waters):
    """HTML table of all treatment steps."""
    rows = []
    for i, w in enumerate(waters):
        rows.append({"#": i, "step": w.label, "pH": f"{w.pH:.2f}",
                     "Alk (mmol/L)": f"{w.alk_mmol:.2f}", "C<sub>T</sub> (mmol/L)": f"{w.ct_mmol:.2f}",
                     "Ca (mg/L)": f"{w.ca_mg:.1f}", "free CO<sub>2</sub> (mg/L)": f"{w.co2_mg:.1f}",
                     "SI (–)": f"{nz(w.SI, 2):+.2f}", "CCPP (mg CaCO<sub>3</sub>/L)": f"{nz(w.CCPP(), 1):+.1f}"})
    legend = ("<div style='font-size:0.85em;opacity:0.85'>Alk = alkalinity · C<sub>T</sub> = total inorganic "
              "carbon (CO<sub>2</sub>* + HCO<sub>3</sub><sup>−</sup> + CO<sub>3</sub><sup>2−</sup>) · "
              "SI = saturation index (&lt; 0: dissolves CaCO<sub>3</sub>) · CCPP = calcium carbonate precipitation "
              "potential (+: can precipitate, −: can still dissolve)</div>")
    return HTML(pd.DataFrame(rows).to_html(index=False, escape=False) + legend)


def target_check(w):
    """Compare a water with the NV 274/2022 targets (slide 'Target water quality')."""
    def mark(ok):
        return f"<b style='color:{GREEN}'>✓</b>" if ok else f"<b style='color:{RED}'>✗</b>"
    (p0, p1), (a0, a1), (c0, c1) = cb.TARGETS["pH"], cb.TARGETS["Alk"], cb.TARGETS["Ca"]
    items = [(f"pH {w.pH:.2f}", p0 <= w.pH <= p1, f"{p0}–{p1}"),
             (f"Alk {w.alk_mmol:.2f} mmol/L", a0 <= w.alk_mmol <= a1, f"{a0}–{a1}"),
             (f"Ca {w.ca_mg:.1f} mg/L", c0 <= w.ca_mg <= c1, f"{c0:g}–{c1:g}"),
             (f"SI {nz(w.SI, 2):+.2f}", w.SI >= -0.1, "≈ 0 or slightly positive (here: ≥ −0.1)")]
    cells = "".join(f"<td style='padding:2px 12px'>{mark(ok)} {txt}<br><small>target {t}</small></td>"
                    for txt, ok, t in items)
    return HTML(f"<b>Final water vs. target (NV 274/2022):</b><table><tr>{cells}</tr></table>")


_CHECK_N = [0]          # numbering of "Check yourself" boxes; reset by setup()


def _check_frame(children):
    """Frame for a self-check: numbered title, accent stripe on the left."""
    _CHECK_N[0] += 1
    title = widgets.HTML(f"<div style='font-weight:600;font-size:1.05em;margin-bottom:4px'>"
                         f"✎ Check yourself {_CHECK_N[0]}</div>")
    return widgets.VBox([title] + list(children), layout=widgets.Layout(
        border=f"1px solid {BORDER}", border_left=f"5px solid {ACCENT}",
        padding="10px 14px", margin="10px 0 10px 0"))


def choice_check(prompt, options, correct, right_text, wrong_text):
    """A multiple-choice self-check with feedback."""
    dd = widgets.Dropdown(options=["– choose –"] + options, layout=widgets.Layout(width="320px"))
    btn = widgets.Button(description="Check", button_style="info")
    out = widgets.Output()

    def on_click(_):
        out.clear_output()
        with out:
            if dd.value == "– choose –":
                display(HTML("Choose an answer first."))
            elif dd.value == correct:
                display(HTML(f"<b style='color:{GREEN}'>✓ Correct.</b> " + right_text))
            else:
                display(HTML(f"<b style='color:{RED}'>✗ Not quite.</b> "
                             + wrong_text.get(dd.value, "") + " Try again."))
    btn.on_click(on_click)
    display(_check_frame([widgets.HTMLMath(value=prompt), widgets.HBox([dd, btn]), out]))


def show_answer(question, answer_html):
    """A question with a 'Show answer' button (answer hidden until clicked)."""
    ans = widgets.HTMLMath(value=f"<div style='border-top:1px solid {BORDER};padding-top:6px'>"
                                 f"{answer_html}</div>", layout=widgets.Layout(display="none"))
    btn = widgets.Button(description="Show answer", button_style="")

    def toggle(_):
        hidden = ans.layout.display == "none"
        ans.layout.display = "block" if hidden else "none"
        btn.description = "Hide answer" if hidden else "Show answer"
    btn.on_click(toggle)
    display(_check_frame([widgets.HTMLMath(value=question), btn, ans]))


def axis_limits(waters, ct_min=2.0, alk_min=1.5):
    ct = max([ct_min] + [1.12 * w.ct_mmol for w in waters])
    alk = max([alk_min] + [1.12 * w.alk_mmol for w in waters])
    return np.ceil(ct * 4) / 4, np.ceil(alk * 4) / 4


def _hover_row(w):
    return [w.pH, w.alk_mmol, w.ct_mmol, w.ca_mg, nz(w.SI, 2), w.co2_mg, w.label or w.name]


HOVER = ("<b>%{customdata[6]}</b><br>pH %{customdata[0]:.2f}"
         "<br>Alk %{customdata[1]:.2f} mmol/L<br>C<sub>T</sub> %{customdata[2]:.2f} mmol/L"
         "<br>Ca %{customdata[3]:.1f} mg/L<br>free CO<sub>2</sub> %{customdata[5]:.1f} mg/L"
         "<br>SI %{customdata[4]:+.2f}<extra></extra>")


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------
class DeffeyesPlotly:
    """A Deffeyes diagram as a Plotly FigureWidget that is updated in place.

    Fixed traces: one per pH line, n_sat calcite-saturation lines, and one
    trace with the water points (hover shows pH, Alk, C_T, Ca, SI).
    Arrows, labels and the target zone are layout annotations/shapes.
    """

    def __init__(self, n_sat=2, height=560, width=760):
        import plotly.graph_objects as go
        self.fig = go.FigureWidget()
        for p in PH_VALUES:
            self.fig.add_scatter(x=[], y=[], mode="lines", name=f"pH {p:g}",
                                 line=dict(color=T("grid"), width=1), showlegend=False,
                                 hovertemplate=f"pH {p:g}<extra></extra>")
        self.n_sat = n_sat
        for k in range(n_sat):
            self.fig.add_scatter(x=[], y=[], mode="lines", name="", visible=False,
                                 line=dict(color=cb.SAT_COLORS[k % 4], width=2, dash="dash"))
        self.fig.add_scatter(x=[], y=[], mode="markers+text", name="waters", showlegend=False,
                             textposition="bottom right", hovertemplate=HOVER,
                             marker=dict(size=8, color=T("ink")))
        self.fig.update_layout(
            template=T("template"), height=height, width=width,
            margin=dict(l=70, r=80, t=100, b=60), hovermode="closest",
            title=dict(text="", x=0.5, xanchor="center"),
            xaxis=dict(title=dict(text="Total inorganic carbon C<sub>T</sub> (mmol/L)")),
            yaxis=dict(title=dict(text="Alkalinity (mmol/L = meq/L)")),
            legend=dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom",
                        bgcolor=T("legend")))

        _style(self.fig)

    def update(self, temp_C, I, ct_max, alk_max, ca_sat=(), steps=None, arrows=None,
               points=None, title="", target=True, activity=True, start_label="raw"):
        """steps: list of waters joined by arrows (treatment path)
        arrows: list of (w_from, w_to, color, text) drawn from one point
        points: list of (water, text) shown as labelled points"""
        ct = np.linspace(1e-4, ct_max, 300)
        ann, shapes, top = [], [], []
        n = len(PH_VALUES)
        with self.fig.batch_update():
            # pH lines and their labels
            for i, p in enumerate(PH_VALUES):
                alk = cb.pH_line(p, ct, temp_C, I, activity)
                ok = (alk >= -0.05) & (alk <= alk_max)
                self.fig.data[i].x = ct[ok]
                self.fig.data[i].y = alk[ok]
                if not ok.any():
                    continue
                j = np.where(ok)[0][-1]
                if alk[j] >= 0.98 * alk_max:
                    top.append((ct[j], p))
                else:
                    ann.append(dict(x=1.0, xref="paper", y=float(alk[j]), yref="y",
                                    text=f"pH {p:g}", showarrow=False, xanchor="left",
                                    xshift=4, font=dict(size=10, color=T("muted"))))
            top.sort()
            xs = [x for x, _ in top]
            for k in range(1, len(xs)):
                xs[k] = max(xs[k], xs[k - 1] + 0.03 * ct_max)
            for (_, p), x in zip(top, xs):
                ann.append(dict(x=float(x), xref="x", y=1.0, yref="paper", text=f"pH {p:g}",
                                textangle=-90, showarrow=False, yanchor="bottom", yshift=3,
                                font=dict(size=10, color=T("muted"))))
            # target zone
            if target:
                (p_lo, p_hi), (a_lo, a_hi) = cb.TARGETS["pH"], cb.TARGETS["Alk"]
                K = cb._Consts(temp_C, I, activity)
                c = [(1000 * cb.CT_from_pH_alk(p, a / 1000, K), a)
                     for p, a in ((p_lo, a_lo), (p_hi, a_lo), (p_hi, a_hi), (p_lo, a_hi))]
                path = "M " + " L ".join(f"{x:.4f},{y:.4f}" for x, y in c) + " Z"
                shapes.append(dict(type="path", path=path, xref="x", yref="y", layer="below",
                                   fillcolor=T("target"), line=dict(color=GREEN, width=1)))
                if c[3][0] < ct_max and c[3][1] < alk_max:
                    ann.append(dict(x=c[3][0], y=c[3][1], xref="x", yref="y",
                                    text="target zone<br>pH 8.0–8.5, Alk 0.6–1.0<br>(NV 274/2022)",
                                    showarrow=True, arrowhead=0, arrowcolor=GREEN, ax=-70, ay=-40,
                                    align="right", font=dict(size=10, color=GREEN)))
            # saturation lines
            for k in range(self.n_sat):
                tr = self.fig.data[n + k]
                if k < len(ca_sat):
                    ca = ca_sat[k]
                    alk = cb.saturation_line(ca, ct[::2], temp_C, I, activity)
                    alk[alk > alk_max] = np.nan
                    tr.x, tr.y = ct[::2], alk
                    tr.name = f"calcite saturation, Ca {ca:.3g} mg/L"
                    tr.hovertemplate = f"saturated with calcite at Ca {ca:.3g} mg/L<extra></extra>"
                    tr.visible = True
                else:
                    tr.visible = False
            # waters: path, arrows, points
            pts, txt, size, color, symbol = [], [], [], [], []
            if steps:
                for i, w in enumerate(steps):
                    pts.append(w); txt.append(start_label if i == 0 else "")
                    last = i == len(steps) - 1 and i > 0
                    size.append(10 if (i == 0 or last) else 6)
                    color.append(T("ink") if i == 0 else T("step"))
                    symbol.append("square" if last else "circle")
                for i in range(1, len(steps)):
                    a, b = steps[i - 1], steps[i]
                    if abs(a.ct_mmol - b.ct_mmol) < 1e-9 and abs(a.alk_mmol - b.alk_mmol) < 1e-9:
                        continue
                    ann.append(_arrow(a, b, T("step")))
                    ann.append(dict(x=(a.ct_mmol + b.ct_mmol) / 2, y=(a.alk_mmol + b.alk_mmol) / 2,
                                    xref="x", yref="y", text=f"<b>{i}</b>", showarrow=False,
                                    xshift=9, yshift=7, font=dict(size=11, color=T("step"))))
            for a, b, col, text in (arrows or []):
                ann.append(_arrow(a, b, col))
                left = b.ct_mmol < a.ct_mmol - 1e-9          # arrow points left
                ann.append(dict(x=b.ct_mmol, y=b.alk_mmol, xref="x", yref="y", text=text,
                                showarrow=False, xanchor="right" if left else "left",
                                xshift=-8 if left else 6, yshift=-12 if left else 8,
                                font=dict(size=11, color=col)))
                pts.append(b); txt.append(""); size.append(5); color.append(col); symbol.append("circle")
            if arrows:
                pts.append(arrows[0][0]); txt.append("start"); size.append(10)
                color.append(T("ink")); symbol.append("circle")
            for w, text in (points or []):
                pts.append(w); txt.append(text); size.append(10); color.append(T("ink")); symbol.append("circle")
            tr = self.fig.data[-1]
            tr.x = [w.ct_mmol for w in pts]
            tr.y = [w.alk_mmol for w in pts]
            tr.customdata = [_hover_row(w) for w in pts]
            tr.text = txt
            tr.marker.size = size
            tr.marker.color = color
            tr.marker.symbol = symbol
            # layout
            self.fig.layout.annotations = ann
            self.fig.layout.shapes = shapes
            self.fig.layout.xaxis.range = [0, ct_max]
            self.fig.layout.yaxis.range = [0, alk_max]
            self.fig.layout.title.text = title or f"Deffeyes diagram, {temp_C:g} °C"
        return self.fig


def _arrow(a, b, color):
    return dict(x=b.ct_mmol, y=b.alk_mmol, ax=a.ct_mmol, ay=a.alk_mmol,
                xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True,
                arrowhead=3, arrowsize=1.2, arrowwidth=2.2, arrowcolor=color, standoff=4)


class SpeciesPlotly:
    """Fractions of CO2*, HCO3-, CO3-- versus pH (FigureWidget)."""

    def __init__(self):
        import plotly.graph_objects as go
        self.fig = go.FigureWidget()
        for name, col in (("CO<sub>2</sub>* (free CO<sub>2</sub>)", "#1f77b4"),
                          ("HCO<sub>3</sub><sup>−</sup>", "#ff7f0e"),
                          ("CO<sub>3</sub><sup>2−</sup>", "#2ca02c")):
            self.fig.add_scatter(x=[], y=[], mode="lines", name=name, line=dict(color=col, width=2.5),
                                 hovertemplate="pH %{x:.2f}: %{y:.1%}<extra>" + name + "</extra>")
        self.fig.update_layout(
            template=T("template"), height=400, width=760, hovermode="x unified",
            margin=dict(l=70, r=30, t=60, b=55), title=dict(text="", x=0.5, xanchor="center"),
            xaxis=dict(title=dict(text="pH"), range=[4, 12]),
            yaxis=dict(title=dict(text="fraction of C<sub>T</sub>"), range=[0, 1.08], tickformat=".0%"),
            legend=dict(x=0.02, y=0.85, xanchor="left", yanchor="top", bgcolor=T("legend")))

        _style(self.fig)

    def update(self, temp_C):
        K = cb._Consts(temp_C, 0, activity=False)
        pH = np.linspace(4, 12, 321)
        fr = cb.alphas(pH, K)
        pK1, pK2 = -np.log10(K.K1), -np.log10(K.K2)
        with self.fig.batch_update():
            for tr, a in zip(self.fig.data, fr):
                tr.x, tr.y = pH, a
            self.fig.layout.shapes = [
                dict(type="rect", x0=6.5, x1=9.5, y0=0, y1=1, xref="x", yref="paper",
                     fillcolor=T("shade"), line=dict(width=0), layer="below")] + [
                dict(type="line", x0=p, x1=p, y0=0, y1=1, xref="x", yref="paper",
                     line=dict(color=T("muted"), width=1, dash="dot")) for p in (pK1, pK2)]
            self.fig.layout.annotations = [
                dict(x=8.0, y=1.0, xref="x", yref="paper", text="drinking water pH 6.5–9.5",
                     showarrow=False, yanchor="bottom", font=dict(size=10, color=T("muted")))] + [
                dict(x=p, y=0.04, xref="x", yref="y", text=f"{n} = {p:.2f}", showarrow=False,
                     xanchor="left", xshift=4, font=dict(size=10, color=T("muted")))
                for p, n in ((pK1, "pK<sub>1</sub>"), (pK2, "pK<sub>2</sub>"))]
            self.fig.layout.title.text = f"Carbonate species at {temp_C:g} °C"
        return self.fig


# ---------------------------------------------------------------------------
# matplotlib helpers (fallback)
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib.pyplot as plt
    return plt


def _species_mpl(temp_C):
    plt = _plt()
    K = cb._Consts(temp_C, 0, activity=False)
    pH = np.linspace(4, 12, 400)
    a0, a1, a2 = cb.alphas(pH, K)
    pK1, pK2 = -np.log10(K.K1), -np.log10(K.K2)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.axvspan(6.5, 9.5, color="0.93", zorder=0)
    ax.text(8.0, 1.03, "drinking water pH 6.5–9.5", ha="center", fontsize=8, color="0.4")
    ax.plot(pH, a0, lw=2, label="CO$_2^*$ (free CO$_2$)")
    ax.plot(pH, a1, lw=2, label="HCO$_3^-$")
    ax.plot(pH, a2, lw=2, label="CO$_3^{2-}$")
    for p, name in ((pK1, "p$K_1$"), (pK2, "p$K_2$")):
        ax.axvline(p, color="0.5", ls=":", lw=1)
        ax.text(p + 0.07, 0.06, f"{name} = {p:.2f}", fontsize=8, color="0.3")
    ax.set_xlim(4, 12); ax.set_ylim(0, 1.08)
    ax.set_xlabel("pH"); ax.set_ylabel("fraction of $C_T$")
    ax.set_title(f"Carbonate species at {temp_C:g} °C", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.85), fontsize=9)
    plt.show()


def _deffeyes_mpl(temp_C, I, ct_max, alk_max, ca_sat=(), steps=None, arrows=None,
                  points=None, title="", target=True, figsize=(8, 6)):
    plt = _plt()
    fig, ax = plt.subplots(figsize=figsize)
    cb.deffeyes(ax, temp_C=temp_C, I=I, ct_max=ct_max, alk_max=alk_max,
                ca_sat=list(ca_sat), target=target)
    if steps:
        cb.plot_path(ax, steps)
    for a, b, col, text in (arrows or []):
        ax.annotate("", xy=(b.ct_mmol, b.alk_mmol), xytext=(a.ct_mmol, a.alk_mmol),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
        ax.annotate(text, (b.ct_mmol, b.alk_mmol), xytext=(6, 4), textcoords="offset points",
                    color=col, fontsize=9)
    if arrows:
        ax.plot(arrows[0][0].ct_mmol, arrows[0][0].alk_mmol, "ko", ms=7)
    for w, text in (points or []):
        ax.plot(w.ct_mmol, w.alk_mmol, "o", color="k", ms=7)
        ax.annotate(text, (w.ct_mmol, w.alk_mmol), xytext=(6, -12),
                    textcoords="offset points", fontsize=8.5)
    if title:
        ax.set_title(title, fontsize=11, pad=34)
    plt.show()


def _pk_text(temp_C):
    k = cb.log_K(temp_C)
    return (f"At {temp_C:g} °C: pK1 = {-k['K1']:.2f}, pK2 = {-k['K2']:.2f}, "
            f"pKs(calcite) = {-k['Ks']:.2f}   (lecture, 5 °C: 6.52, 10.55, 8.39)")


# ---------------------------------------------------------------------------
# Views used in the notebook
# ---------------------------------------------------------------------------
def species():
    """A1: carbonate species versus pH, temperature slider."""
    t_sl = slider(5, 0, 30, 1, "Temperature (°C)")
    info = widgets.HTML()
    if USE_PLOTLY:
        sp = SpeciesPlotly()

        def refresh(_=None):
            sp.update(t_sl.value)
            info.value = _pk_text(t_sl.value)
        t_sl.observe(refresh, names="value")
        refresh()
        display(t_sl, sp.fig, info)
    else:
        def show(temp_C):
            _species_mpl(temp_C)
            print(_pk_text(temp_C))
        display(t_sl, widgets.interactive_output(show, {"temp_C": t_sl}))


def calculator():
    """A2: water calculator (no figure)."""
    inp = {
        "pH": widgets.BoundedFloatText(value=6.6, min=4, max=11, step=0.1, description="pH", style=STYLE),
        "alk": widgets.BoundedFloatText(value=0.10, min=0.0, max=10, step=0.05,
                                        description="Alkalinity (mmol/L)", style=STYLE),
        "ca": widgets.BoundedFloatText(value=2.5, min=0.1, max=200, step=0.5,
                                       description="Calcium (mg Ca/L)", style=STYLE),
        "T": widgets.BoundedFloatText(value=5, min=0, max=30, step=1,
                                      description="Temperature (°C)", style=STYLE),
        "ec": widgets.BoundedFloatText(value=35, min=1, max=2000, step=5,
                                       description="Conductivity (µS/cm)", style=STYLE),
    }

    def calc(pH, alk, ca, T, ec):
        try:
            w = cb.Water.from_pH_alk(pH, alk, ca, temp_C=T, ec_uS_cm=ec)
        except ValueError as e:
            display(HTML(f"<b style='color:{RED}'>Cannot calculate:</b> {e}"))
            return
        co2, hco3, co3 = (1000 * x for x in w.species)
        rows = [("C<sub>T</sub> (total inorganic carbon)", f"{w.ct_mmol:.3f} mmol/L", ""),
                ("free CO2 (CO2*)", f"{co2:.3f} mmol/L", f"{w.co2_mg:.1f} mg CO2/L"),
                ("HCO3-", f"{hco3:.3f} mmol/L", f"{hco3 * 61.02:.1f} mg/L"),
                ("CO3--", f"{co3:.4f} mmol/L", f"{co3 * 60.01:.2f} mg/L"),
                ("Saturation index SI", f"{nz(w.SI, 2):+.2f}", "< 0: dissolves CaCO3 (aggressive)"),
                ("CCPP", f"{nz(w.CCPP(), 1):+.1f} mg CaCO3/L", "− : can dissolve, + : can precipitate"),
                ("Ionic strength (from conductivity)", f"{w.I * 1000:.2f} mmol/L", "")]
        display(HTML(pd.DataFrame(rows, columns=["quantity", "value", "also / meaning"]).to_html(index=False, escape=False)))

    display(widgets.VBox(list(inp.values())), widgets.interactive_output(calc, inp))


def overview():
    """A3: empty Deffeyes diagram with the two example waters."""
    pts = [(w, name) for name, w in WATERS.items()]
    if USE_PLOTLY:
        d = DeffeyesPlotly(n_sat=0)
        d.update(5, 5e-4, 2.0, 1.5, points=pts)
        display(d.fig)
    else:
        _deffeyes_mpl(5, 5e-4, 2.0, 1.5, points=pts, figsize=(7.5, 5.8))


ARROWS = ["↑ up", "↓ down", "← left", "→ right", "↗ up-right, slope 2"]
QUIZ = {"NaOH dosing": "↑ up", "CO2 stripping": "← left",
        "CO2 dosing": "→ right", "CaCO3 (marble) dissolution": "↗ up-right, slope 2"}
WHY = {"NaOH dosing": "OH⁻ adds alkalinity, no carbon.",
       "CO2 stripping": "removes carbon, no charge.",
       "CO2 dosing": "adds carbon, no charge.",
       "CaCO3 (marble) dissolution": "per mmol: +1 mmol C<sub>T</sub> and +2 meq alkalinity "
                                     "(CaCO₃ + CO₂ + H₂O → Ca²⁺ + 2 HCO₃⁻)."}


def arrows_quiz():
    """A3: 'treatment as arrows' self-check (Menti question)."""
    picks = {k: widgets.Dropdown(options=["– choose –"] + ARROWS, description=k,
                                 style=STYLE, layout=WIDE) for k in QUIZ}
    btn = widgets.Button(description="Check", button_style="info")
    out = widgets.Output()

    def check(_):
        out.clear_output()
        with out:
            lines = []
            for k, dd in picks.items():
                ok = dd.value == QUIZ[k]
                mark = f"<b style='color:{GREEN}'>✓</b>" if ok else f"<b style='color:{RED}'>✗</b>"
                lines.append(f"{mark} <b>{k}</b>: {QUIZ[k]} – {WHY[k]}")
            display(HTML("<br>".join(lines)))
            w = WATERS["Soft surface water (example)"]
            arr = [(w, w.apply(s), col, text) for text, s, col in (
                ("NaOH 8 mg/L", cb.naoh(8), "#1f77b4"),
                ("strip 2 mg CO2/L", cb.co2_strip(2), "#2ca02c"),
                ("CO2 10 mg/L", cb.co2_dose(10), "#d62728"),
                ("CaCO3 30 mg/L", cb.caco3(30), "#9467bd"))]
            if USE_PLOTLY:
                d = DeffeyesPlotly(n_sat=0, height=500, width=680)
                d.update(5, w.I, 1.0, 0.8, arrows=arr, target=False)
                display(d.fig)
            else:
                _deffeyes_mpl(5, w.I, 1.0, 0.8, arrows=arr, target=False, figsize=(7, 5))
    btn.on_click(check)
    display(_check_frame([widgets.HTML("<b>In which direction does each step move the water in the "
                                       "Deffeyes diagram?</b> Start: the soft surface water.")]
                         + list(picks.values()) + [btn, out]))
    return picks, btn


def explore_path(water, strip, co2, marble, marble_mg, lime, naoh, start=None):
    """Apply the explorer steps in fixed order; returns (waters, notes).
    start: begin from this water instead of the raw water (e.g. after coagulation)."""
    w = start if start is not None else WATERS[water]
    notes, ws = [], [w]
    if strip > 0:
        air = w.apply(cb.air_equilibrium(w))
        if strip >= w.co2_mg - air.co2_mg:
            ws.append(air)
            notes.append(f"Stripping limited to equilibrium with air ({air.co2_mg:.1f} mg CO2/L left).")
        else:
            ws.append(w.apply(cb.co2_strip(strip)))
    if co2 > 0:
        ws.append(ws[-1].apply(cb.co2_dose(co2)))
    if marble == "to equilibrium":
        ws.append(ws[-1].apply(cb.marble_to_equilibrium()))
    elif marble == "fixed amount" and marble_mg > 0:
        possible = -ws[-1].CCPP()
        if possible <= 0:
            notes.append("The water is already saturated: marble does not dissolve.")
        elif marble_mg > possible:
            ws.append(ws[-1].apply(cb.marble_to_equilibrium()))
            notes.append(f"Only {possible:.1f} mg/L CaCO3 can dissolve before saturation – "
                         "limited to equilibrium.")
        else:
            ws.append(ws[-1].apply(cb.caco3(marble_mg)))
    if lime > 0:
        ws.append(ws[-1].apply(cb.lime(lime)))
    if naoh > 0:
        ws.append(ws[-1].apply(cb.naoh(naoh)))
    return ws, notes


def explorer():
    """A4: treatment explorer."""
    ctrl = {
        "water": widgets.Dropdown(options=list(WATERS), description="Raw water", style=STYLE, layout=WIDE),
        "strip": slider(0, 0, 40, 0.5, "1  CO2 stripped (mg/L)"),
        "co2": slider(0, 0, 40, 0.5, "2  CO2 dosed (mg/L)"),
        "marble": widgets.Dropdown(options=["no marble filter", "fixed amount", "to equilibrium"],
                                   description="3  Marble filter", style=STYLE, layout=WIDE),
        "marble_mg": slider(20, 0, 100, 1, "    CaCO3 dissolved (mg/L)"),
        "lime": slider(0, 0, 50, 0.5, "4  Lime (mg Ca(OH)2/L)"),
        "naoh": slider(0, 0, 30, 0.1, "5  NaOH (mg/L)"),
    }

    def results(ws, notes):
        display(table(ws))
        display(target_check(ws[-1]))
        for n in notes:
            display(HTML(f"<i>Note: {n}</i>"))

    def ca_lines(ws):
        w, final = ws[0], ws[-1]
        return [w.ca_mg] + ([final.ca_mg] if abs(final.ca_mg - w.ca_mg) > 0.5 else [])

    if USE_PLOTLY:
        d = DeffeyesPlotly(n_sat=2)
        out = widgets.Output()

        def refresh(_=None):
            v = {k: c.value for k, c in ctrl.items()}
            ws, notes = explore_path(**v)
            ct_max, alk_max = axis_limits(ws)
            d.update(ws[0].temp_C, ws[-1].I, ct_max, alk_max, ca_sat=ca_lines(ws), steps=ws,
                     title=f"{v['water']}, {ws[0].temp_C:g} °C")
            out.clear_output(wait=True)
            with out:
                results(ws, notes)
        for c in ctrl.values():
            c.observe(refresh, names="value")
        refresh()
        display(widgets.VBox(list(ctrl.values())), d.fig, out)
    else:
        def show(**v):
            ws, notes = explore_path(**v)
            ct_max, alk_max = axis_limits(ws)
            _deffeyes_mpl(ws[0].temp_C, ws[-1].I, ct_max, alk_max, ca_sat=ca_lines(ws), steps=ws,
                          title=f"{v['water']}, {ws[0].temp_C:g} °C")
            results(ws, notes)
        display(widgets.VBox(list(ctrl.values())), widgets.interactive_output(show, ctrl))
    return ctrl


def saturation_demo():
    """A5: dissolve marble step by step; the saturation line moves with Ca."""
    w0 = WATERS["Soft surface water (example)"].apply(cb.co2_dose(16))
    w0 = cb.Water(w0.temp_C, w0.I, w0.Ca, w0.Alk, w0.CT, label="after 16 mg CO2/L")
    max_mg = -w0.CCPP()
    s = slider(0, 0, np.floor(max_mg), 1, "CaCO3 dissolved (mg/L)")
    info = widgets.HTML()
    title = "Soft surface water + 16 mg CO<sub>2</sub>/L, then marble"

    def water(dissolved):
        if dissolved >= np.floor(max_mg):          # last slider position = equilibrium
            return w0.apply(cb.marble_to_equilibrium()), max_mg
        return (w0.apply(cb.caco3(dissolved)) if dissolved > 0 else w0), dissolved

    def text(w, dissolved):
        return (f"dissolved {dissolved:.0f} of {max_mg:.0f} mg CaCO3/L possible &nbsp;|&nbsp; "
                f"pH {w.pH:.2f} &nbsp; Ca {w.ca_mg:.1f} mg/L &nbsp; <b>SI {nz(w.SI, 2):+.2f}</b> "
                f"&nbsp; CCPP {nz(w.CCPP(), 1):+.1f} mg CaCO3/L")

    if USE_PLOTLY:
        d = DeffeyesPlotly(n_sat=1, height=520, width=700)

        def refresh(_=None):
            w, dissolved = water(s.value)
            d.update(5, w.I, 1.25, 1.25, ca_sat=[w.ca_mg], steps=[w0, w] if s.value > 0 else [w0],
                     title=title, start_label="start")
            info.value = text(w, dissolved)
        s.observe(refresh, names="value")
        refresh()
        display(s, d.fig, info)
    else:
        def show(dissolved):
            w, dd = water(dissolved)
            _deffeyes_mpl(5, w.I, 1.25, 1.25, ca_sat=[w.ca_mg], steps=[w0, w],
                          title=title.replace("<sub>", "$_").replace("</sub>", "$"),
                          figsize=(7.5, 5.6))
            display(HTML(text(w, dd)))
        display(s, widgets.interactive_output(show, {"dissolved": s}))
    return s


def temperature_table():
    """A5: the same water at different temperatures."""
    w_eq = WATERS["Soft surface water (example)"].treat(
        [cb.co2_dose(16), cb.marble_to_equilibrium()])[-1]
    rows = []
    for T in (2, 5, 10, 20, 40, 60):
        wt = cb.Water(T, w_eq.I, w_eq.Ca, w_eq.Alk, w_eq.CT)
        rows.append({"temperature (°C)": T, "pH": f"{wt.pH:.2f}", "SI": f"{nz(wt.SI, 2):+.2f}",
                     "CCPP (mg CaCO3/L)": f"{nz(wt.CCPP(), 1):+.1f}"})
    display(HTML(pd.DataFrame(rows).to_html(index=False)))
    display(HTML("<small>Values above 40 °C are approximate (activity model).</small>"))


def calcium_question():
    """A5: Menti question 'Calcium AND alkalinity'."""
    choice_check(
        "<b>Which method adds BOTH calcium and alkalinity to the water?</b>",
        ["NaOH", "CO2 stripping", "CO2 + marble filter", "Waterglass"],
        "CO2 + marble filter",
        "CaCO₃ + CO₂ + H₂O → Ca²⁺ + 2 HCO₃⁻: each mmol CO₂ used brings 1 mmol Ca and 2 meq "
        "alkalinity. That is why marble filters are so common in Norway. (Lime + CO₂ also adds both.)",
        {"NaOH": "NaOH adds alkalinity, but no calcium.",
         "CO2 stripping": "Stripping only removes CO₂; alkalinity and calcium stay the same.",
         "Waterglass": "Waterglass is a base with silicate; it adds no calcium."})


def hand_check():
    """A6: check a hand-calculated pHs."""
    ans = widgets.FloatText(value=0.0, description="Your pHs", style=STYLE)
    btn = widgets.Button(description="Check", button_style="info")
    out = widgets.Output()
    w = cb.Water.from_pH_alk(8.3, 1.0, 20, temp_C=5, ec_uS_cm=200)
    w_ideal = cb.Water.from_pH_alk(8.3, 1.0, 20, temp_C=5, I=1e-9, activity=False)

    def check(_):
        out.clear_output()
        with out:
            ref, ref_act = w_ideal.pHs(), w.pHs()
            if abs(ans.value - ref) > 0.03:
                display(HTML(f"<b style='color:{RED}'>✗ Not the same as the tool.</b> Check: Ca in "
                             "mol/L (mg/L ÷ 1000 ÷ 40.08), [HCO₃⁻] in mol/L, and the signs: −log of a "
                             "small number is positive. pHs should come out between 7 and 10. Try again."))
                return
            display(HTML(f"<b style='color:{GREEN}'>✓ Matches the tool</b> (pHs = {ref:.2f} without "
                         f"activity correction, LSI = {8.3 - ref:+.2f})."))
            display(HTML(f"With the ionic-strength correction (I = {w.I*1000:.1f} mmol/L from the "
                         f"conductivity) the tool gets pHs = {ref_act:.2f} and LSI = {w.LSI:+.2f}. "
                         f"The correction S = {ref_act - ref:+.2f} is as large as the difference between "
                         "'just saturated' and 'clearly aggressive' – use it in your design."))
    btn.on_click(check)
    display(widgets.HBox([ans, btn]), out)
    return ans, btn


# ===========================================================================
# Chapter notebooks (A1, A2, ...): layout helpers
# ===========================================================================
import os
from pathlib import Path

FIG_DIR = Path.home() / "shared" / "figures" / "stabilisation"   # lecture pictures (not part of the repository)
PSTYLE = {"description_width": "130px"}
PWIDE = widgets.Layout(width="390px")
OK_BG, BAD_BG, NEUTRAL_BG = "rgba(67,160,71,0.25)", "rgba(229,57,53,0.25)", "rgba(128,128,128,0.15)"
COL = {"CO2": "#d62728", "HCO3": "#ff7f0e", "CO3": "#2ca02c", "base": "#1f77b4",
       "marble": "#9467bd", "Ca": "#8c564b", "H": "#7f7f7f", "OH": "#17becf"}


def setup():
    """Called in the first cell of each chapter (after sv.DARK is set)."""
    _CHECK_N[0] = 0
    if not USE_PLOTLY:
        _plt().style.use("dark_background" if DARK else "default")
    fig = ("Plotly (hover, zoom)" if USE_PLOTLY else "matplotlib") + (", dark" if DARK else ", light")
    if FIG_DIR.is_dir():
        pics = "lecture pictures: available"
    else:
        pics = "lecture pictures: not available – the text refers to the lecture slides instead"
    display(HTML(f"<small>Toolbox loaded · figures: {fig} · {pics}</small>"))


def pslider(value, vmin, vmax, step, text, fmt=None):
    kw = dict(value=value, min=vmin, max=vmax, step=step, description=text, description_allow_html=True,
              style=PSTYLE, layout=PWIDE, continuous_update=False)
    if fmt:
        kw["readout_format"] = fmt
    return widgets.FloatSlider(**kw)


def picture(items, fallback=""):
    """Show lecture pictures side by side (if available).
    items: list of (file name, caption, width in px)."""
    boxes, missing = [], []
    for fname, cap, w in items:
        p = FIG_DIR / fname
        if p.is_file():
            fmt = "png" if p.suffix.lower() == ".png" else "jpeg"
            img = widgets.Image(value=p.read_bytes(), format=fmt, layout=widgets.Layout(width=f"{w}px"))
            txt = widgets.HTML(f"<small><i>{cap}</i></small>", layout=widgets.Layout(width=f"{w}px"))
            boxes.append(widgets.VBox([img, txt], layout=widgets.Layout(margin="0 16px 8px 0")))
        else:
            missing.append(cap)
    if boxes:
        display(widgets.HBox(boxes, layout=widgets.Layout(flex_flow="row wrap")))
    if missing:
        display(HTML(f"<div style='opacity:0.8;border-left:3px solid {BORDER};padding-left:8px'>"
                     f"<i>Picture not available here: {'; '.join(missing)}. {fallback}</i></div>"))


def cards(items):
    """Key numbers as small cards. items: (label, value, status) with
    status 'ok', 'bad' or None."""
    bg = {"ok": OK_BG, "bad": BAD_BG, None: NEUTRAL_BG}
    html = "".join(
        f"<div style='display:inline-block;min-width:105px;margin:4px 8px 4px 0;padding:5px 9px;"
        f"border-radius:6px;background:{bg[s]}'><div style='font-size:0.78em;opacity:0.75'>{lab}</div>"
        f"<div style='font-size:1.25em;font-weight:600'>{val}</div></div>" for lab, val, s in items)
    return html


def panel(title, hint, controls, figure, readout, presets=None, below=None, symbols=None):
    """One framed interactive element: title, what to look for,
    controls + presets + key numbers on the left, figure on the right."""
    head = widgets.HTML(f"<div style='font-size:1.15em;font-weight:600;margin-bottom:2px'>{title}</div>"
                        f"<div style='opacity:0.8;margin-bottom:6px'>{hint}</div>"
                        + (f"<div style='font-size:0.85em;opacity:0.85;margin-bottom:6px'><b>Symbols:</b> {symbols}</div>"
                           if symbols else ""))
    left = ([presets] if presets is not None else []) + list(controls) + [readout]
    left = widgets.VBox(left, layout=widgets.Layout(width="410px", margin="0 18px 0 0"))
    body = widgets.HBox([left, figure], layout=widgets.Layout(flex_flow="row wrap", align_items="flex-start"))
    parts = [head, body] + ([below] if below is not None else [])
    display(widgets.VBox(parts, layout=widgets.Layout(
        border=f"1px solid {BORDER}", padding="12px", margin="12px 0 12px 0")))


def wire(ctrls, refresh, presets=None):
    """Connect controls and preset buttons to refresh(); returns the preset box."""
    state = {"hold": False}

    def on_change(_=None):
        if not state["hold"]:
            refresh()
    for c in ctrls.values():
        c.observe(on_change, names="value")
    if not presets:
        return None
    btns = []
    for name, vals in presets.items():
        b = widgets.Button(description=name, layout=widgets.Layout(width="auto", margin="0 6px 4px 0"))

        def click(_, vals=vals):
            state["hold"] = True
            for k, v in vals.items():
                ctrls[k].value = v() if callable(v) else v
            state["hold"] = False
            refresh()
        b.on_click(click)
        btns.append(b)
    return widgets.VBox([widgets.HTML("<small><b>Try:</b> click a preset, then change the sliders</small>"),
                         widgets.HBox(btns, layout=widgets.Layout(flex_flow="row wrap"))],
                        layout=widgets.Layout(margin="0 0 8px 0"))


def numeric_check(prompt, answer, tol, unit, hint, explanation):
    """Self-check for a calculated number (relative tolerance tol)."""
    box = widgets.FloatText(value=0.0, description="Your answer", style=PSTYLE,
                            layout=widgets.Layout(width="260px"))
    btn = widgets.Button(description="Check", button_style="info")
    out = widgets.Output()

    def check(_):
        out.clear_output()
        with out:
            if box.value != 0 and abs(box.value - answer) <= tol * abs(answer):
                display(HTML(f"<b style='color:{GREEN}'>✓ Correct</b> ({answer:.3g} {unit}). {explanation}"))
            else:
                display(HTML(f"<b style='color:{RED}'>✗ Not yet.</b> {hint}"))
    btn.on_click(check)
    display(_check_frame([widgets.HTMLMath(value=prompt),
                          widgets.HBox([box, widgets.Label(value=unit), btn]), out]))


def _figure_slot(make_plotly):
    """Plotly FigureWidget, or an Output widget for matplotlib."""
    return make_plotly() if USE_PLOTLY else widgets.Output()


def _mpl_into(out, draw):
    plt = _plt()
    out.clear_output(wait=True)
    with out:
        draw(plt)
        plt.show()


# ===========================================================================
# A1 – raw water versus target
# ===========================================================================
TYPICAL = {
    "Clear mountain lake": dict(pH=6.2, alk=0.05, ca=1.5),
    "Humic surface water": dict(pH=6.6, alk=0.10, ca=2.5),
    "CO2-rich groundwater": dict(pH=6.4, alk=0.70, ca=12.0),
    "Treated water (CO2 + marble)": dict(pH=8.3, alk=0.95, ca=19.5),
}
_ROWS = [("pH", "pH", (5.0, 9.5), (8.0, 8.5)),
         ("alk", "Alkalinity (mmol/L)", (0.0, 1.5), (0.6, 1.0)),
         ("ca", "Calcium (mg Ca/L)", (0.0, 30.0), (15.0, 25.0)),
         ("SI", "Saturation index SI", (-5.0, 1.0), (-0.1, 0.3))]


def _si_estimate(pH, alk, ca):
    """SI at 5 °C with ionic strength estimated from Ca (Ca(HCO3)2-type water)."""
    try:
        I = 5e-4 + 3 * ca / cb.M["Ca"] / 1000
        return cb.Water.from_pH_alk(pH, alk, max(ca, 0.05), temp_C=5, I=I).SI
    except ValueError:
        return None


def raw_vs_target():
    """A1: compare a raw water with the target water quality (NV 274/2022)."""
    ctrls = {"pH": pslider(6.6, 5.0, 9.5, 0.05, "pH", ".2f"),
             "alk": pslider(0.10, 0.0, 1.5, 0.01, "Alkalinity (mmol/L)", ".2f"),
             "ca": pslider(2.5, 0.0, 30.0, 0.5, "Calcium (mg/L)", ".1f")}
    readout = widgets.HTML()
    if USE_PLOTLY:
        import plotly.graph_objects as go
        rows = []
        for key, label, rng, tgt in _ROWS:
            f = go.FigureWidget()
            f.add_scatter(x=[], y=[0.5], mode="markers", marker=dict(size=16, color=T("ink"), symbol="diamond"),
                          hovertemplate=label + ": %{x:.2f}<extra></extra>", showlegend=False)
            f.update_layout(template=T("template"), height=105, width=520,
                            margin=dict(l=150, r=20, t=8, b=30),
                            xaxis=dict(range=list(rng)), yaxis=dict(visible=False, range=[0, 1]),
                            shapes=[dict(type="rect", x0=tgt[0], x1=tgt[1], y0=0.15, y1=0.85, xref="x",
                                         yref="y", fillcolor=T("target"), line=dict(width=0))],
                            annotations=[dict(x=0, y=0.5, xref="paper", yref="y", text=f"<b>{label}</b>",
                                              showarrow=False, xanchor="right", xshift=-10)])
            _style(f)
            rows.append(f)
        figure = widgets.VBox(rows + [widgets.HTML(
            "<small>Green: target range (NV 274/2022; SI: our rule of thumb). ◆ the selected water.</small>")])
    else:
        figure = widgets.Output()

    def refresh():
        v = {k: c.value for k, c in ctrls.items()}
        si = _si_estimate(v["pH"], v["alk"], v["ca"])
        vals = {"pH": v["pH"], "alk": v["alk"], "ca": v["ca"], "SI": si}
        if USE_PLOTLY:
            for f, (key, label, rng, tgt) in zip(rows, _ROWS):
                x = vals[key]
                f.data[0].x, f.data[0].y = ([], []) if x is None else ([min(max(x, rng[0]), rng[1])], [0.5])
        else:
            def draw(plt):
                fig, axs = plt.subplots(4, 1, figsize=(6.5, 3.6))
                for ax, (key, label, rng, tgt) in zip(axs, _ROWS):
                    ax.axvspan(*tgt, color="#2e7d32", alpha=0.3)
                    if vals[key] is not None:
                        ax.plot([min(max(vals[key], rng[0]), rng[1])], [0.5], "kD", ms=9)
                    ax.set_xlim(rng); ax.set_ylim(0, 1); ax.set_yticks([])
                    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
                fig.tight_layout()
            _mpl_into(figure, draw)
        (p0, p1), (a0, a1), (c0, c1) = cb.TARGETS["pH"], cb.TARGETS["Alk"], cb.TARGETS["Ca"]
        items = [("pH", f"{v['pH']:.2f}", "ok" if p0 <= v["pH"] <= p1 else "bad"),
                 ("Alkalinity", f"{v['alk']:.2f} mmol/L", "ok" if a0 <= v["alk"] <= a1 else "bad"),
                 ("Calcium", f"{v['ca']:.1f} mg/L", "ok" if c0 <= v["ca"] <= c1 else "bad"),
                 ("SI (5 °C)", "–" if si is None else f"{nz(si, 2):+.2f}",
                  None if si is None else ("ok" if si >= -0.1 else "bad"))]
        need = []
        if v["alk"] < a0:
            need.append(f"+{a0 - v['alk']:.2f} mmol/L alkalinity")
        if v["ca"] < c0:
            need.append(f"+{c0 - v['ca']:.1f} mg/L calcium")
        msg = ("To reach the lower target limits this water needs at least " + " and ".join(need) + "."
               if need else "Alkalinity and calcium are within or above the target.")
        readout.value = cards(items) + f"<div style='margin-top:6px'>{msg}</div>"

    presets = {k: dict(v) for k, v in TYPICAL.items()}
    pbox = wire(ctrls, refresh, presets)
    refresh()
    panel("Your raw water versus the target",
          "Choose a typical Norwegian water (or set your own values). Which parameters are far from the "
          "target, and what would have to be <i>added</i> to the water?",
          list(ctrls.values()), figure, readout, pbox,
          symbols="Alkalinity = the water's ability to neutralise acid, mostly bicarbonate (chapter A4). SI = saturation index for calcium carbonate: &lt; 0 the water dissolves CaCO<sub>3</sub> (aggressive), &gt; 0 it can deposit CaCO<sub>3</sub> (chapter A5); calculated here for 5 °C.")


# ===========================================================================
# A2 – CO2 and the gas phase
# ===========================================================================
GASES = [("air, 0.042 % (420 ppm)", cb.P_CO2_AIR), ("0.5 %", 0.005), ("1 %", 0.01),
         ("3 % (typical soil air)", 0.03), ("5 %", 0.05), ("10 %", 0.10)]


def co2_gas():
    """A2: dissolved CO2 in equilibrium with air or soil air; pH of the water."""
    ctrls = {"gas": widgets.Dropdown(options=GASES, value=cb.P_CO2_AIR, description="CO2 in the gas",
                                     style=PSTYLE, layout=PWIDE),
             "T": pslider(5, 0, 25, 1, "Temperature (°C)", ".0f"),
             "alk": pslider(0.1, 0.0, 2.0, 0.05, "Alkalinity (mmol/L)", ".2f")}
    readout = widgets.HTML()
    temps = np.linspace(0, 25, 51)
    if USE_PLOTLY:
        import plotly.graph_objects as go
        fig = go.FigureWidget()
        for label, p in GASES:
            fig.add_scatter(x=temps, y=cb.co2_saturation_mg(temps, p), mode="lines", name=label,
                            line=dict(color=T("faint"), width=1.5),
                            hovertemplate="%{x:.0f} °C: %{y:.1f} mg CO<sub>2</sub>/L<extra>" + label + "</extra>")
        fig.add_scatter(x=[], y=[], mode="markers", name="selected", showlegend=False,
                        marker=dict(size=13, color=COL["CO2"]),
                        hovertemplate="%{x:.0f} °C: %{y:.2f} mg CO<sub>2</sub>/L<extra></extra>")
        fig.update_layout(template=T("template"), height=430, width=600, showlegend=False,
                          margin=dict(l=70, r=150, t=50, b=55), hovermode="closest",
                          title=dict(text="Free CO<sub>2</sub> in equilibrium with the gas", x=0.5,
                                     xanchor="center"),
                          xaxis=dict(title=dict(text="Temperature (°C)"), range=[0, 25]),
                          yaxis=dict(title=dict(text="free CO<sub>2</sub> (mg/L), log scale"), type="log",
                                     range=[np.log10(0.3), np.log10(400)]),
                          annotations=[dict(x=1.0, xref="paper", y=float(np.log10(cb.co2_saturation_mg(25, p))),
                                            yref="y", text=label, showarrow=False, xanchor="left", xshift=4,
                                            font=dict(size=10, color=T("muted"))) for label, p in GASES])
        _style(fig)
        figure = fig
    else:
        figure = widgets.Output()

    def refresh():
        p, t, alk = ctrls["gas"].value, ctrls["T"].value, ctrls["alk"].value
        co2 = cb.co2_saturation_mg(t, p)
        w = cb.gas_equilibrium(alk, p, temp_C=t, I=5e-4)
        if USE_PLOTLY:
            with fig.batch_update():
                for i, (label, pp) in enumerate(GASES):
                    sel = pp == p
                    fig.data[i].line.color = COL["CO2"] if sel else T("faint")
                    fig.data[i].line.width = 3 if sel else 1.5
                fig.data[-1].x, fig.data[-1].y = [t], [co2]
        else:
            def draw(plt):
                f, ax = plt.subplots(figsize=(6.2, 4.2))
                for label, pp in GASES:
                    sel = pp == p
                    ax.plot(temps, cb.co2_saturation_mg(temps, pp), color=COL["CO2"] if sel else "0.75",
                            lw=2.5 if sel else 1)
                    ax.text(25.3, cb.co2_saturation_mg(25, pp), label, fontsize=8, va="center", color="0.4")
                ax.plot(t, co2, "o", color=COL["CO2"], ms=9)
                ax.set_yscale("log"); ax.set_xlim(0, 25); ax.set_ylim(0.3, 400)
                ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("free CO$_2$ (mg/L)")
                f.tight_layout()
            _mpl_into(figure, draw)
        readout.value = cards([("free CO2", f"{co2:.2f} mg/L", None),
                               ("free CO2", f"{co2 / 44.01:.3f} mmol/L", None),
                               ("pH of the water", f"{w.pH:.2f}", None),
                               ("C<sub>T</sub>", f"{w.ct_mmol:.2f} mmol/L", None)]) + (
            "<div style='margin-top:6px'><small>pH and C<sub>T</sub>: water with the chosen alkalinity "
            "after it has reached equilibrium with the gas.</small></div>")

    air = cb.P_CO2_AIR
    presets = {"Rain water in air": dict(gas=air, T=5, alk=0.0),
               "Lake water in air": dict(gas=air, T=5, alk=0.1),
               "Groundwater in soil air": dict(gas=0.03, T=6, alk=0.7),
               "…the same, stripped (air)": dict(gas=air, T=6, alk=0.7)}
    pbox = wire(ctrls, refresh, presets)
    refresh()
    panel("CO₂ in water: how much dissolves, and what it does to the pH",
          "Compare air with soil air, and cold with warm water. Then keep the alkalinity and change only "
          "the gas: what happens to the pH when a groundwater is stripped?",
          list(ctrls.values()), figure, readout, pbox,
          symbols='free CO<sub>2</sub> = CO<sub>2</sub>* (dissolved CO<sub>2</sub> + H<sub>2</sub>CO<sub>3</sub>) · C<sub>T</sub> = total inorganic carbon = CO<sub>2</sub>* + HCO<sub>3</sub><sup>−</sup> + CO<sub>3</sub><sup>2−</sup> · alkalinity: acid-neutralising capacity, mostly HCO<sub>3</sub><sup>−</sup> (chapter A4) · % in the gas = partial pressure of CO<sub>2</sub>.')


# ===========================================================================
# A3 – carbonate species explorer
# ===========================================================================
def species_explorer():
    """A3: fractions or log concentrations of the carbonate species, pH marker."""
    ctrls = {"view": widgets.ToggleButtons(options=["fractions", "log concentrations"], value="fractions",
                                           description="Show", style=PSTYLE),
             "pH": pslider(6.6, 4.0, 12.0, 0.05, "pH", ".2f"),
             "T": pslider(5, 0, 30, 1, "Temperature (°C)", ".0f"),
             "CT": pslider(1.0, 0.1, 3.0, 0.1, "C<sub>T</sub> (mmol/L)", ".1f")}
    readout = widgets.HTML()
    pH = np.linspace(4, 12, 321)
    names = [("CO<sub>2</sub>*", COL["CO2"]), ("HCO<sub>3</sub><sup>−</sup>", COL["HCO3"]),
             ("CO<sub>3</sub><sup>2−</sup>", COL["CO3"]), ("H<sup>+</sup>", COL["H"]),
             ("OH<sup>−</sup>", COL["OH"])]
    if USE_PLOTLY:
        import plotly.graph_objects as go
        fig = go.FigureWidget()
        for n, c in names:
            fig.add_scatter(x=[], y=[], mode="lines", name=n, line=dict(color=c, width=2.5),
                            hovertemplate="pH %{x:.2f}: %{y:.3f}<extra>" + n + "</extra>")
        fig.add_scatter(x=[], y=[], mode="markers", name="at selected pH", showlegend=False,
                        marker=dict(size=11, color=[c for _, c in names[:3]], line=dict(color=T("ink"), width=1)),
                        hovertemplate="%{y:.3f}<extra></extra>")
        fig.update_layout(template=T("template"), height=430, width=620, hovermode="closest",
                          margin=dict(l=70, r=30, t=50, b=55), title=dict(text="", x=0.5, xanchor="center"),
                          xaxis=dict(title=dict(text="pH"), range=[4, 12]),
                          legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top",
                                      bgcolor=T("legend")))
        _style(fig)
        figure = fig
    else:
        figure = widgets.Output()

    def refresh():
        v = {k: c.value for k, c in ctrls.items()}
        K = cb._Consts(v["T"], 0, activity=False)
        pK1, pK2, pKw = -np.log10(K.K1), -np.log10(K.K2), -np.log10(K.Kw)
        ct = v["CT"] / 1000
        fr = cb.alphas(pH, K)
        f0 = [float(a) for a in cb.alphas(v["pH"], K)]
        log = v["view"] == "log concentrations"
        if log:
            ys = [np.log10(np.maximum(ct * a, 1e-20)) for a in fr] + [-pH, pH - pKw]
            pts = [float(np.log10(max(ct * a, 1e-20))) for a in f0]
            yr, ytitle = [-9, -1.5], "log concentration (mol/L)"
        else:
            ys = list(fr) + [None, None]
            pts = f0
            yr, ytitle = [0, 1.08], "fraction of C<sub>T</sub>"
        if USE_PLOTLY:
            with fig.batch_update():
                for i, y in enumerate(ys):
                    fig.data[i].visible = y is not None
                    if y is not None:              # always send x together with y
                        fig.data[i].x, fig.data[i].y = pH, y
                fig.data[-1].x, fig.data[-1].y = [v["pH"]] * 3, pts
                fig.layout.yaxis.range = yr
                fig.layout.yaxis.title.text = ytitle
                fig.layout.title.text = f"Carbonate species at {v['T']:g} °C"
                fig.layout.shapes = [
                    dict(type="rect", x0=6.5, x1=9.5, y0=0, y1=1, xref="x", yref="paper",
                         fillcolor=T("shade"), line=dict(width=0), layer="below"),
                    dict(type="line", x0=v["pH"], x1=v["pH"], y0=0, y1=1, xref="x", yref="paper",
                         line=dict(color=T("ink"), width=1.5))] + [
                    dict(type="line", x0=p, x1=p, y0=0, y1=1, xref="x", yref="paper",
                         line=dict(color=T("muted"), width=1, dash="dot")) for p in (pK1, pK2)]
                fig.layout.annotations = [
                    dict(x=8.0, y=1.0, xref="x", yref="paper", text="drinking water pH 6.5–9.5",
                         showarrow=False, yanchor="bottom", font=dict(size=10, color=T("muted")))] + [
                    dict(x=p, y=0.02, xref="x", yref="paper", text=f"{n} = {p:.2f}", showarrow=False,
                         xanchor="left", yanchor="bottom", xshift=4, font=dict(size=10, color=T("muted")))
                    for p, n in ((pK1, "pK<sub>1</sub>"), (pK2, "pK<sub>2</sub>"))]
        else:
            def draw(plt):
                f, ax = plt.subplots(figsize=(6.4, 4.2))
                labels = ["CO$_2^*$", "HCO$_3^-$", "CO$_3^{2-}$", "H$^+$", "OH$^-$"]
                for (n, c), y, lab in zip(names, ys, labels):
                    if y is not None:
                        ax.plot(pH, y, color=c, lw=2, label=lab)
                ax.plot([v["pH"]] * 3, pts, "o", color="k", ms=6)
                ax.axvline(v["pH"], color="k", lw=1)
                for p in (pK1, pK2):
                    ax.axvline(p, color="0.5", ls=":", lw=1)
                ax.axvspan(6.5, 9.5, color="0.94", zorder=0)
                ax.set_xlim(4, 12); ax.set_ylim(yr); ax.set_xlabel("pH")
                ax.set_ylabel("log concentration (mol/L)" if log else "fraction of $C_T$")
                ax.set_title(f"Carbonate species at {v['T']:g} °C", fontsize=11)
                ax.legend(fontsize=8, loc="upper left")
                f.tight_layout()
            _mpl_into(figure, draw)
        co2, hco3, co3 = (ct * a for a in f0)
        h = 10 ** -v["pH"]
        alk = co3 * 2 + hco3 + K.Kw / h - h
        readout.value = cards([("CO2*", f"{100 * f0[0]:.1f} %", None),
                               ("HCO3⁻", f"{100 * f0[1]:.1f} %", None),
                               ("CO3²⁻", f"{100 * f0[2]:.2f} %", None)]) + cards([
            ("free CO2", f"{co2 * 44010:.1f} mg/L", None),
            ("HCO3⁻", f"{hco3 * 61020:.1f} mg/L", None),
            ("alkalinity", f"{alk * 1000:.2f} mmol/L", None)]) + (
            f"<div><small>pK<sub>1</sub> = {pK1:.2f}, pK<sub>2</sub> = {pK2:.2f} at {v['T']:g} °C "
            "(no activity correction – see chapter 5).</small></div>")

    def pk(i):
        return lambda: round(-np.log10(getattr(cb._Consts(ctrls["T"].value, 0, False), f"K{i}")), 2)
    presets = {"Rain water, pH 5.5": dict(pH=5.5), "Soft surface water, pH 6.6": dict(pH=6.6),
               "pH = pK₁": dict(pH=pk(1)), "Treated water, pH 8.3": dict(pH=8.3), "pH = pK₂": dict(pH=pk(2))}
    pbox = wire(ctrls, refresh, presets)
    refresh()
    panel("Carbonate species: which form dominates at which pH?",
          "Move the pH slider and watch the three fractions. Where do two curves cross? Switch to "
          "<i>log concentrations</i> to see the small concentrations that a linear plot hides.",
          list(ctrls.values()), figure, readout, pbox,
          symbols='C<sub>T</sub> = total inorganic carbon (sum of the three species) · pK<sub>1</sub>, pK<sub>2</sub> = −log of the acidity constants · log concentration = log<sub>10</sub> of mol/L · alkalinity ≈ HCO<sub>3</sub><sup>−</sup> + 2 CO<sub>3</sub><sup>2−</sup> (explained in chapter A4).')


# ===========================================================================
# A4 – alkalinity and buffering
# ===========================================================================
def _with_alk(w, d_mmol):
    """Same water after adding d mmol/L strong base (negative: strong acid)."""
    return cb.Water(w.temp_C, w.I, w.Ca, w.Alk + d_mmol / 1000, w.CT)


def _beta(w):
    """Buffer intensity in mmol/L per pH unit (numerical derivative)."""
    d = 1e-3
    return d / (_with_alk(w, d).pH - w.pH)


def _titration(w):
    d = np.linspace(-(w.alk_mmol + 0.08), 0.5, 140)
    ph = []
    for x in d:
        try:
            ph.append(_with_alk(w, x).pH)
        except ValueError:
            ph.append(np.nan)
    return d, np.array(ph)


def buffer_panel():
    """A4: how much does the pH move when base or acid is added?"""
    ctrls = {"pH": pslider(6.6, 5.5, 9.0, 0.05, "pH", ".2f"),
             "alk": pslider(0.10, 0.05, 2.0, 0.05, "Alkalinity (mmol/L)", ".2f"),
             "T": pslider(5, 0, 25, 1, "Temperature (°C)", ".0f"),
             "ref": widgets.Checkbox(value=True, description="show target water (pH 8.3, Alk 1.0) for comparison",
                                     style=PSTYLE, layout=PWIDE)}
    readout = widgets.HTML()
    if USE_PLOTLY:
        import plotly.graph_objects as go
        fig = go.FigureWidget()
        fig.add_scatter(x=[], y=[], mode="lines", name="target water", line=dict(color=T("faint"), width=2, dash="dash"),
                        hovertemplate="%{x:+.2f} mmol/L → pH %{y:.2f}<extra>target water</extra>")
        fig.add_scatter(x=[], y=[], mode="lines", name="your water", line=dict(color=COL["base"], width=3),
                        hovertemplate="%{x:+.2f} mmol/L → pH %{y:.2f}<extra>your water</extra>")
        fig.add_scatter(x=[0], y=[], mode="markers", name="start", showlegend=False,
                        marker=dict(size=12, color=T("ink")), hovertemplate="start: pH %{y:.2f}<extra></extra>")
        fig.update_layout(template=T("template"), height=430, width=600, hovermode="closest",
                          margin=dict(l=70, r=30, t=50, b=60), title=dict(text="Titration curve: add acid (−) or base (+)",
                                                                          x=0.5, xanchor="center"),
                          xaxis=dict(title=dict(text="strong acid (−) or base (+) added (mmol/L)")),
                          yaxis=dict(title=dict(text="pH"), range=[3.5, 11]),
                          legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top", bgcolor=T("legend")),
                          shapes=[dict(type="rect", x0=0, x1=1, y0=8.0, y1=8.5, xref="paper", yref="y",
                                       fillcolor=T("target"), line=dict(width=0), layer="below")] + [
                              dict(type="line", x0=0, x1=1, y0=v, y1=v, xref="paper", yref="y",
                                   line=dict(color=T("grid"), width=1, dash="dot")) for v in (4.5, 8.3)],
                          annotations=[dict(x=1, y=v, xref="paper", yref="y", text=t, showarrow=False,
                                            xanchor="right", yanchor="bottom", font=dict(size=10, color=T("muted")))
                                       for v, t in ((4.5, "end point pH 4.5 (alkalinity)"), (8.3, "end point pH 8.3"))])
        _style(fig)
        figure = fig
    else:
        figure = widgets.Output()

    def refresh():
        v = {k: c.value for k, c in ctrls.items()}
        try:
            w = cb.Water.from_pH_alk(v["pH"], v["alk"], 5, temp_C=v["T"], I=1e-3)
        except ValueError as e:
            readout.value = f"<b style='color:{RED}'>Cannot calculate:</b> {e}"
            return
        d, ph = _titration(w)
        ref = cb.Water.from_pH_alk(8.3, 1.0, 20, temp_C=v["T"], I=1e-3)
        dr, phr = _titration(ref)
        if USE_PLOTLY:
            with fig.batch_update():
                fig.data[0].x, fig.data[0].y = (dr, phr) if v["ref"] else ([], [])
                fig.data[1].x, fig.data[1].y = d, ph
                fig.data[2].x, fig.data[2].y = [0], [w.pH]
                fig.layout.xaxis.range = [float(min(d.min(), dr.min() if v["ref"] else 0)) - 0.02, 0.52]
        else:
            def draw(plt):
                f, ax = plt.subplots(figsize=(6.2, 4.2))
                ax.axhspan(8.0, 8.5, color="#66bb6a", alpha=0.35)
                for y in (4.5, 8.3):
                    ax.axhline(y, ls=":", color="0.5", lw=1)
                if v["ref"]:
                    ax.plot(dr, phr, "--", color="0.6", lw=2, label="target water")
                ax.plot(d, ph, color=COL["base"], lw=2.5, label="your water")
                ax.plot(0, w.pH, "o", color="0.5", ms=8)
                ax.set_ylim(3.5, 11); ax.set_xlabel("strong acid (−) or base (+) added (mmol/L)"); ax.set_ylabel("pH")
                ax.legend(fontsize=8, loc="upper left"); f.tight_layout()
            _mpl_into(figure, draw)
        up, down = _with_alk(w, 0.05).pH, _with_alk(w, -0.05).pH
        readout.value = cards([("C<sub>T</sub>", f"{w.ct_mmol:.2f} mmol/L", None),
                               ("buffer intensity β", f"{_beta(w):.3f} mmol/L per pH", None)]) + cards([
            ("+0.05 mmol/L NaOH (2 mg/L)", f"pH {up:.2f}", None),
            ("+0.05 mmol/L acid", f"pH {down:.2f}", None),
            ("acid to reach pH 4.5", f"≈ {w.alk_mmol:.2f} mmol/L", None)])

    presets = {"Soft surface water": dict(pH=6.6, alk=0.10, T=5),
               "Target water": dict(pH=8.3, alk=1.0, T=5),
               "Low-alkalinity water at pH 8.3": dict(pH=8.3, alk=0.10, T=5),
               "CO₂-rich groundwater": dict(pH=6.4, alk=0.70, T=6)}
    pbox = wire(ctrls, refresh, presets)
    refresh()
    panel("Buffering: how far does the pH move?",
          "Each curve shows the pH after adding strong acid (left) or base (right). Compare a soft water with "
          "the target water: which one keeps its pH when a little acid is added?",
          list(ctrls.values()), figure, readout, pbox,
          symbols='β = buffer intensity: mmol/L of base needed to raise the pH by one unit at the current pH (small β = pH changes easily) · C<sub>T</sub> = total inorganic carbon · end point pH 4.5 = end of the alkalinity titration.')


# ===========================================================================
# A5 – calcium carbonate saturation
# ===========================================================================
def saturation_panel():
    """A5: SI, LSI, pHs and CCPP of a water; SI as a function of pH."""
    ctrls = {"pH": pslider(8.3, 6.0, 9.5, 0.05, "pH", ".2f"),
             "alk": pslider(1.0, 0.05, 5.0, 0.05, "Alkalinity (mmol/L)", ".2f"),
             "ca": pslider(20, 1, 150, 0.5, "Calcium (mg/L)", ".1f"),
             "T": pslider(5, 0, 60, 1, "Temperature (°C)", ".0f"),
             "ec": pslider(200, 20, 1000, 10, "Conductivity (µS/cm)", ".0f")}
    readout = widgets.HTML()
    if USE_PLOTLY:
        import plotly.graph_objects as go
        fig = go.FigureWidget()
        fig.add_scatter(x=[], y=[], mode="lines", name="SI", line=dict(color=COL["marble"], width=3),
                        hovertemplate="pH %{x:.2f}: SI %{y:+.2f}<extra></extra>", showlegend=False)
        fig.add_scatter(x=[], y=[], mode="markers", showlegend=False, marker=dict(size=13, color=T("ink")),
                        hovertemplate="this water: SI %{y:+.2f}<extra></extra>")
        fig.update_layout(template=T("template"), height=420, width=600, hovermode="closest",
                          margin=dict(l=70, r=30, t=50, b=55),
                          title=dict(text="Saturation index if only the pH were different", x=0.5, xanchor="center"),
                          xaxis=dict(title=dict(text="pH"), range=[6, 9.5]),
                          yaxis=dict(title=dict(text="saturation index SI"), range=[-4, 2]))
        _style(fig)
        figure = fig
    else:
        figure = widgets.Output()

    def refresh():
        v = {k: c.value for k, c in ctrls.items()}
        try:
            w = cb.Water.from_pH_alk(v["pH"], v["alk"], v["ca"], temp_C=v["T"], ec_uS_cm=v["ec"])
        except ValueError as e:
            readout.value = f"<b style='color:{RED}'>Cannot calculate:</b> {e}"
            return
        phs = np.linspace(6, 9.5, 71)
        si = []
        for p in phs:
            try:
                si.append(cb.Water.from_pH_alk(p, v["alk"], v["ca"], temp_C=v["T"], ec_uS_cm=v["ec"]).SI)
            except ValueError:
                si.append(np.nan)
        pHs = w.pHs()
        if USE_PLOTLY:
            with fig.batch_update():
                fig.data[0].x, fig.data[0].y = phs, si
                fig.data[1].x, fig.data[1].y = [v["pH"]], [w.SI]
                fig.layout.shapes = [
                    dict(type="rect", x0=0, x1=1, y0=-4, y1=0, xref="paper", yref="y", layer="below",
                         fillcolor="rgba(229,57,53,0.12)", line=dict(width=0)),
                    dict(type="rect", x0=0, x1=1, y0=0, y1=2, xref="paper", yref="y", layer="below",
                         fillcolor="rgba(30,136,229,0.10)", line=dict(width=0)),
                    dict(type="line", x0=pHs, x1=pHs, y0=0, y1=1, xref="x", yref="paper",
                         line=dict(color=T("muted"), width=1, dash="dot"))]
                fig.layout.annotations = [
                    dict(x=0.02, y=-3.7, xref="paper", yref="y", text="SI < 0: dissolves CaCO₃ (aggressive)",
                         showarrow=False, xanchor="left", font=dict(size=11, color=T("muted"))),
                    dict(x=0.02, y=1.7, xref="paper", yref="y", text="SI > 0: can deposit CaCO₃",
                         showarrow=False, xanchor="left", font=dict(size=11, color=T("muted"))),
                    dict(x=pHs, y=1.0, xref="x", yref="paper", text=f"pHₛ = {pHs:.2f}", showarrow=False,
                         yanchor="bottom", font=dict(size=11, color=T("muted")))]
        else:
            def draw(plt):
                f, ax = plt.subplots(figsize=(6.2, 4.2))
                ax.axhspan(-4, 0, color="#e53935", alpha=0.12); ax.axhspan(0, 2, color="#1e88e5", alpha=0.10)
                ax.plot(phs, si, color=COL["marble"], lw=2.5); ax.plot(v["pH"], w.SI, "o", color="0.5", ms=9)
                ax.axvline(pHs, ls=":", color="0.5"); ax.text(pHs, 1.75, f" pHs = {pHs:.2f}", fontsize=9)
                ax.set_xlim(6, 9.5); ax.set_ylim(-4, 2); ax.set_xlabel("pH"); ax.set_ylabel("saturation index SI")
                f.tight_layout()
            _mpl_into(figure, draw)
        ok = "ok" if w.SI >= -0.1 else "bad"
        readout.value = cards([("SI (exact)", f"{nz(w.SI, 2):+.2f}", ok),
                               ("LSI = pH − pHₛ", f"{nz(w.LSI, 2):+.2f}", None),
                               ("pHₛ (slide formula)", f"{pHs:.2f}", None)]) + cards([
            ("CCPP", f"{nz(w.CCPP(), 1):+.1f} mg CaCO₃/L", None),
            ("ionic strength", f"{w.I * 1000:.1f} mmol/L", None)]) + (
            "<div><small>CCPP &lt; 0: this much CaCO₃ can still dissolve; &gt; 0: this much can precipitate.</small></div>")

    presets = {"Soft raw water": dict(pH=6.6, alk=0.10, ca=2.5, T=5, ec=40),
               "Treated water": dict(pH=8.3, alk=1.0, ca=20, T=5, ec=200),
               "…heated to 60 °C": dict(pH=7.9, alk=1.0, ca=20, T=60, ec=200),
               "Hard European water": dict(pH=7.6, alk=4.0, ca=110, T=12, ec=600)}
    pbox = wire(ctrls, refresh, presets)
    refresh()
    panel("Calcium carbonate saturation of a water",
          "The curve shows the saturation index the same water would have at other pH values; where it crosses "
          "zero is pHₛ. Change calcium, alkalinity and temperature and watch pHₛ and the CCPP.",
          list(ctrls.values()), figure, readout, pbox,
          symbols='SI = saturation index (exact, with activities) · pHₛ = saturation pH (Langelier, with ionic-strength correction) · LSI = pH − pHₛ · CCPP = calcium carbonate precipitation potential, mg CaCO<sub>3</sub>/L that can still dissolve (−) or can precipitate (+) · ionic strength estimated from the conductivity.')


def marble_demo_panel():
    """A5: dissolving marble moves the water AND its saturation line (calcium!)."""
    w0 = WATERS["Soft surface water (example)"].apply(cb.co2_dose(16))
    w0 = cb.Water(w0.temp_C, w0.I, w0.Ca, w0.Alk, w0.CT, label="after 16 mg CO2/L")
    max_mg = -w0.CCPP()
    ctrls = {"mg": pslider(0, 0, float(np.floor(max_mg)), 1, "CaCO₃ dissolved (mg/L)", ".0f")}
    readout = widgets.HTML()
    title = "Soft surface water + 16 mg CO<sub>2</sub>/L, then marble"

    def water(dissolved):
        if dissolved >= np.floor(max_mg):
            return w0.apply(cb.marble_to_equilibrium()), max_mg
        return (w0.apply(cb.caco3(dissolved)) if dissolved > 0 else w0), dissolved

    if USE_PLOTLY:
        d = DeffeyesPlotly(n_sat=2, height=500, width=620)
        figure = d.fig
    else:
        figure = widgets.Output()

    def refresh():
        w, dissolved = water(ctrls["mg"].value)
        steps = [w0, w] if ctrls["mg"].value > 0 else [w0]
        if USE_PLOTLY:
            d.update(5, w.I, 1.25, 1.25, ca_sat=[w0.ca_mg, w.ca_mg] if w.ca_mg - w0.ca_mg > 0.5 else [w0.ca_mg],
                     steps=steps, title=title, start_label="start")
        else:
            cas = [w0.ca_mg, w.ca_mg] if w.ca_mg - w0.ca_mg > 0.5 else [w0.ca_mg]
            _mpl_into(figure, lambda plt: _deffeyes_body_mpl(5, w.I, 1.25, 1.25, cas, steps))
        readout.value = cards([("dissolved", f"{dissolved:.0f} of {max_mg:.0f} mg/L", None),
                               ("pH", f"{w.pH:.2f}", None), ("Ca", f"{w.ca_mg:.1f} mg/L", None)]) + cards([
            ("SI", f"{nz(w.SI, 2):+.2f}", "ok" if w.SI >= -0.1 else "bad"),
            ("CCPP", f"{nz(w.CCPP(), 1):+.1f} mg/L", None)])

    presets = {"Start": dict(mg=0), "Halfway": dict(mg=float(np.floor(max_mg / 2))),
               "Equilibrium": dict(mg=float(np.floor(max_mg)))}
    pbox = wire(ctrls, refresh, presets)
    refresh()
    panel("Calcium is the hidden third variable",
          "Dissolve marble step by step. The water moves up and to the right – and because its calcium rises, "
          "its saturation line (dashed) moves towards it. At equilibrium they meet.",
          list(ctrls.values()), figure, readout, pbox,
          symbols='C<sub>T</sub> = total inorganic carbon · dashed lines: calcite saturation for the calcium concentration given in the legend · SI = saturation index · CCPP = mg CaCO<sub>3</sub>/L that can still dissolve (−).')


def _deffeyes_body_mpl(temp_C, I, ct_max, alk_max, ca_sat, steps, arrows=None, alk_min=0.0):
    """matplotlib Deffeyes for the fallback of the chapter panels (draws into current figure)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 5))
    cb.deffeyes(ax, temp_C=temp_C, I=I, ct_max=ct_max, alk_max=alk_max, ca_sat=list(ca_sat))
    if alk_min < 0:
        ax.set_ylim(alk_min, alk_max)
        ax.axhline(0, color="0.5", lw=0.8)
    if steps:
        cb.plot_path(ax, steps)
    for a, b, col, text in (arrows or []):
        ax.annotate("", xy=(b.ct_mmol, b.alk_mmol), xytext=(a.ct_mmol, a.alk_mmol),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))


# ===========================================================================
# A6 – design with vectors
# ===========================================================================
def vector_design_panel():
    """A6: CO2 + marble to a chosen target point, constructed with two vectors."""
    ctrls = {"water": widgets.Dropdown(options=list(WATERS), description="Raw water", style=PSTYLE, layout=PWIDE),
             "pH": pslider(8.3, 7.8, 8.8, 0.05, "Target pH", ".2f"),
             "alk": pslider(1.0, 0.4, 1.6, 0.05, "Target alkalinity", ".2f")}
    readout = widgets.HTML()
    if USE_PLOTLY:
        d = DeffeyesPlotly(n_sat=1, height=500, width=620)
        figure = d.fig
    else:
        figure = widgets.Output()

    def refresh():
        v = {k: c.value for k, c in ctrls.items()}
        w = WATERS[v["water"]]
        K = cb._Consts(w.temp_C, w.I)
        ct_t = 1000 * cb.CT_from_pH_alk(v["pH"], v["alk"] / 1000, K)
        marble = (v["alk"] - w.alk_mmol) / 2               # mmol/L CaCO3
        if marble < 0:
            readout.value = (f"<b style='color:{RED}'>Not possible with marble:</b> the target alkalinity is "
                             "below the raw water's. Marble can only add alkalinity.")
            return
        co2 = ct_t - marble - w.ct_mmol                    # mmol/L, negative = stripping
        step1 = w.apply(cb.co2_dose(co2 * 44.01) if co2 >= 0 else cb.co2_strip(-co2 * 44.01))
        step2 = step1.apply(cb.caco3(marble * 100.09))
        ws = [w, step1, step2]
        ct_max, alk_max = axis_limits(ws)
        if USE_PLOTLY:
            d.update(w.temp_C, step2.I, ct_max, alk_max, ca_sat=[step2.ca_mg], steps=ws,
                     title=f"{v['water']}: CO<sub>2</sub> + marble")
        else:
            _mpl_into(figure, lambda plt: _deffeyes_body_mpl(w.temp_C, step2.I, ct_max, alk_max, [step2.ca_mg], ws))
        label1 = "1  →  CO₂ dose" if co2 >= 0 else "1  ←  strip CO₂"
        readout.value = cards([(label1, f"{abs(co2) * 44.01:.1f} mg/L", None),
                               ("2  ↗  marble", f"{marble * 100.09:.1f} mg CaCO₃/L", None)]) + cards([
            ("final pH", f"{step2.pH:.2f}", None), ("final Ca", f"{step2.ca_mg:.1f} mg/L",
                                                     "ok" if 15 <= step2.ca_mg <= 25 else "bad"),
            ("SI", f"{nz(step2.SI, 2):+.2f}", "ok" if step2.SI >= -0.1 else "bad")]) + (
            "<div style='margin-top:4px'><small>The arrows reach the target point exactly. Whether the water is "
            "then saturated is a separate check (SI): see the dashed saturation line.</small></div>")

    presets = {"Soft water → pH 8.3, Alk 1.0": dict(water="Soft surface water (example)", pH=8.3, alk=1.0),
               "Soft water → pH 8.5, Alk 1.1": dict(water="Soft surface water (example)", pH=8.5, alk=1.1),
               "Groundwater → pH 8.3, Alk 1.3": dict(water="CO2-rich groundwater (example)", pH=8.3, alk=1.3)}
    pbox = wire(ctrls, refresh, presets)
    refresh()
    panel("Design with vectors: CO₂ first, then marble",
          "Choose a target point. Arrow 1 (CO₂, horizontal) and arrow 2 (marble, slope 2) are constructed so "
          "that they end exactly there. Read the doses – and check whether the result is saturated.",
          list(ctrls.values()), figure, readout, pbox,
          symbols='Arrow 1: CO<sub>2</sub> dosing (→) or stripping (←), changes only C<sub>T</sub> · Arrow 2: marble (CaCO<sub>3</sub>) dissolution, slope 2 · dashed line: calcite saturation for the final calcium · SI = saturation index at the end point.')


# ===========================================================================
# A7 – treatment explorer with presets
# ===========================================================================
def explorer_panel():
    """A7: combine treatment steps (coagulant, stripping, CO2, marble, lime, NaOH)."""
    ctrls = {
        "water": widgets.Dropdown(options=list(WATERS), description="Raw water", style=PSTYLE, layout=PWIDE),
        "fe": pslider(0, 0, 10, 0.5, "0  Fe coagulant (mg Fe/L)", ".1f"),
        "strip": pslider(0, 0, 40, 0.5, "1  CO₂ stripped (mg/L)", ".1f"),
        "co2": pslider(0, 0, 40, 0.5, "2  CO₂ dosed (mg/L)", ".1f"),
        "marble": widgets.Dropdown(options=["no marble filter", "fixed amount", "to equilibrium"],
                                   description="3  Marble filter", style=PSTYLE, layout=PWIDE),
        "marble_mg": pslider(20, 0, 100, 1, "    CaCO₃ dissolved (mg/L)", ".0f"),
        "lime": pslider(0, 0, 50, 0.5, "4  Lime (mg Ca(OH)₂/L)", ".1f"),
        "naoh": pslider(0, 0, 30, 0.1, "5  NaOH (mg/L)", ".1f")}
    readout = widgets.HTML()
    below = widgets.Output()
    if USE_PLOTLY:
        d = DeffeyesPlotly(n_sat=2, height=520, width=640)
        figure = d.fig
    else:
        figure = widgets.Output()

    def refresh():
        v = {k: c.value for k, c in ctrls.items()}
        w = WATERS[v["water"]]
        notes, ws = [], [w]
        if v["fe"] > 0:
            ws.append(w.apply(cb.coagulant(v["fe"], "Fe")))
        rest, rest_notes = explore_path(v["water"], v["strip"], v["co2"], v["marble"], v["marble_mg"],
                                        v["lime"], v["naoh"], start=ws[-1])
        ws += rest[1:]
        notes += rest_notes
        final = ws[-1]
        ct_max, alk_max = axis_limits(ws)
        alk_min = min(0.0, min(x.alk_mmol for x in ws) - 0.05)
        cas = [w.ca_mg] + ([final.ca_mg] if abs(final.ca_mg - w.ca_mg) > 0.5 else [])
        if USE_PLOTLY:
            d.update(w.temp_C, final.I, ct_max, alk_max, ca_sat=cas, steps=ws,
                     title=f"{v['water']}, {w.temp_C:g} °C")
            d.fig.layout.yaxis.range = [alk_min, alk_max]
        else:
            _mpl_into(figure, lambda plt: _deffeyes_body_mpl(w.temp_C, final.I, ct_max, alk_max, cas, ws,
                                                             alk_min=alk_min))
        (p0, p1), (a0, a1), (c0, c1) = cb.TARGETS["pH"], cb.TARGETS["Alk"], cb.TARGETS["Ca"]
        readout.value = "<b>Final water</b>" + cards([
            ("pH", f"{final.pH:.2f}", "ok" if p0 <= final.pH <= p1 else "bad"),
            ("Alkalinity", f"{final.alk_mmol:.2f} mmol/L", "ok" if a0 <= final.alk_mmol <= a1 else "bad"),
            ("Calcium", f"{final.ca_mg:.1f} mg/L", "ok" if c0 <= final.ca_mg <= c1 else "bad"),
            ("SI", f"{nz(final.SI, 2):+.2f}", "ok" if final.SI >= -0.1 else "bad")]) + "".join(
            f"<div><small><i>Note: {n}</i></small></div>" for n in notes)
        below.clear_output(wait=True)
        with below:
            display(table(ws))

    S, G = "Soft surface water (example)", "CO2-rich groundwater (example)"
    zero = dict(fe=0, strip=0, co2=0, marble="no marble filter", marble_mg=20, lime=0, naoh=0)
    presets = {"CO₂ + marble": dict(zero, water=S, co2=17, marble="to equilibrium"),
               "NaOH only": dict(zero, water=S, naoh=3.1),
               "Lime + CO₂": dict(zero, water=S, co2=32.5, lime=30),
               "Fe coagulation + marble (Molde)": dict(zero, water=S, fe=5, co2=4.5, marble="to equilibrium"),
               "Groundwater: marble only": dict(zero, water=G, marble="to equilibrium"),
               "Groundwater: strip + marble": dict(zero, water=G, strip=24, marble="to equilibrium"),
               "Reset": dict(zero, water=S)}
    pbox = wire(ctrls, refresh, presets)
    refresh()
    panel("Treatment explorer",
          "Steps are applied in the order 0 → 5. Load a Norwegian treatment concept with a preset, then change "
          "doses. Green cards: final water within the NV 274 target. The table below lists every step.",
          list(ctrls.values()), figure, readout, pbox, below=below,
          symbols='Steps 0–5 are applied in this order · C<sub>T</sub> = total inorganic carbon · SI = saturation index · CCPP = calcium carbonate precipitation potential · the table below lists every step, its abbreviations are explained under the table.')
