"""
design_views.py - interactive panels for Part B (worked design example).

Calculations: design_example.py. Look and helpers (panels, sliders, cards,
self-checks, dark mode): stab_views.py. Set sv.DARK before the panels are
drawn, exactly as in Part A.
"""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML
import carbonate as cb
import stab_views as sv
import design_example as de

B = de.BASIS
ISTYLE = {"description_width": "130px"}


def _islider(value, vmin, vmax, text):
    return widgets.IntSlider(value=value, min=vmin, max=vmax, step=1, description=text,
                             description_allow_html=True, style=ISTYLE, layout=sv.PWIDE,
                             continuous_update=False)


def _ok(cond):
    return "ok" if cond else "bad"


def _note(text):
    return f"<div style='margin-top:4px'><small>{text}</small></div>"


def sources_box(items):
    """Small framed list 'Where the numbers come from'. items: (what, source)."""
    rows = "".join(f"<tr><td style='padding:1px 10px 1px 0;vertical-align:top'>{a}</td>"
                   f"<td style='padding:1px 0;opacity:0.85'>{b}</td></tr>" for a, b in items)
    display(HTML(f"<div style='border:1px solid {sv.BORDER};border-left:4px solid #8c564b;"
                 f"padding:6px 12px;margin:8px 0'><b>Where the numbers come from</b>"
                 f"<table style='font-size:0.9em'>{rows}</table></div>"))


def frame_table(rows, columns, legend=""):
    """A plain HTML table (no index) with an optional legend below."""
    df = pd.DataFrame(rows, columns=columns)
    leg = f"<div style='font-size:0.85em;opacity:0.85'>{legend}</div>" if legend else ""
    display(HTML(df.to_html(index=False, escape=False) + leg))


# ===========================================================================
# B1 – design basis: the raw water over the year
# ===========================================================================
def basis_panel():
    """B1: the same raw water at different temperatures and within the assumed seasonal range."""
    ctrls = {"T": sv.pslider(5, B.T_min, B.T_max, 1, "Temperature (°C)", ".0f"),
             "pH": sv.pslider(B.pH, 6.2, 7.0, 0.05, "Raw pH (lab, 5 °C)", ".2f"),
             "alk": sv.pslider(B.alk, 0.03, 0.30, 0.01, "Raw alkalinity (mmol/L)", ".2f")}
    readout = widgets.HTML()
    if sv.USE_PLOTLY:
        d = sv.DeffeyesPlotly(n_sat=1, height=480, width=600)
        figure = d.fig
    else:
        figure = widgets.Output()

    def refresh():
        v = {k: c.value for k, c in ctrls.items()}
        try:
            w = B.raw(temp_C=v["T"], pH=v["pH"], alk=v["alk"])
        except ValueError as e:
            readout.value = f"<b style='color:{sv.RED}'>Cannot calculate:</b> {e}"
            return
        ct_max, alk_max = 1.5, 1.2
        if sv.USE_PLOTLY:
            d.update(v["T"], w.I, ct_max, alk_max, ca_sat=[w.ca_mg], points=[(w, "raw water")],
                     title=f"Raw water at {v['T']:g} °C and the target zone")
        else:
            sv._mpl_into(figure, lambda plt: sv._deffeyes_body_mpl(v["T"], w.I, ct_max, alk_max, [w.ca_mg], [w]))
        readout.value = cards_raw(w)

    presets = {"Design raw water": dict(T=5, pH=B.pH, alk=B.alk),
               "Winter, 2 °C": dict(T=2, pH=B.pH, alk=B.alk),
               "Snowmelt (low Alk)": dict(T=2, pH=6.4, alk=0.05),
               "Late summer, 12 °C": dict(T=12, pH=6.8, alk=0.15)}
    pbox = sv.wire(ctrls, refresh, presets)
    refresh()
    sv.panel("The raw water over the year",
             "The lab measured the sample at 5 °C. The same water (same C<sub>T</sub> and alkalinity) has another pH "
             "at 2 or 12 °C. Compare the raw water with the target zone – how far away is it?",
             list(ctrls.values()), figure, readout, pbox,
             symbols="C<sub>T</sub> = total inorganic carbon · Alk = alkalinity · SI = saturation index · "
                     "green zone: NV 274 target pH 8.0–8.5, Alk 0.6–1.0 mmol/L · dashed line: calcite saturation "
                     "for the raw-water calcium.")


def cards_raw(w):
    return sv.cards([("pH at this temperature", f"{w.pH:.2f}", "bad"),
                     ("C<sub>T</sub>", f"{w.ct_mmol:.2f} mmol/L", None),
                     ("free CO₂", f"{w.co2_mg:.1f} mg/L", None)]) + sv.cards([
        ("Alkalinity", f"{w.alk_mmol:.2f} mmol/L", "bad"),
        ("Calcium", f"{w.ca_mg:.1f} mg/L", "bad"),
        ("SI", f"{sv.nz(w.SI, 2):+.2f}", "bad")]) + _note(
        "Red: outside the target (NV 274/2022). All three need to rise: pH, alkalinity and calcium.")


# ===========================================================================
# B3 – doses from the target point
# ===========================================================================
def dose_panel():
    """B3: CO2 dose and marble dissolution from a chosen target point; checks at 2 and 12 °C."""
    ctrls = {"pH_t": sv.pslider(B.pH_target, 7.9, 8.6, 0.05, "Target pH", ".2f"),
             "alk_t": sv.pslider(B.alk_target, 0.6, 1.2, 0.05, "Target Alk (mmol/L)", ".2f"),
             "Td": sv.pslider(B.T_design, B.T_min, B.T_max, 1, "Target met at (°C)", ".0f"),
             "pH": sv.pslider(B.pH, 6.2, 7.0, 0.05, "Raw pH (5 °C)", ".2f"),
             "alk": sv.pslider(B.alk, 0.03, 0.30, 0.01, "Raw Alk (mmol/L)", ".2f")}
    readout = widgets.HTML()
    below = widgets.Output()
    if sv.USE_PLOTLY:
        d = sv.DeffeyesPlotly(n_sat=1, height=520, width=620)
        figure = d.fig
    else:
        figure = widgets.Output()

    def refresh():
        v = {k: c.value for k, c in ctrls.items()}
        try:
            raw = B.raw(pH=v["pH"], alk=v["alk"])
            r = de.doses(raw, v["pH_t"], v["alk_t"], v["Td"])
            eq = de.equilibrium_limit(raw, max(r["co2_mg"], 0), v["Td"]) if r["co2_mg"] >= 0 else None
        except ValueError as e:
            readout.value = f"<b style='color:{sv.RED}'>Cannot calculate:</b> {e}"
            return
        ws = r["waters"]
        final = ws[-1]
        ct_max, alk_max = sv.axis_limits(ws + ([eq] if eq else []), ct_min=1.5, alk_min=1.2)
        pts = [(eq, "equilibrium limit")] if eq else []
        if sv.USE_PLOTLY:
            d.update(v["Td"], final.I, ct_max, alk_max, ca_sat=[final.ca_mg], steps=ws, points=pts,
                     title=f"CO<sub>2</sub> + marble to the target point, {v['Td']:g} °C")
        else:
            sv._mpl_into(figure, lambda plt: sv._deffeyes_body_mpl(v["Td"], final.I, ct_max, alk_max,
                                                                  [final.ca_mg], ws))
        w2 = de.treated(raw, r["co2_mg"], r["marble_mg"], B.T_min)
        w12 = de.treated(raw, r["co2_mg"], r["marble_mg"], B.T_max)
        lo, hi = de.nv_table_12_1(v["alk"])
        frac = (v["alk_t"] - raw.alk_mmol) / (eq.alk_mmol - raw.alk_mmol) if eq else float("nan")
        co2_txt = f"{r['co2_mg']:.1f} mg/L" if r["co2_mg"] >= 0 else f"strip {-r['co2_mg']:.1f} mg/L"
        p0, p1 = cb.TARGETS["pH"]
        readout.value = sv.cards([("1 → CO₂ dose", co2_txt, None),
                                  ("2 ↗ marble dissolved", f"{r['marble_mg']:.0f} mg CaCO₃/L", None),
                                  ("NV 212 Table 12.1", f"{lo:.0f}–{hi:.0f} mg CO₂/L", None)]) + sv.cards([
            ("pH at 2 °C", f"{w2.pH:.2f}", _ok(p0 <= w2.pH <= p1)),
            ("pH at 12 °C", f"{w12.pH:.2f}", _ok(p0 <= w12.pH <= p1)),
            ("Calcium", f"{final.ca_mg:.1f} mg/L", _ok(15 <= final.ca_mg <= 25))]) + sv.cards([
            ("SI at 2 / 12 °C", f"{sv.nz(w2.SI, 2):+.2f} / {sv.nz(w12.SI, 2):+.2f}", _ok(w2.SI < -0.05)),
            ("share of equilibrium", f"{100 * frac:.0f} %", None)]) + _note(
            "SI must stay clearly below 0 at the target point (green: SI &lt; −0.05) – otherwise marble cannot "
            "dissolve that far. "
            "'Share of equilibrium': how much of the marble that <i>could</i> dissolve (after this CO₂ dose) "
            "the filter has to dissolve.")
        below.clear_output(wait=True)
        with below:
            display(sv.table(ws + ([replace(eq, label="(limit) marble to equilibrium")] if eq else [])))

    presets = {"Design: pH 8.3, Alk 1.0 at 2 °C": dict(pH_t=8.3, alk_t=1.0, Td=2, pH=B.pH, alk=B.alk),
               "Same target, met at 12 °C": dict(pH_t=8.3, alk_t=1.0, Td=12, pH=B.pH, alk=B.alk),
               "Lower Alk 0.8": dict(pH_t=8.3, alk_t=0.8, Td=2, pH=B.pH, alk=B.alk),
               "Snowmelt raw water": dict(pH_t=8.3, alk_t=1.0, Td=2, pH=6.4, alk=0.05),
               "Too ambitious: pH 8.6": dict(pH_t=8.6, alk_t=1.0, Td=2, pH=B.pH, alk=B.alk)}
    pbox = sv.wire(ctrls, refresh, presets)
    refresh()
    sv.panel("Doses from the target point",
             "Choose the target point. Arrow 1 (CO₂) and arrow 2 (marble, slope 2) end exactly there. The cards "
             "check the result at the coldest and warmest water and compare the CO₂ dose with NV 212 Table 12.1.",
             list(ctrls.values()), figure, readout, pbox, below=below,
             symbols="Arrow 1: CO<sub>2</sub> dosing (changes only C<sub>T</sub>) · Arrow 2: CaCO<sub>3</sub> "
                     "dissolved in the marble filter (slope 2) · 'equilibrium limit': the point an infinitely long "
                     "filter would reach after the same CO<sub>2</sub> dose · dashed line: calcite saturation for "
                     "the final calcium · SI = saturation index.")


def temperature_table(raw=None):
    """B3: the designed water at the temperatures of the year."""
    D = de.main_design()
    raw = raw or D["raw"]
    rows = []
    for T in (2, 5, 8, 12, 20):
        w = de.treated(raw, D["doses"]["co2_mg"], D["doses"]["marble_mg"], T)
        rows.append([f"{T}", f"{w.pH:.2f}", f"{w.alk_mmol:.2f}", f"{w.ca_mg:.1f}",
                     f"{sv.nz(w.SI, 2):+.2f}", f"{sv.nz(w.CCPP(), 1):+.1f}",
                     "in the house (warm-water side starts above this)" if T == 20 else ""])
    frame_table(rows, ["T (°C)", "pH", "Alk (mmol/L)", "Ca (mg/L)", "SI", "CCPP (mg CaCO<sub>3</sub>/L)", "note"],
                "Same doses (C<sub>T</sub>, alkalinity and calcium unchanged); only the temperature differs. "
                "CCPP = calcium carbonate precipitation potential (−: can still dissolve, +: can precipitate).")


def dose_range_table():
    rows = []
    for (p, a), co2 in de.dose_range().items():
        rows.append([f"{p}", f"{a:.2f}", f"{co2:.1f}"])
    frame_table(rows, ["raw pH (5 °C)", "raw Alk (mmol/L)", "CO<sub>2</sub> dose (mg/L)"],
                "Same target point (pH 8.3, Alk 1.0 mmol/L at 2 °C) for the four corners of the assumed raw-water "
                "range (B1). Marble dissolved: (1.0 − raw Alk)/2 mmol/L.")


# ===========================================================================
# B4 – marble filter dimensioning
# ===========================================================================
LAYERS = [("support", "gravel support (assumption)", "#9e9e9e"),
          ("depth", "marble, design depth", "#9467bd"),
          ("reserve", "marble reserve (dissolves)", "#c5b0d5"),
          ("water_above", "water above bed (assumption)", "#64b5f6"),
          ("freeboard", "freeboard", "#e0e0e0")]


def filter_panel():
    """B4: number of filters, EBCT and filter rate -> area, depth, check values."""
    ctrls = {"Q": sv.pslider(B.Q_max, 30, 150, 5, "Q<sub>dim</sub> (L/s)", ".0f"),
             "n": _islider(4, 2, 8, "Number of filters"),
             "ebct": sv.pslider(20, 10, 40, 1, "EBCT, one out (min)", ".0f"),
             "v": sv.pslider(8, 4, 14, 0.5, "Rate, one out (m/h)", ".1f")}
    readout = widgets.HTML()
    if sv.USE_PLOTLY:
        import plotly.graph_objects as go
        fig = go.FigureWidget()
        for key, name, col in LAYERS:
            fig.add_bar(x=["one filter"], y=[0], name=name, marker=dict(color=col),
                        hovertemplate=name + ": %{y:.2f} m<extra></extra>")
        fig.update_layout(template=sv.T("template"), barmode="stack", height=430, width=420,
                          margin=dict(l=60, r=20, t=50, b=40),
                          title=dict(text="Filter box, cross-section", x=0.5, xanchor="center"),
                          yaxis=dict(title=dict(text="height above floor (m)")),
                          legend=dict(traceorder="reversed", bgcolor=sv.T("legend")))
        sv._style(fig)
        figure = fig
    else:
        figure = widgets.Output()

    def refresh():
        v = {k: c.value for k, c in ctrls.items()}
        fd = de.FilterDesign(v["Q"] * 3.6, n=int(v["n"]), ebct_min=v["ebct"], v=v["v"])
        r = fd.calc(B.Q_avg * 3.6)
        h = dict(support=fd.support, depth=r["depth"], reserve=fd.reserve, water_above=fd.water_above,
                 freeboard=fd.freeboard)
        if sv.USE_PLOTLY:
            with fig.batch_update():
                for i, (key, _, _) in enumerate(LAYERS):
                    fig.data[i].x, fig.data[i].y = ["one filter"], [h[key]]
                fig.layout.yaxis.range = [0, 7]
        else:
            def draw(plt):
                f, ax = plt.subplots(figsize=(3.6, 4.4))
                bottom = 0
                for key, name, col in LAYERS:
                    ax.bar(["one filter"], [h[key]], bottom=bottom, color=col, label=name)
                    bottom += h[key]
                ax.set_ylim(0, 7); ax.set_ylabel("height above floor (m)")
                ax.legend(fontsize=7, loc="upper right"); f.tight_layout()
            sv._mpl_into(figure, draw)
        readout.value = sv.cards([
            ("area per filter", f"{r['area_each']:.1f} m² (≈ {r['side']:.1f} × {r['side']:.1f} m)", _ok(r["area_each"] < 100)),
            ("total area", f"{r['area_total']:.0f} m²", None),
            ("marble depth", f"{r['depth']:.1f} m (+ {fd.reserve:.1f} reserve)", _ok(r["depth"] >= 2.5))]) + sv.cards([
            ("rate, one out", f"{r['v_one_out']:.1f} m/h", _ok(r["v_one_out"] <= 10)),
            ("rate, all in", f"{r['v_all']:.1f} m/h", None),
            ("EBCT, one out", f"{r['ebct_one_out']:.0f} min", _ok(r["ebct_one_out"] >= 20)),
            ("EBCT, all in", f"{r['ebct_all']:.0f} min", None)]) + sv.cards([
            ("marble, first fill", f"{r['marble_t']:.0f} t", None),
            ("backwash flow", f"{r['Q_bw_m3h']:.0f} m³/h", None),
            ("box height", f"{r['box_height']:.1f} m", None)]) + (sv.cards([
            (f"EBCT one out at Q<sub>max</sub> = {B.Q_max:g} L/s",
             f"{r['depth'] * r['area_each'] * (fd.n - 1) / (B.Q_max * 3.6) * 60:.0f} min",
             _ok(r['depth'] * r['area_each'] * (fd.n - 1) / (B.Q_max * 3.6) * 60 >= 20))])
            if v["Q"] < B.Q_max else "") + _note(
            "Green/red: checked against NV 212 (area &lt; 100 m² Table 4.16; depth ≥ 2.5 m Table 12.3; rate ≤ 10 m/h "
            "§12.2.2.4; EBCT ≥ 20 min Table 12.3). Depth = rate × EBCT, rounded up; area rounded up to 0.5 m².")

    presets = {"Design: 4 filters, 20 min, 8 m/h": dict(Q=B.Q_max, n=4, ebct=20, v=8),
               "Only 2 filters": dict(Q=B.Q_max, n=2, ebct=20, v=8),
               "Faster: 10 m/h": dict(Q=B.Q_max, n=4, ebct=20, v=10),
               "Cautious: 30 min": dict(Q=B.Q_max, n=4, ebct=30, v=8),
               "Q_avg instead of Q_max (wrong!)": dict(Q=B.Q_avg, n=4, ebct=20, v=8)}
    pbox = sv.wire(ctrls, refresh, presets)
    refresh()
    sv.panel("Marble filter dimensioning",
             "Size the filters for Q<sub>dim</sub> with one filter out of service (backwash or repair). "
             "Choose the number of filters, the minimum EBCT and the filter rate; the figure shows the resulting "
             "filter box.",
             list(ctrls.values()), figure, readout, pbox,
             symbols="Q<sub>dim</sub> = design flow (max hour of the max day) · EBCT = empty bed contact time = "
                     "bed volume / flow · rate = filter velocity v = flow / filter area (m³/(m²·h) = m/h) · "
                     "'one out' = one filter out of service, the others take the whole flow.")


# ===========================================================================
# B5 – chemicals and costs (estimates)
# ===========================================================================
def cost_panel():
    """B5: consumption, storage and running costs (ESTIMATES)."""
    D0 = de.main_design()
    ctrls = {"pco2": sv.pslider(3.0, 1.0, 6.0, 0.25, "CO₂ price (NOK/kg)", ".2f"),
             "pm": sv.pslider(2.0, 1.0, 4.0, 0.25, "Marble price (NOK/kg)", ".2f"),
             "util": sv.pslider(0.90, 0.6, 1.0, 0.05, "CO₂ utilisation (–)", ".2f"),
             "days": sv.pslider(30, 7, 60, 1, "CO₂ storage (days)", ".0f"),
             "Qavg": sv.pslider(B.Q_avg, 20, 80, 5, "Q<sub>avg</sub> (L/s)", ".0f")}
    readout = widgets.HTML()
    if sv.USE_PLOTLY:
        import plotly.graph_objects as go
        fig = go.FigureWidget()
        fig.add_bar(x=["CO₂", "marble"], y=[0, 0], marker=dict(color=[sv.COL["CO2"], sv.COL["marble"]]),
                    hovertemplate="%{x}: %{y:.0f} kNOK/year<extra></extra>", showlegend=False)
        fig.update_layout(template=sv.T("template"), height=380, width=440, margin=dict(l=70, r=20, t=50, b=40),
                          title=dict(text="Chemical costs per year (ESTIMATE)", x=0.5, xanchor="center"),
                          yaxis=dict(title=dict(text="kNOK/year"), rangemode="tozero"))
        sv._style(fig)
        figure = fig
    else:
        figure = widgets.Output()

    def refresh():
        v = {k: c.value for k, c in ctrls.items()}
        b = replace(B, Q_avg=v["Qavg"])
        d = D0["doses"]
        ch = de.Chemicals(d["co2_mg"], max(D0["range"].values()), min(D0["range"].values()), d["marble_mg"],
                          utilisation=v["util"], storage_days=v["days"], price_co2=v["pco2"],
                          price_marble=v["pm"]).calc(b, D0["filter"])
        y = [ch["cost_co2"] / 1000, ch["cost_marble"] / 1000]
        if sv.USE_PLOTLY:
            with fig.batch_update():
                fig.data[0].x, fig.data[0].y = ["CO₂", "marble"], y
        else:
            def draw(plt):
                f, ax = plt.subplots(figsize=(4.4, 3.6))
                ax.bar(["CO₂", "marble"], y, color=[sv.COL["CO2"], sv.COL["marble"]])
                ax.set_ylabel("kNOK/year"); ax.set_title("Chemical costs per year (ESTIMATE)", fontsize=10)
                f.tight_layout()
            sv._mpl_into(figure, draw)
        readout.value = sv.cards([("CO₂ use", f"{ch['co2_kg_d']:.0f} kg/d", None),
                                  ("marble use", f"{ch['marble_kg_d']:.0f} kg/d", None),
                                  ("water per year", f"{ch['V_year'] / 1e6:.2f} mill. m³", None)]) + sv.cards([
            ("CO₂ tank", f"{ch['co2_storage_t']:.1f} t ≈ {ch['co2_storage_m3']:.1f} m³", None),
            ("refill marble every", f"{ch['refill_days']:.0f} days", None)]) + sv.cards([
            ("chemicals, total", f"{ch['cost_total'] / 1000:.0f} kNOK/year", None),
            ("per m³", f"{ch['cost_per_m3']:.2f} NOK/m³", None),
            ("first marble fill", f"{ch['first_fill_cost'] / 1000:.0f} kNOK", None)]) + _note(
            "<b>All prices are ESTIMATES</b> (see the text above for where they come from). The quantities (kg/d) "
            "are design values; the costs are only as good as the prices.")

    presets = {"Base case": dict(pco2=3.0, pm=2.0, util=0.9, days=30, Qavg=B.Q_avg),
               "Cheap chemicals": dict(pco2=2.0, pm=1.5, util=0.9, days=30, Qavg=B.Q_avg),
               "Expensive chemicals": dict(pco2=5.0, pm=3.0, util=0.9, days=30, Qavg=B.Q_avg),
               "Poor CO₂ injection (70 %)": dict(pco2=3.0, pm=2.0, util=0.7, days=30, Qavg=B.Q_avg)}
    pbox = sv.wire(ctrls, refresh, presets)
    refresh()
    sv.panel("Consumption, storage and running costs",
             "The doses come from B3, the filter from B4. Change the prices and the CO₂ utilisation and see "
             "which chemical dominates the running cost.",
             list(ctrls.values()), figure, readout, pbox,
             symbols="CO<sub>2</sub> utilisation = share of the dosed CO<sub>2</sub> that dissolves in the water · "
                     "Q<sub>avg</sub> = average flow (used for consumption and costs; Q<sub>max</sub> is used for "
                     "the equipment size) · kNOK = 1000 NOK.")


# ===========================================================================
# B6 – the link to disinfection
# ===========================================================================
def hocl_panel():
    """B6: share of free chlorine present as HOCl versus pH (relevant if chlorine is used after stabilisation)."""
    ctrls = {"pH": sv.pslider(8.3, 6.0, 9.0, 0.05, "pH", ".2f"),
             "T": sv.pslider(5, 0, 25, 1, "Temperature (°C)", ".0f")}
    readout = widgets.HTML()
    if sv.USE_PLOTLY:
        import plotly.graph_objects as go
        fig = go.FigureWidget()
        fig.add_scatter(x=[], y=[], mode="lines", line=dict(color=sv.COL["base"], width=3), showlegend=False,
                        hovertemplate="pH %{x:.2f}: %{y:.0f} % HOCl<extra></extra>")
        fig.add_scatter(x=[], y=[], mode="markers+text", textposition="top right", showlegend=False,
                        marker=dict(size=12, color=sv.T("ink")), hovertemplate="%{text}<extra></extra>")
        fig.update_layout(template=sv.T("template"), height=400, width=560, margin=dict(l=70, r=20, t=50, b=50),
                          title=dict(text="Free chlorine present as HOCl", x=0.5, xanchor="center"),
                          xaxis=dict(title=dict(text="pH"), range=[6, 9]),
                          yaxis=dict(title=dict(text="HOCl (% of free chlorine)"), range=[0, 100]))
        sv._style(fig)
        figure = fig
    else:
        figure = widgets.Output()

    def refresh():
        v = {k: c.value for k, c in ctrls.items()}
        ph = np.linspace(6, 9, 121)
        y = 100 * de.hocl_fraction(ph, v["T"])
        raw_f = 100 * de.hocl_fraction(B.pH, v["T"])
        now = 100 * de.hocl_fraction(v["pH"], v["T"])
        if sv.USE_PLOTLY:
            with fig.batch_update():
                fig.data[0].x, fig.data[0].y = ph, y
                fig.data[1].x, fig.data[1].y = [B.pH, v["pH"]], [raw_f, now]
                fig.data[1].text = [f"raw water {raw_f:.0f} %", f"pH {v['pH']:.2f}: {now:.0f} %"]
        else:
            def draw(plt):
                f, ax = plt.subplots(figsize=(5.6, 4))
                ax.plot(ph, y, lw=2.5); ax.plot([B.pH, v["pH"]], [raw_f, now], "o", color="0.4")
                ax.set_xlabel("pH"); ax.set_ylabel("HOCl (% of free chlorine)"); ax.set_ylim(0, 100)
                f.tight_layout()
            sv._mpl_into(figure, draw)
        readout.value = sv.cards([("pK<sub>a</sub> of HOCl", f"{de.pKa_hocl(v['T']):.2f}", None),
                                  ("HOCl at raw pH 6.6", f"{raw_f:.0f} %", None),
                                  ("HOCl at this pH", f"{now:.0f} %", None)]) + _note(
            "HOCl is the strong disinfectant; OCl⁻ is much weaker. NV 212 Table 11.1 therefore asks for a higher "
            "C·t when the pH is higher (viruses, 3 log, 4 °C: 4 at pH &lt; 7, 6 at pH 7–8, 8 mg·min/L at pH &gt; 8; "
            "lecture M08 slide 'pH and disinfection').")

    presets = {"Treated water, winter": dict(pH=8.3, T=2), "Raw water pH": dict(pH=6.6, T=5),
               "pH 7.5, 10 °C": dict(pH=7.5, T=10)}
    pbox = sv.wire(ctrls, refresh, presets)
    refresh()
    sv.panel("Why the pH after stabilisation matters for chlorination",
             "If chlorine is dosed after the marble filter, most of it is present as the weak OCl⁻. Move the pH "
             "and temperature.",
             list(ctrls.values()), figure, readout, pbox,
             symbols="HOCl = hypochlorous acid (strong disinfectant) · OCl⁻ = hypochlorite (weak) · pK<sub>a</sub> "
                     "= pH at which half is HOCl (Morris 1966) · C·t = disinfectant concentration × contact time.")


# ===========================================================================
# B7 – second example: groundwater, stripping + marble
# ===========================================================================
def groundwater_panel():
    """B7: tower stripping of part of the flow, blending, marble filter to equilibrium."""
    ctrls = {"target": sv.pslider(13, 4, 30, 0.5, "CO₂ after stripping (mg/L)", ".1f"),
             "E": sv.pslider(0.90, 0.70, 0.95, 0.01, "Tower efficiency E", ".2f"),
             "vf": sv.pslider(60, 40, 100, 5, "Surface load (m/h)", ".0f"),
             "lv": sv.pslider(10, 2, 40, 1, "Air/water ratio L/V", ".0f"),
             "nt": _islider(2, 1, 3, "Number of towers")}
    readout = widgets.HTML()
    below = widgets.Output()
    raw = de.gw_raw()
    if sv.USE_PLOTLY:
        d = sv.DeffeyesPlotly(n_sat=2, height=520, width=620)
        figure = d.fig
    else:
        figure = widgets.Output()

    def refresh():
        v = {k: c.value for k, c in ctrls.items()}
        s = de.stripping(raw, v["target"], v["E"], v["vf"], v["lv"], B.Q_dim, int(v["nt"]))
        blend = s["blend"]
        final = blend.apply(cb.marble_to_equilibrium())
        final = replace(final, label="marble filter (equilibrium)")
        ws = [raw, blend, final]
        ct_max, alk_max = sv.axis_limits(ws)
        if sv.USE_PLOTLY:
            d.update(raw.temp_C, final.I, ct_max, alk_max, ca_sat=[raw.ca_mg, final.ca_mg], steps=ws,
                     title=f"Groundwater: stripping + marble, {raw.temp_C:g} °C")
        else:
            sv._mpl_into(figure, lambda plt: sv._deffeyes_body_mpl(raw.temp_C, final.I, ct_max, alk_max,
                                                                  [raw.ca_mg, final.ca_mg], ws))
        (p0, p1), (a0, a1), (c0, c1) = cb.TARGETS["pH"], cb.TARGETS["Alk"], cb.TARGETS["Ca"]
        lv_ok = v["lv"] >= 3 * s["lv_min"]
        readout.value = sv.cards([("raw free CO₂", f"{s['c_in']:.1f} mg/L", None),
                                  ("out of tower", f"{s['c_tower']:.1f} mg/L", None),
                                  ("share through tower", f"{100 * s['fraction']:.0f} %", None)]) + sv.cards([
            ("tower diameter", f"{s['diameter']:.2f} m (× {int(v['nt'])})", None),
            ("air flow", f"{s['Q_air']:.0f} m³/h", None),
            ("minimum L/V", f"{s['lv_min']:.1f}", _ok(lv_ok))]) + "<b>After the marble filter</b>" + sv.cards([
            ("pH", f"{final.pH:.2f}", _ok(p0 <= final.pH <= p1)),
            ("Alkalinity", f"{final.alk_mmol:.2f} mmol/L", _ok(a0 <= final.alk_mmol <= a1)),
            ("Calcium", f"{final.ca_mg:.1f} mg/L", _ok(c0 <= final.ca_mg <= c1)),
            ("marble dissolved", f"{(final.alk_mmol - blend.alk_mmol) / 2 * cb.M['CaCO3']:.0f} mg/L", None)]) + _note(
            "Marble filter shown at its limit (equilibrium, SI = 0); a real filter stops a little short. "
            "Minimum L/V: the least air that could carry the stripped CO₂ away (Henry's law); green when the chosen "
            "L/V is at least 3 × this (assumption).")
        below.clear_output(wait=True)
        with below:
            display(sv.table([raw, s["tower"], blend, final]))

    presets = {"NV 212: strip to 13 mg/L": dict(target=13, E=0.90, vf=60, lv=10, nt=2),
               "Strip to 7 mg/L (Alk 1.0?)": dict(target=7, E=0.90, vf=60, lv=10, nt=2),
               "Strip only a little (25 mg/L)": dict(target=25, E=0.90, vf=60, lv=10, nt=2),
               "One tower, high load": dict(target=13, E=0.90, vf=100, lv=10, nt=1)}
    pbox = sv.wire(ctrls, refresh, presets)
    refresh()
    sv.panel("Groundwater: stripping tower + marble filter",
             "Part of the flow goes through the stripping tower, the rest bypasses it; the blend then passes the "
             "marble filter. Choose how much CO₂ should remain after stripping and size the tower.",
             list(ctrls.values()), figure, readout, pbox, below=below,
             symbols="E = stripping efficiency (c<sub>in</sub> − c<sub>out</sub>)/(c<sub>in</sub> − c<sub>sat</sub>) · "
                     "surface load = water flow per tower cross-section, m³/(m²·h) · L/V = air flow / water flow "
                     "(NV 212 notation) · c<sub>sat</sub> = CO<sub>2</sub> in equilibrium with air · table: raw → "
                     "tower outlet → blend → marble filter.")


# ===========================================================================
# B8 – design calculator, summary and CSV export
# ===========================================================================
def calculator_panel():
    """B8: all main inputs in one place; summary table with sources; CSV export."""
    ctrls = {"pH": sv.pslider(B.pH, 6.2, 7.0, 0.05, "Raw pH (5 °C)", ".2f"),
             "alk": sv.pslider(B.alk, 0.03, 0.30, 0.01, "Raw Alk (mmol/L)", ".2f"),
             "ca": sv.pslider(B.ca, 0.5, 8, 0.5, "Raw Ca (mg/L)", ".1f"),
             "pH_t": sv.pslider(B.pH_target, 7.9, 8.6, 0.05, "Target pH", ".2f"),
             "alk_t": sv.pslider(B.alk_target, 0.6, 1.2, 0.05, "Target Alk (mmol/L)", ".2f"),
             "Qavg": sv.pslider(B.Q_avg, 5, 150, 1, "Q<sub>avg</sub> (L/s)", ".0f"),
             "Qmax": sv.pslider(B.Q_max, 5, 250, 1, "Q<sub>max</sub> (L/s)", ".0f"),
             "Qmin": sv.pslider(B.Q_min, 2, 60, 1, "Q<sub>min</sub> (L/s)", ".0f"),
             "n": _islider(4, 2, 8, "Number of filters"),
             "ebct": sv.pslider(20, 10, 40, 1, "EBCT, one out (min)", ".0f"),
             "v": sv.pslider(8, 4, 14, 0.5, "Rate, one out (m/h)", ".1f"),
             "pco2": sv.pslider(3.0, 1.0, 6.0, 0.25, "CO₂ price (NOK/kg)", ".2f"),
             "pm": sv.pslider(2.0, 1.0, 4.0, 0.25, "Marble price (NOK/kg)", ".2f")}
    readout = widgets.HTML()
    tbl = widgets.Output()
    btn = widgets.Button(description="Save table as CSV", button_style="info", icon="download")
    msg = widgets.HTML()
    state = {}

    def refresh():
        v = {k: c.value for k, c in ctrls.items()}
        b = replace(B, pH=v["pH"], alk=v["alk"], ca=v["ca"], pH_target=v["pH_t"], alk_target=v["alk_t"],
                    Q_avg=v["Qavg"], Q_max=v["Qmax"], Q_min=v["Qmin"],
                    pH_range=(v["pH"] - 0.2, v["pH"] + 0.2),
                    alk_range=(max(0.02, v["alk"] - 0.05), v["alk"] + 0.05))
        try:
            D = de.main_design(b, n=int(v["n"]), ebct=v["ebct"], v=v["v"], price_co2=v["pco2"],
                               price_marble=v["pm"])
        except ValueError as e:
            readout.value = f"<b style='color:{sv.RED}'>Cannot calculate:</b> {e}"
            return
        rows = de.register(b, D)
        state["rows"] = rows
        w2, w12 = D["finals"][b.T_min], D["finals"][b.T_max]
        f = D["filter"]
        (p0, p1), (c0, c1) = cb.TARGETS["pH"], cb.TARGETS["Ca"]
        warn = []
        if not (b.Q_min <= b.Q_avg <= b.Q_max):
            warn.append("Q<sub>min</sub> ≤ Q<sub>avg</sub> ≤ Q<sub>max</sub> is not fulfilled.")
        if w2.SI >= 0:
            warn.append("SI ≥ 0 at the target point: the marble filter cannot reach it.")
        readout.value = sv.cards([("CO₂ dose", f"{D['doses']['co2_mg']:.1f} mg/L", None),
                                  ("marble dissolved", f"{D['doses']['marble_mg']:.0f} mg/L", None)]) + sv.cards([
            ("pH 2 / 12 °C", f"{w2.pH:.2f} / {w12.pH:.2f}", _ok(p0 <= w12.pH and w2.pH <= p1)),
            ("Ca", f"{w2.ca_mg:.1f} mg/L", _ok(c0 <= w2.ca_mg <= c1)),
            ("SI", f"{sv.nz(w2.SI, 2):+.2f}", _ok(w2.SI < 0))]) + sv.cards([
            ("filters", f"{f['n']} × {f['area_each']:.1f} m²", _ok(f["area_each"] < 100)),
            ("marble depth", f"{f['depth']:.1f} m", _ok(f["depth"] >= 2.5)),
            ("cost (ESTIMATE)", f"{D['chem']['cost_per_m3']:.2f} NOK/m³", None)]) + "".join(
            f"<div><small style='color:{sv.RED}'>⚠ {w}</small></div>" for w in warn)
        tbl.clear_output(wait=True)
        with tbl:
            df = pd.DataFrame(rows)[["group", "parameter", "value", "unit", "source"]]
            display(HTML(df.to_html(index=False, escape=False)))
        msg.value = ""

    def save(_):
        name = "design_summary.csv"
        try:
            pd.DataFrame(state["rows"]).to_csv(name, index=False, encoding="utf-8-sig", sep=";")
            msg.value = (f"<small>Saved <code>{Path(name).resolve()}</code> (semicolon-separated, opens in Excel). "
                         "Saving again overwrites it – rename the file to keep a version.</small>")
        except OSError as e:
            msg.value = f"<small style='color:{sv.RED}'>Could not save: {e}</small>"
    btn.on_click(save)

    presets = {"Example waterworks (B1–B6)": dict(pH=B.pH, alk=B.alk, ca=B.ca, pH_t=B.pH_target, alk_t=B.alk_target,
                                                  Qavg=B.Q_avg, Qmax=B.Q_max, Qmin=B.Q_min, n=4, ebct=20, v=8,
                                                  pco2=3.0, pm=2.0),
               "Small plant (10 L/s)": dict(Qavg=6, Qmax=10, Qmin=2, n=2),
               "Large plant (200 L/s)": dict(Qavg=120, Qmax=200, Qmin=40, n=6)}
    pbox = sv.wire(ctrls, refresh, presets)
    refresh()
    sv.panel("Design calculator: CO₂ + marble filter",
             "Change any input; all results and the summary table below are recalculated. The table states the "
             "source of every value. Save it as CSV for your own notes.",
             list(ctrls.values()), widgets.VBox([readout]), widgets.HTML(""), pbox,
             below=widgets.VBox([widgets.HBox([btn, msg]), tbl]),
             symbols="Q<sub>avg</sub>, Q<sub>max</sub>, Q<sub>min</sub> = average, maximum (max hour, max day) and "
                     "minimum flow · EBCT = empty bed contact time · 'one out' = one filter out of service · "
                     "raw-water range for the dosing equipment: ±0.2 pH and ±0.05 mmol/L alkalinity around the "
                     "values chosen (assumption).")
