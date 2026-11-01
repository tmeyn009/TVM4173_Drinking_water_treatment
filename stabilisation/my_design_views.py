"""
my_design_views.py - display functions for Part C (your own design).

Each show_*() function reads one input cell, calculates with my_design.py,
shows results and 'points to check', and stores the result for the next
sections. Run the cells from top to bottom.
"""
from dataclasses import replace
from pathlib import Path
import html as _html
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML
import carbonate as cb
import stab_views as sv
import my_design as md

STATE = {}
ICON = {"stop": "⛔", "check": "⚠", "info": "ℹ"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _box(title, body, color):
    return (f"<div style='border:1px solid {sv.BORDER};border-left:5px solid {color};padding:6px 12px;"
            f"margin:8px 0'><b>{title}</b>{body}</div>")


def show_notes(notes, what="this section"):
    if not notes:
        display(HTML(_box("Points to check",
                          f"<div style='opacity:0.85'>Nothing unusual found in {what}. This does not mean the design "
                          "is right – only that the checks found nothing that differs from NV 212, NV 274 or the "
                          "chemistry.</div>", sv.GREEN)))
        return
    items = "".join(f"<li style='margin:3px 0'>{ICON[l]} {t}</li>" for l, t in notes)
    color = sv.RED if any(l == "stop" for l, _ in notes) else "#fb8c00"
    display(HTML(_box(f"Points to check ({len(notes)})", f"<ul style='margin:4px 0 0 0'>{items}</ul>", color)))


def _error(e):
    display(HTML(_box("⛔ Cannot calculate", f"<div>{e}</div>", sv.RED)))


def _need(*keys):
    missing = [k for k in keys if k not in STATE]
    if missing:
        names = {"B": "C1 (design basis)", "pre": "C2 (pre-treatment)", "chem": "C4 (chemistry)",
                 "units": "C5 (units)"}
        raise md.InputError("No valid result yet from " + ", ".join(names[k] for k in missing) + ". Run that section "
                            "(and fix it if it shows an error), then run this cell again.")


def _kind_badge(k):
    col = {"source": "rgba(67,160,71,0.25)", "assumption": "rgba(255,167,38,0.30)",
           "example": "rgba(229,57,53,0.25)", "missing": "rgba(229,57,53,0.35)",
           "calculated": "rgba(128,128,128,0.15)", "decision": "rgba(30,136,229,0.20)",
           "estimate": "rgba(229,57,53,0.22)"}.get(k, "transparent")
    txt = {"missing": "⚠ source missing", "example": "⚠ example value"}.get(k, k)
    return f"<span style='background:{col};padding:0 4px;border-radius:3px'>{txt}</span>"


def _html_table(df, legend=""):
    leg = f"<div style='font-size:0.85em;opacity:0.85'>{legend}</div>" if legend else ""
    return df.to_html(index=False, escape=False) + leg


def _deffeyes(ws, title, T, ca_list):
    """Deffeyes diagram with a treatment path (Plotly or matplotlib)."""
    ct_max, alk_max = sv.axis_limits(ws)
    alk_min = min(0.0, min(w.alk_mmol for w in ws) - 0.05)
    I = ws[-1].I
    if sv.USE_PLOTLY:
        d = sv.DeffeyesPlotly(n_sat=min(len(ca_list), 4), height=500, width=640)
        d.update(T, I, ct_max, alk_max, ca_sat=ca_list, steps=ws, title=title)
        d.fig.layout.yaxis.range = [alk_min, alk_max]
        display(d.fig)
    else:
        out = widgets.Output()
        sv._mpl_into(out, lambda plt: sv._deffeyes_body_mpl(T, I, ct_max, alk_max, ca_list, ws, alk_min=alk_min))
        display(out)


def _waters_table(ws):
    display(sv.table(ws))


# ---------------------------------------------------------------------------
# C1
# ---------------------------------------------------------------------------
def show_basis(inp):
    STATE.pop("B", None)
    try:
        B = md.basis(inp)
    except md.InputError as e:
        _error(e)
        return
    STATE["B"] = B
    rows = [dict(parameter=e["label"], value=e["value"], unit=e["unit"], source=_html.escape(str(e["source"])),
                 type=_kind_badge(e["kind"])) for e in B["E"].values()]
    display(HTML(_html_table(pd.DataFrame(rows))))
    v = B["v"]
    w = B["raw"]
    out = []
    for T in (v["T_min"], v["T_max"]):
        x = replace(w, temp_C=T)
        out.append([f"{T:g}", f"{x.pH:.2f}", f"{x.ct_mmol:.3f}", f"{x.co2_mg:.1f}", f"{x.SI:+.2f}", f"{x.CCPP():+.1f}"])
    display(HTML("<b>Raw water (design values) at the lowest and highest temperature</b>" + _html_table(
        pd.DataFrame(out, columns=["T (°C)", "pH", "C<sub>T</sub> (mmol/L)", "free CO<sub>2</sub> (mg/L)", "SI",
                                   "CCPP (mg CaCO<sub>3</sub>/L)"]),
        f"Flows in m³/h: Q<sub>max</sub> {v['Q_max']:.0f} · Q<sub>avg</sub> {v['Q_avg']:.0f} · Q<sub>min</sub> {v['Q_min']:.0f}. "
        f"Ionic strength from the conductivity: {w.I * 1000:.2f} mmol/L. C<sub>T</sub> is calculated from pH (at "
        f"{v['T_lab']:g} °C) and alkalinity and does not change with temperature; the pH does.")))
    show_notes(B["notes"], "the design basis")


# ---------------------------------------------------------------------------
# C2
# ---------------------------------------------------------------------------
def show_pretreatment(steps):
    STATE.pop("pre", None)
    try:
        _need("B")
        B = STATE["B"]
        T = B["v"]["T_min"]
        ws = md.pretreat(replace(B["raw"], temp_C=T), steps)
        notes = md.pretreat_notes(B, steps, ws)
    except md.InputError as e:
        _error(e)
        return
    STATE["pre"] = dict(steps=steps, ws=ws)
    if len(ws) == 1:
        display(HTML("<i>No pre-treatment steps given: the raw water enters corrosion control directly.</i>"))
    else:
        rows = []
        for i, (a, b) in enumerate(zip(ws[:-1], ws[1:]), 1):
            rows.append([i, md.STEPS.get(steps[i - 1]["step"], steps[i - 1]["step"]),
                         f"{b.alk_mmol - a.alk_mmol:+.3f}", f"{b.ct_mmol - a.ct_mmol:+.3f}",
                         f"{b.ca_mg - a.ca_mg:+.1f}", f"{a.pH:.2f} → {b.pH:.2f}"])
        display(HTML("<b>What each step changes</b>" + _html_table(pd.DataFrame(rows, columns=[
            "#", "step", "Δ Alk (mmol/L)", "Δ C<sub>T</sub> (mmol/L)", "Δ Ca (mg/L)", "pH"]),
            f"Shown at the lowest temperature ({T:g} °C). Alk = alkalinity, C<sub>T</sub> = total inorganic carbon.")))
        _deffeyes(ws, f"Raw water → water entering corrosion control, {T:g} °C", T,
                  sorted({round(ws[0].ca_mg, 1), round(ws[-1].ca_mg, 1)}))
        _waters_table(ws)
    show_notes(notes, "the pre-treatment")


# ---------------------------------------------------------------------------
# C3
# ---------------------------------------------------------------------------
def show_options():
    """Where single steps could take the water entering corrosion control (information, not a recommendation)."""
    try:
        _need("B", "pre")
    except md.InputError as e:
        _error(e)
        return
    B = STATE["B"]
    w = STATE["pre"]["ws"][-1]
    T = B["v"]["T_min"]
    arrows, rows = [], []

    def add(label, w2, col, note):
        arrows.append((w, w2, col, label))
        rows.append([label, f"{w2.pH:.2f}", f"{w2.alk_mmol:.2f}", f"{w2.ca_mg:.1f}", f"{w2.SI:+.2f}", note])
    try:
        add("marble only", w.apply(cb.marble_to_equilibrium()), sv.COL["marble"],
            "marble filter without CO₂ dosing, to equilibrium (upper limit)")
        if w.co2_mg > 13:
            s = w.apply(cb.strip_co2_to(w, 13.0))
            add("strip to 13 + marble", s.apply(cb.marble_to_equilibrium()), "#00897b",
                "NV 212 §12.2.1 rule for CO₂-rich water")
        for co2 in (10, 20):
            add(f"{co2} mg/L CO₂ + marble", w.apply(cb.co2_dose(co2)).apply(cb.marble_to_equilibrium()),
                sv.COL["CO2"] if co2 == 20 else "#ef9a9a", "marble filter to equilibrium (upper limit)")
        nao = 0.0
        for x in np.arange(0, 60, 0.2):
            if w.apply(cb.naoh(x)).pH >= 8.3:
                nao = x
                break
        if nao:
            add("NaOH to pH 8.3", w.apply(cb.naoh(nao)), sv.COL["base"], f"{nao:.1f} mg NaOH/L – no calcium added")
    except ValueError as e:
        _error(e)
        return
    ct_max, alk_max = sv.axis_limits([w] + [a[1] for a in arrows])
    if sv.USE_PLOTLY:
        d = sv.DeffeyesPlotly(n_sat=1, height=500, width=640)
        d.update(T, w.I, ct_max, alk_max, ca_sat=[w.ca_mg], arrows=arrows,
                 title=f"What single steps could do, {T:g} °C")
        display(d.fig)
    else:
        out = widgets.Output()
        sv._mpl_into(out, lambda plt: sv._deffeyes_body_mpl(T, w.I, ct_max, alk_max, [w.ca_mg], None, arrows))
        display(out)
    display(HTML(_html_table(pd.DataFrame(rows, columns=["step", "pH", "Alk (mmol/L)", "Ca (mg/L)", "SI", "note"]),
                             "These are illustrations from the water entering corrosion control, not designs. "
                             "Marble results are the equilibrium limit; a real filter stops a little short. "
                             "Lime + CO₂ is not shown: its result depends on both doses, which you set in C4.")))


# ---------------------------------------------------------------------------
# C4
# ---------------------------------------------------------------------------
def show_chemistry(meth):
    STATE.pop("chem", None)
    try:
        _need("B", "pre")
        B, P = STATE["B"], STATE["pre"]
        Td = md.entry(meth, "T_design", "temperature at which the design point is met")["base"]
        ws_pre = md.pretreat(replace(B["raw"], temp_C=Td), P["steps"])
        r = md.chemistry(B, ws_pre, P["steps"], meth)
        r["range"] = md.dose_range(B, P["steps"], meth, r)
    except md.InputError as e:
        _error(e)
        return
    STATE["chem"] = r
    STATE["meth"] = meth
    log = pd.DataFrame(r["log"], columns=["quantity", "calculation", "result"])
    display(HTML("<b>Calculation steps</b>" + _html_table(log)))
    full = ws_pre + r["waters"][1:]
    _deffeyes(full, f"{r['method']}, {Td:g} °C", Td, sorted({round(full[0].ca_mg, 1), round(r['final'].ca_mg, 1)}))
    _waters_table(full)
    if r.get("equilibrium") is not None:
        eq = r["equilibrium"]
        display(HTML(f"<div><small>Equilibrium limit after this CO₂ dose: pH {eq.pH:.2f}, Alk {eq.alk_mmol:.2f} mmol/L, "
                     f"Ca {eq.ca_mg:.1f} mg/L. Your design point needs {100 * r['share']:.0f} % of the marble that could "
                     "dissolve. The outlet pH is very sensitive to the last few mg/L (B3).</small></div>"))
    if r.get("vector") is not None:
        vv = r["vector"]
        txt = (f"strip {-vv['co2_mg']:.1f} mg/L" if vv["co2_mg"] < 0 else f"dose {vv['co2_mg']:.1f} mg/L CO₂")
        display(HTML(f"<div><small>For comparison – the vector design to your design point would {txt} and dissolve "
                     f"{vv['marble_mg']:.0f} mg CaCO₃/L (leaving {vv['waters'][1].co2_mg:.1f} mg/L CO₂ before the "
                     "filter; see B7).</small></div>"))
    rows = [[f"{T:g}", f"{w.pH:.2f}", f"{w.SI:+.2f}", f"{w.CCPP():+.1f}",
             {20.0: "house, cold water", 60.0: "hot-water system"}.get(T, "")] for T, w in r["temps"].items()]
    display(HTML("<b>Treated water at other temperatures</b>" + _html_table(pd.DataFrame(
        rows, columns=["T (°C)", "pH", "SI", "CCPP (mg CaCO<sub>3</sub>/L)", "note"]))))
    rr = []
    for x in r["range"]:
        if "error" in x:
            rr.append([x["pH"], f"{x['alk']:.2f}", f"{x['ca']:.1f}", "–", "–", "–", x["error"][:80]])
        else:
            dose = "; ".join(f"{k} {x[k]:.1f}" for k in ("co2", "marble", "lime") if x.get(k) is not None)
            rr.append([x["pH"], f"{x['alk']:.2f}", f"{x['ca']:.1f}", dose, f"{x['pH_out']:.2f}", f"{x['ca_out']:.1f}", ""])
    display(HTML("<b>Your raw-water range (same design point and pre-treatment)</b>" + _html_table(
        pd.DataFrame(rr, columns=["raw pH", "raw Alk", "raw Ca", "doses (mg/L)", "pH out", "Ca out", "note"]),
        "Use the highest dose for the capacity of the dosing equipment (C5) and discuss the range in your "
        "sensitivity check.")))
    show_notes(r["notes"], "the chemistry")


def explore_panel():
    """Sliders to try design points before writing them into the method cell."""
    try:
        _need("B", "pre", "chem")
    except md.InputError as e:
        _error(e)
        return
    meth0 = STATE["meth"]
    r0 = STATE["chem"]
    B, P = STATE["B"], STATE["pre"]
    v = B["v"]
    ctrls = {"pH_t": sv.pslider(float(np.clip(r0["E"]["pH_t"]["base"], 7.5, 9.0)), 7.5, 9.0, 0.05, "Design pH", ".2f"),
             "alk_t": sv.pslider(float(np.clip(r0["E"]["alk_t"]["base"], 0.3, 2.5)), 0.3, 2.5, 0.05,
                                 "Design Alk (mmol/L)", ".2f"),
             "Td": sv.pslider(float(np.clip(r0["Td"], 0, 25)), 0, 25, 1, "Met at (°C)", ".0f")}
    if r0["method"] == "CO2 stripping + marble filter":
        ctrls["strip"] = sv.pslider(float(np.clip(r0["E"]["strip"]["base"], 2, 40)), 2, 40, 0.5,
                                    "CO₂ after stripping", ".1f")
    readout = widgets.HTML()
    if sv.USE_PLOTLY:
        d = sv.DeffeyesPlotly(n_sat=2, height=480, width=600)
        figure = d.fig
    else:
        figure = widgets.Output()

    def refresh():
        x = {k: c.value for k, c in ctrls.items()}
        m = dict(meth0)
        m["pH_target"] = (x["pH_t"], "-", "exploring")
        m["alk_target"] = (x["alk_t"], "mmol/L", "exploring")
        m["T_design"] = (x["Td"], "°C", "exploring")
        if "strip" in x:
            m["co2_after_stripping"] = (x["strip"], "mg/L", "exploring")
        try:
            ws_pre = md.pretreat(replace(B["raw"], temp_C=x["Td"]), P["steps"])
            r = md.chemistry(B, ws_pre, P["steps"], m)
        except (md.InputError, ValueError) as e:
            readout.value = f"<b style='color:{sv.RED}'>Cannot calculate:</b> {e}"
            return
        full = ws_pre + r["waters"][1:]
        ct_max, alk_max = sv.axis_limits(full)
        alk_min = min(0.0, min(w.alk_mmol for w in full) - 0.05)
        cas = sorted({round(full[0].ca_mg, 1), round(r["final"].ca_mg, 1)})
        if sv.USE_PLOTLY:
            d.update(x["Td"], r["final"].I, ct_max, alk_max, ca_sat=cas, steps=full, title=f"Trying: {r['method']}")
            d.fig.layout.yaxis.range = [alk_min, alk_max]
        else:
            sv._mpl_into(figure, lambda plt: sv._deffeyes_body_mpl(x["Td"], r["final"].I, ct_max, alk_max, cas, full,
                                                                  alk_min=alk_min))
        f = r["final"]
        tl, th = r["temps"][v["T_min"]], r["temps"][v["T_max"]]
        doses = [(lab, f"{r[k]:.1f} mg/L", None) for k, lab in (("co2", "CO₂ (− = removed)"), ("marble", "CaCO₃ dissolved"),
                                                               ("lime", "lime")) if r.get(k) is not None]
        ok = lambda c: "ok" if c else "bad"
        readout.value = sv.cards(doses) + sv.cards([
            (f"pH {v['T_min']:g} / {v['T_max']:g} °C", f"{tl.pH:.2f} / {th.pH:.2f}",
             ok(v["pH_t_lo"] <= min(tl.pH, th.pH) and max(tl.pH, th.pH) <= v["pH_t_hi"])),
            ("Alk", f"{f.alk_mmol:.2f} mmol/L", ok(v["alk_t_lo"] <= f.alk_mmol <= v["alk_t_hi"])),
            ("Ca", f"{f.ca_mg:.1f} mg/L", ok(v["ca_t_lo"] <= f.ca_mg <= v["ca_t_hi"])),
            ("SI", f"{f.SI:+.2f}", None)]) + (
            f"<div><small>{len(r['notes'])} point(s) to check with these values. When you have decided, write the "
            "values into the method cell above (with your reasoning as the source) and run it again – the sliders "
            "are not saved.</small></div>")

    pbox = sv.wire(ctrls, refresh)
    refresh()
    sv.panel("Try design points", "Move the sliders; nothing is saved until you copy the values into the method cell.",
             list(ctrls.values()), figure, readout, pbox,
             symbols="green/red: inside/outside your target ranges from C1 · SI = saturation index of the treated water.")


# ---------------------------------------------------------------------------
# C5
# ---------------------------------------------------------------------------
def show_units(U):
    STATE.pop("units", None)
    try:
        _need("B", "chem")
        out = md.units(STATE["B"], STATE["chem"], U)
    except md.InputError as e:
        _error(e)
        return
    STATE["units"] = out
    rows = [dict(parameter=e["label"], value=e["value"], unit=e["unit"], source=_html.escape(str(e["source"])),
                 type=_kind_badge(e["kind"])) for e in out["entries"]]
    if rows:
        display(HTML("<b>Your design values</b>" + _html_table(pd.DataFrame(rows))))
    display(HTML("<b>Results</b>" + _html_table(pd.DataFrame(out["log"], columns=["quantity", "calculation", "result"]))))
    mf = out.get("marble_filter")
    if mf:
        txt = (f"Marble filters: {mf['n']} × {mf['area_each']:.1f} m² (total {mf['area_total']:.0f} m²), marble depth "
               f"{mf['depth']:.1f} m + {mf['reserve']:.1f} m reserve. Filter rate {mf['v_all']:.1f} m/h with all filters in")
        if mf["n"] >= 2:
            txt += f", {mf['v_one_out']:.1f} m/h with one out; EBCT {mf['ebct_all']:.0f} / {mf['ebct_one_out']:.0f} min"
        txt += (f". At Q<sub>min</sub> the EBCT is {mf['ebct_min_flow']:.0f} min. Backwash water ≈ {mf['bw_share']:.1f} % "
                "of the average production (gross vs. net production!).")
        display(HTML(f"<div><small>{txt}</small></div>"))
    show_notes(out["notes"], "the unit design")


# ---------------------------------------------------------------------------
# C6
# ---------------------------------------------------------------------------
def show_costs(prices):
    STATE.pop("costs", None)
    try:
        _need("B", "chem", "units")
        C = md.costs(STATE["B"], STATE["units"], STATE["chem"], prices)
    except md.InputError as e:
        _error(e)
        return
    STATE["costs"] = C
    if not C["rows"]:
        display(HTML("<i>No costs calculated (no prices given).</i>"))
    else:
        df = pd.DataFrame([[r["item"], f"{r['kg_d']:.0f}", f"{r['price']:g}", f"{r['kNOK']:.0f}",
                            _html.escape(str(r["source"]))] for r in C["rows"]],
                          columns=["chemical", "kg/d", "NOK/kg", "kNOK/year", "price source"])
        display(HTML("<b>Chemical costs – ESTIMATE</b>" + _html_table(
            df, f"Total ≈ {C['total']:.0f} kNOK/year ≈ {C['per_m3']:.2f} NOK/m³ (at Q<sub>avg</sub>). Not included: "
                "construction, equipment, energy, labour, maintenance, analyses. Round and give a range in the report.")))
    show_notes(C["notes"], "the costs")


# ---------------------------------------------------------------------------
# C7
# ---------------------------------------------------------------------------
def show_summary():
    try:
        _need("B", "pre", "chem", "units")
    except md.InputError as e:
        _error(e)
        return
    B, P, r, U = STATE["B"], STATE["pre"], STATE["chem"], STATE["units"]
    C = STATE.get("costs", {"rows": [], "notes": []})
    rows = md.summary_rows(B, P["steps"], r, U, C)
    df = pd.DataFrame(rows)
    shown = df.copy()
    shown["type"] = shown["type"].map(_kind_badge)
    shown["source"] = shown["source"].map(lambda s: s if "<sub>" in str(s) else _html.escape(str(s)))
    n_missing = int((df["type"] == "missing").sum())
    n_example = int((df["type"] == "example").sum())
    allnotes = (list(B["notes"]) + list(md.pretreat_notes(B, P["steps"], P["ws"])) + list(r["notes"])
                + list(U["notes"]) + list(C.get("notes", [])))
    head = (f"<b>{len(df)} values</b> · {n_missing} without source · {n_example} still example values · "
            f"{len([n for n in allnotes if n[0] != 'info'])} points to check in total (see the sections above)")
    display(HTML(_box("Status", f"<div>{head}</div>", sv.RED if (n_missing or n_example) else sv.GREEN)))
    display(HTML(_html_table(shown)))
    ass = [e for e in list(B["E"].values()) + [x for x in r["E"].values() if x] + U["entries"] if e["kind"] == "assumption"]
    if ass:
        reg = pd.DataFrame([[e["label"], f"{e['value']} {e['unit']}", _html.escape(str(e["source"])),
                             _html.escape(str(e["consequence"])) or "⚠ not stated"] for e in ass],
                           columns=["assumption", "value", "justification (source text)", "consequence if wrong"])
        display(HTML("<b>Assumption register</b> (all inputs whose source starts with 'assumption')" + _html_table(reg)))
    STATE["summary"] = df
    btn_csv = widgets.Button(description="Save table as CSV", button_style="info")
    btn_html = widgets.Button(description="Save table as HTML (for Word)", button_style="info")
    btn_png = widgets.Button(description="Save Deffeyes figure (PNG)", button_style="info")
    msg = widgets.HTML()
    calc_rows = r["log"] + U["log"]

    def save_csv(_):
        try:
            df.to_csv("my_design_summary.csv", index=False, sep=";", encoding="utf-8-sig")
            msg.value = f"<small>Saved <code>{Path('my_design_summary.csv').resolve()}</code> (opens in Excel).</small>"
        except OSError as e:
            msg.value = f"<small style='color:{sv.RED}'>Could not save: {e}</small>"

    def save_html(_):
        try:
            calc = pd.DataFrame(calc_rows, columns=["quantity", "calculation", "result"])
            page = ("<html><head><meta charset='utf-8'><title>Corrosion control – design summary</title></head><body>"
                    "<h2>Corrosion-control design – summary</h2>" + df.to_html(index=False, escape=False)
                    + "<h2>Calculation steps</h2>" + calc.to_html(index=False, escape=False) + "</body></html>")
            Path("my_design_summary.html").write_text(page, encoding="utf-8")
            msg.value = (f"<small>Saved <code>{Path('my_design_summary.html').resolve()}</code>. Download it, open it "
                         "in a browser, then copy the tables into Word.</small>")
        except OSError as e:
            msg.value = f"<small style='color:{sv.RED}'>Could not save: {e}</small>"

    def save_png(_):
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            ws_pre = md.pretreat(replace(B["raw"], temp_C=r["Td"]), P["steps"])
            full = ws_pre + r["waters"][1:]
            ct_max, alk_max = sv.axis_limits(full)
            with matplotlib.rc_context({"figure.facecolor": "white"}), plt.style.context("default"):
                fig, ax = plt.subplots(figsize=(7, 5.5))
                cb.deffeyes(ax, temp_C=r["Td"], I=full[-1].I, ct_max=ct_max, alk_max=alk_max,
                            ca_sat=sorted({round(full[0].ca_mg, 1), round(r["final"].ca_mg, 1)}))
                alk_min = min(0.0, min(x.alk_mmol for x in full) - 0.05)
                if alk_min < 0:
                    ax.set_ylim(alk_min, alk_max)
                    ax.axhline(0, color="0.5", lw=0.8)
                cb.plot_path(ax, full)
                fig.tight_layout()
                fig.savefig("my_design_deffeyes.png", dpi=200)
                plt.close(fig)
            msg.value = (f"<small>Saved <code>{Path('my_design_deffeyes.png').resolve()}</code> (white background, "
                         "200 dpi). Number the steps in your figure caption.</small>")
        except Exception as e:     # noqa: BLE001 - show any plotting problem to the student
            msg.value = f"<small style='color:{sv.RED}'>Could not save the figure: {e}</small>"
    btn_csv.on_click(save_csv)
    btn_html.on_click(save_html)
    btn_png.on_click(save_png)
    display(widgets.VBox([widgets.HBox([btn_csv, btn_html, btn_png], layout=widgets.Layout(flex_flow="row wrap")), msg]))
