"""
my_design.py - calculations and checks for Part C (your own corrosion-control design).

The notebook C_My_corrosion_control_design.ipynb keeps all inputs in code
cells as  key: (value, unit, source)  or  key: (value, unit, source, consequence).
This module converts units, applies the pre-treatment, calculates the doses for
the chosen method, sizes the units, and returns 'points to check'.

It never chooses anything: method, target point and design values are the
students' decisions. The checks only compare with NV 212/2015, NV 274/2022
and the chemistry, and are worded as suggestions.
"""
from dataclasses import replace
import numpy as np
import carbonate as cb
import design_example as de

M = cb.M
M_MN = 54.938

# ---------------------------------------------------------------------------
# units and entries
# ---------------------------------------------------------------------------
FLOW = {"L/s": 3.6, "l/s": 3.6, "m3/h": 1.0, "m³/h": 1.0, "m3/d": 1 / 24, "m³/d": 1 / 24}
EC = {"µS/cm": 1.0, "uS/cm": 1.0, "mS/m": 10.0}
ALK = {"mmol/L": 1.0, "mmol/l": 1.0, "meq/L": 1.0, "meq/l": 1.0,
       "mg CaCO3/L": 1 / 50.04, "mg HCO3/L": 1 / 61.02}
CA = {"mg/L": 1.0, "mg/l": 1.0, "mmol/L": M["Ca"], "mmol/l": M["Ca"]}


class InputError(Exception):
    pass


def kind(source):
    """Classify a source text."""
    s = (source or "").strip()
    if not s:
        return "missing"
    low = s.lower()
    if low.startswith("example"):
        return "example"
    if low.startswith("assumption"):
        return "assumption"
    return "source"


def entry(d, key, label, units=None, required=True):
    """Read d[key] = (value, unit, source[, consequence]); returns dict."""
    if key not in d:
        if required:
            raise InputError(f"'{key}' ({label}) is missing in the input cell.")
        return None
    t = d[key]
    if not isinstance(t, (tuple, list)) or len(t) < 3:
        raise InputError(f"'{key}' must be written as (value, unit, source), e.g. "
                         f"\"{key}\": (1.0, \"{(list(units)[0] if units else '-')}\", \"client brief, Table 1\").")
    value, unit, source = t[0], t[1], t[2]
    cons = t[3] if len(t) > 3 else ""
    if value is None:
        if required:
            raise InputError(f"'{key}' ({label}) has no value. If the brief gives none, make an assumption, "
                             "write it in, and mark the source as 'assumption: …'.")
        return None
    if isinstance(units, dict):
        if unit not in units:
            raise InputError(f"Unit '{unit}' for '{key}' is not recognised. Allowed: {', '.join(units)}.")
        factor = units[unit]
    else:
        factor = 1.0
    try:
        base = float(value) * factor
    except (TypeError, ValueError):
        raise InputError(f"'{key}': the value {value!r} is not a number.")
    return dict(key=key, label=label, value=value, unit=unit, source=source, consequence=cons,
                kind=kind(source), base=base)


# ---------------------------------------------------------------------------
# results container and checks
# ---------------------------------------------------------------------------
class Notes(list):
    """Points to check: (level, text) with level 'stop', 'check' or 'info'."""
    def stop(self, t): self.append(("stop", t))
    def check(self, t): self.append(("check", t))
    def info(self, t): self.append(("info", t))


def source_notes(entries, notes):
    miss = [e["label"] for e in entries if e is not None and e["kind"] == "missing"]
    ex = [e["label"] for e in entries if e is not None and e["kind"] == "example"]
    if miss:
        notes.check("⚠ No source given for: <b>" + "</b>, <b>".join(miss) + "</b>. Design values need a verifiable "
                    "source (client brief, measurement, guideline with table/section) – or mark them as 'assumption: …'.")
    if ex:
        notes.check("Still the example values from Part B – replace them with your own values and sources: <b>"
                    + "</b>, <b>".join(ex) + "</b>.")


# ---------------------------------------------------------------------------
# C1 design basis
# ---------------------------------------------------------------------------
BASIS_FIELDS = [
    ("Q_max", "maximum flow (design flow)", FLOW), ("Q_avg", "average flow", FLOW), ("Q_min", "minimum flow", FLOW),
    ("T_min", "lowest water temperature", None), ("T_max", "highest water temperature", None),
    ("T_lab", "temperature at which the pH was measured", None),
    ("pH", "pH (design value)", None), ("pH_min", "pH, lowest", None), ("pH_max", "pH, highest", None),
    ("alk", "alkalinity (design value)", ALK), ("alk_min", "alkalinity, lowest", ALK), ("alk_max", "alkalinity, highest", ALK),
    ("ca", "calcium (design value)", CA), ("ca_min", "calcium, lowest", CA), ("ca_max", "calcium, highest", CA),
    ("ec", "conductivity", EC),
    ("pH_t_lo", "target pH, lower limit", None), ("pH_t_hi", "target pH, upper limit", None),
    ("alk_t_lo", "target alkalinity, lower limit", ALK), ("alk_t_hi", "target alkalinity, upper limit", ALK),
    ("ca_t_lo", "target calcium, lower limit", CA), ("ca_t_hi", "target calcium, upper limit", CA),
]


def basis(inp):
    """Read and check the design basis. Returns dict with entries, values, waters, notes."""
    notes = Notes()
    E = {k: entry(inp, k, lab, u) for k, lab, u in BASIS_FIELDS}
    v = {k: e["base"] for k, e in E.items()}
    source_notes(E.values(), notes)
    if not v["Q_min"] <= v["Q_avg"] <= v["Q_max"]:
        notes.check("The flows do not appear to fulfil Q<sub>min</sub> ≤ Q<sub>avg</sub> ≤ Q<sub>max</sub> – please check "
                    "values and units.")
    for p in ("pH", "alk", "ca"):
        if not v[p + "_min"] <= v[p] <= v[p + "_max"]:
            notes.check(f"The design value of <b>{E[p]['label'].split(' (')[0]}</b> appears to lie outside its own "
                        "range (lowest–highest) – please check.")
    if v["T_min"] > v["T_max"]:
        notes.check("The lowest temperature appears to be higher than the highest temperature.")
    if v["T_max"] > 25 or v["T_min"] < 0:
        notes.check("The temperature range appears unusual for a Norwegian water source – please check.")
    ec_raw = E["ec"]
    if ec_raw["unit"] in ("µS/cm", "uS/cm") and ec_raw["base"] < 10:
        notes.check(f"A conductivity of {ec_raw['value']} µS/cm appears very low – is the unit perhaps mS/m "
                    "(1 mS/m = 10 µS/cm)?")
    if ec_raw["unit"] == "mS/m" and ec_raw["base"] > 2000:
        notes.check(f"A conductivity of {ec_raw['value']} mS/m appears very high – is the unit perhaps µS/cm?")
    if min(v["pH_min"], v["pH"]) < 4.6 and v["alk"] > 0.02:
        notes.check("A pH below about 4.5 together with a positive alkalinity is not consistent "
                    "(alkalinity is titrated to pH 4.5) – please check pH and alkalinity.")
    if v["ca"] < 0.1:
        notes.stop("Calcium must be larger than 0. If the brief gives no calcium, make an assumption "
                   "(e.g. from typical values for the source type) and state it.")
    if v["pH_t_lo"] < 8.0 or v["pH_t_hi"] > 8.5 or v["alk_t_lo"] < 0.6 or v["alk_t_hi"] > 1.0 \
            or v["ca_t_lo"] < 15 or v["ca_t_hi"] > 25:
        notes.check("Your target ranges differ from NV 274/2022 (pH 8.0–8.5, alkalinity 0.6–1.0 mmol/L, Ca 15–25 mg/L). "
                    "If this is intended (e.g. a copper-rich network), please justify it in the report.")
    try:
        w = cb.Water.from_pH_alk(v["pH"], v["alk"], max(v["ca"], 0.01), temp_C=v["T_lab"], ec_uS_cm=v["ec"],
                                 name="raw water")
    except ValueError as e:
        raise InputError(f"The raw water cannot be calculated from pH, alkalinity and conductivity: {e}")
    return dict(E=E, v=v, raw=w, notes=notes)


def raw_at(B, T, pH=None, alk=None, ca=None):
    v = B["v"]
    w = cb.Water.from_pH_alk(pH if pH is not None else v["pH"], alk if alk is not None else v["alk"],
                             ca if ca is not None else v["ca"], temp_C=v["T_lab"], ec_uS_cm=v["ec"],
                             name="raw water")
    return replace(w, temp_C=T)


# ---------------------------------------------------------------------------
# C2 pre-treatment (bookkeeping only; performance values come from the students)
# ---------------------------------------------------------------------------
STEPS = {
    "alkali": "base before coagulation / for pH (NaOH, Ca(OH)2 or Na2CO3)",
    "co2_dose": "CO2 dosing", "co2_strip_to": "CO2 stripping to a residual",
    "coagulant": "coagulation with Fe(III) or Al(III)",
    "fe_mn_oxidation": "oxidation and removal of Fe(II) and Mn(II)",
    "membrane": "NF or RO (with optional bypass)",
    "ion_exchange": "ion exchange (with optional bypass)",
}


def blend(a, b, f, label):
    """f = share of stream a."""
    return replace(a, Ca=f * a.Ca + (1 - f) * b.Ca, Alk=f * a.Alk + (1 - f) * b.Alk,
                   CT=f * a.CT + (1 - f) * b.CT, I=f * a.I + (1 - f) * b.I, label=label)


def _frac(s, k, default=None):
    x = s.get(k, default)
    if x is None:
        raise InputError(f"Step '{s.get('step')}': '{k}' is missing.")
    x = float(x)
    if not 0 <= x <= 1:
        raise InputError(f"Step '{s.get('step')}': '{k}' must be a fraction between 0 and 1 (you gave {x}).")
    return x


def apply_step(w, s):
    t = s.get("step")
    if t == "alkali":
        chem = s.get("chemical", "NaOH")
        f = {"NaOH": cb.naoh, "Ca(OH)2": cb.lime, "Na2CO3": cb.soda}.get(chem)
        if f is None:
            raise InputError("Step 'alkali': chemical must be 'NaOH', 'Ca(OH)2' or 'Na2CO3'.")
        return w.apply(f(float(s["dose"])))
    if t == "co2_dose":
        return w.apply(cb.co2_dose(float(s["dose"])))
    if t == "co2_strip_to":
        return w.apply(cb.strip_co2_to(w, float(s["residual"])))
    if t == "coagulant":
        metal = s.get("metal", "Fe")
        if metal not in ("Fe", "Al"):
            raise InputError("Step 'coagulant': metal must be 'Fe' or 'Al'.")
        return w.apply(cb.coagulant(float(s["dose"]), metal, float(s.get("basicity", 0.0))))
    if t == "fe_mn_oxidation":
        fe, mn = float(s.get("fe", 0)), float(s.get("mn", 0))
        # Fe2+ + 1/4 O2 + 5/2 H2O -> Fe(OH)3 + 2 H+ ;  Mn2+ + 1/2 O2 + H2O -> MnO2 + 2 H+
        d = 2 * (fe / M["Fe"] + mn / M_MN)
        return w.apply(cb.Step(f"Fe/Mn oxidation ({fe:g} mg Fe/L, {mn:g} mg Mn/L)", dAlk=-d))
    if t == "membrane":
        rca, ralk, rco2, share = (_frac(s, "rej_ca"), _frac(s, "rej_alk"), _frac(s, "rej_co2", 0.0),
                                  _frac(s, "share", 1.0))
        co2 = w.species[0]
        rest = w.CT - co2
        perm = replace(w, Ca=w.Ca * (1 - rca), Alk=w.Alk * (1 - ralk), CT=co2 * (1 - rco2) + rest * (1 - ralk),
                       I=w.I * (1 - (rca + ralk) / 2), label="membrane permeate")
        if share < 1:
            return blend(perm, w, share, f"membrane ({100 * share:.0f} %) + bypass")
        return replace(perm, label="membrane permeate")
    if t == "ion_exchange":
        rca, ralk, share = _frac(s, "rem_ca"), _frac(s, "rem_alk"), _frac(s, "share", 1.0)
        rest = w.CT - w.species[0]
        out = replace(w, Ca=w.Ca * (1 - rca), Alk=w.Alk * (1 - ralk), CT=w.species[0] + rest * (1 - ralk),
                      label="ion exchange outlet")
        return blend(out, w, share, f"ion exchange ({100 * share:.0f} %) + bypass") if share < 1 else out
    raise InputError(f"Unknown step '{t}'. Allowed steps: {', '.join(STEPS)}.")


def pretreat(w, steps):
    """Returns the list of waters [raw, after step 1, ...]."""
    ws = [w]
    for s in steps:
        try:
            ws.append(apply_step(ws[-1], s))
        except ValueError as e:
            raise InputError(f"Step '{s.get('step')}' cannot be calculated: {e}")
    return ws


def pretreat_notes(B, steps, ws):
    notes = Notes()
    for s, w in zip(steps, ws[1:]):
        src = s.get("source", "")
        if kind(src) in ("missing", "example"):
            notes.check(f"⚠ Step '{s['step']}': " + ("no source given for the values" if kind(src) == "missing"
                                                     else "still the example values – replace them") + ".")
        if s["step"] == "coagulant":
            if w.alk_mmol <= 0.02:
                notes.check(f"The coagulant dose appears to consume (almost) all the alkalinity (Alk after coagulation "
                            f"{w.alk_mmol:+.2f} mmol/L, pH {w.pH:.1f}). Please check the dose, or whether a base must "
                            "be added before coagulation.")
            if s.get("metal", "Fe") == "Al" and not 5.5 <= w.pH <= 7.0:
                notes.check(f"The pH after Al coagulation appears to be {w.pH:.1f}; NV 212 §12.3.1 gives a coagulation "
                            "pH of about 6.0–6.5 for aluminium – please check dose and pH adjustment.")
            if s.get("metal", "Fe") == "Fe" and w.pH > 6.0:
                notes.check(f"The pH after Fe coagulation appears to be {w.pH:.1f}. For humic water NV 212 §12.3.1 "
                            "describes iron coagulation at low pH (about 4.5 with marble filters) – please check.")
        if s["step"] == "membrane":
            notes.info(f"After the membrane step the water has {w.alk_mmol:.2f} mmol/L alkalinity, {w.ca_mg:.1f} mg/L Ca, "
                       f"but still {w.co2_mg:.1f} mg/L free CO₂ (pH {w.pH:.1f}, SI {w.SI:+.1f}) – it is more aggressive "
                       "than the feed water.")
        if s["step"] == "ion_exchange" and float(s.get("rem_alk", 0)) > 0:
            notes.info("Anion exchange removes bicarbonate mainly at the start of a run; later the removal drops. "
                       "Please check which situation is critical for your design.")
        if s["step"] == "co2_strip_to" and w.SI > 0.2:
            notes.check(f"After stripping the water appears to be supersaturated (SI {w.SI:+.1f}) – CaCO₃ may deposit "
                        "in the aerator (M08, slide 'Physical treatment: CO₂ stripping').")
    final = ws[-1]
    (p0, p1), (a0, a1) = (B["v"]["pH_t_lo"], B["v"]["pH_t_hi"]), (B["v"]["alk_t_lo"], B["v"]["alk_t_hi"])
    if final.SI > -0.3 and p0 <= final.pH <= p1 + 0.3 and final.alk_mmol >= a0:
        notes.info(f"The water entering corrosion control appears to be close to calcite saturation already "
                   f"(SI {final.SI:+.2f}, pH {final.pH:.2f}, Alk {final.alk_mmol:.2f} mmol/L). Please check whether "
                   "corrosion control is needed at all, or only because of a pre-treatment step.")
    return notes


# ---------------------------------------------------------------------------
# C4 chemistry: methods
# ---------------------------------------------------------------------------
METHODS = ["CO2 + marble filter", "CO2 stripping + marble filter", "lime + CO2",
           "coagulation + three-media filter (Molde)", "coagulation + separate marble filter"]
MARBLE = {"CO2 + marble filter", "CO2 stripping + marble filter",
          "coagulation + three-media filter (Molde)", "coagulation + separate marble filter"}


def chemistry(B, ws_pre, steps, meth):
    """meth: dict from the method cell. Returns dict with doses, waters, notes, log."""
    notes = Notes()
    name = meth.get("method")
    if name not in METHODS:
        raise InputError(f"'method' must be one of: {', '.join(METHODS)}.")
    E = {"pH_t": entry(meth, "pH_target", "design point: pH"),
         "alk_t": entry(meth, "alk_target", "design point: alkalinity", ALK),
         "T_d": entry(meth, "T_design", "temperature at which the design point is met"),
         "naoh": entry(meth, "naoh_trim", "NaOH after the corrosion-control step", required=False)}
    if name == "CO2 stripping + marble filter":
        E["strip"] = entry(meth, "co2_after_stripping", "free CO₂ after stripping")
    source_notes(E.values(), notes)
    pH_t, alk_t, Td = E["pH_t"]["base"], E["alk_t"]["base"], E["T_d"]["base"]
    naoh = E["naoh"]["base"] if E["naoh"] else 0.0
    v = B["v"]
    start = replace(ws_pre[-1], temp_C=Td)
    coag = [s for s in steps if s.get("step") == "coagulant"]
    if name.startswith("coagulation") and not coag:
        notes.check("You chose a coagulation method, but there is no coagulant step in C2 – the effect of the "
                    "coagulant on alkalinity and CO₂ is then missing.")
    r = dict(method=name, E=E, start=start, naoh=naoh, Td=Td)
    log = []
    if name == "CO2 stripping + marble filter":
        res = E["strip"]["base"]
        if res >= start.co2_mg:
            raise InputError(f"The CO₂ after stripping ({res:g} mg/L) must be lower than the CO₂ in the water entering "
                             f"corrosion control ({start.co2_mg:.1f} mg/L).")
        s1 = start.apply(cb.strip_co2_to(start, res))
        s1 = replace(s1, label=f"stripped to {res:g} mg/L CO₂")
        s2 = replace(s1.apply(cb.marble_to_equilibrium()), label="marble filter (equilibrium limit)")
        r.update(co2=-(start.ct_mmol - s1.ct_mmol) * M["CO2"], marble=(s2.alk_mmol - s1.alk_mmol) / 2 * M["CaCO3"],
                 waters=[start, s1, s2])
        log += [("CO₂ removed by stripping", f"C<sub>T</sub> {start.ct_mmol:.3f} → {s1.ct_mmol:.3f} mmol/L",
                 f"{start.co2_mg - res:.1f} mg/L"),
                ("marble dissolved (to equilibrium, upper limit)", "(Alk<sub>out</sub> − Alk<sub>in</sub>)/2 × 100.1",
                 f"{r['marble']:.1f} mg CaCO₃/L")]
        vec = de.doses(start, pH_t, alk_t, Td) if alk_t >= start.alk_mmol else None
        r["vector"] = vec
        if not 12 <= res <= 15:
            notes.check(f"NV 212 §12.2.1 recommends stripping to 12–15 mg/L before a marble filter; you chose {res:g} mg/L. "
                        "Less CO₂ means slower marble dissolution, more CO₂ means higher alkalinity – please justify.")
    elif name == "lime + CO2":
        l = (alk_t - start.alk_mmol) / 2
        if l < 0:
            raise InputError("The design-point alkalinity is below that of the water entering corrosion control: "
                             "lime cannot remove alkalinity.")
        after = start.apply(cb.lime(l * M["Ca(OH)2"]))
        ct_t = 1000 * cb.CT_from_pH_alk(pH_t, alk_t / 1000, after.K)
        d = ct_t - start.ct_mmol
        s1 = start.apply(cb.co2_dose(d * M["CO2"]) if d >= 0 else cb.co2_strip(-d * M["CO2"]))
        s2 = s1.apply(cb.lime(l * M["Ca(OH)2"]))
        r.update(co2=d * M["CO2"], lime=l * M["Ca(OH)2"], waters=[start, s1, s2])
        log += [("lime dose", "(Alk<sub>t</sub> − Alk<sub>in</sub>)/2 × 74.1",
                 f"({alk_t:.2f} − {start.alk_mmol:.3f})/2 × 74.1 = {r['lime']:.1f} mg Ca(OH)₂/L"),
                ("target C<sub>T</sub>", f"from pH {pH_t:g}, Alk {alk_t:g} at {Td:g} °C", f"{ct_t:.3f} mmol/L"),
                ("CO₂ dose", "(C<sub>T,t</sub> − C<sub>T,in</sub>) × 44.0",
                 f"({ct_t:.3f} − {start.ct_mmol:.3f}) × 44.0 = {r['co2']:.1f} mg/L")]
        if d < 0:
            notes.check("The CO₂ dose comes out negative: the water already contains more inorganic carbon than the "
                        "design point needs. Please check the design point, or consider stripping instead.")
        if not 20 <= r["lime"] <= 35 or not 20 <= r["co2"] <= 40:
            notes.check(f"NV 212 §12.2.3.4 gives typical doses for Norwegian water of 20–35 mg Ca(OH)₂/L and 20–40 mg CO₂/L; "
                        f"your doses are {r['lime']:.0f} and {r['co2']:.0f} mg/L. This can be right for your water – "
                        "please check and explain.")
    else:
        if alk_t < start.alk_mmol:
            raise InputError("The design-point alkalinity is below that of the water entering corrosion control: "
                             "marble cannot remove alkalinity.")
        d_ = de.doses(start, pH_t, alk_t, Td)
        s1, s2 = d_["waters"][1], d_["waters"][2]
        r.update(co2=d_["co2_mg"], marble=d_["marble_mg"], waters=[start, s1, s2])
        ct_t = s2.ct_mmol
        log += [("marble dissolved", "(Alk<sub>t</sub> − Alk<sub>in</sub>)/2 × 100.1",
                 f"({alk_t:.2f} − {start.alk_mmol:.3f})/2 × 100.1 = {r['marble']:.1f} mg CaCO₃/L"),
                ("target C<sub>T</sub>", f"from pH {pH_t:g}, Alk {alk_t:g} at {Td:g} °C", f"{ct_t:.3f} mmol/L"),
                ("CO₂ dose", "C<sub>T,t</sub> − C<sub>T,in</sub> − marble (all mmol/L) × 44.0",
                 f"{r['co2']:.1f} mg/L" + (" (negative = removal needed)" if r["co2"] < 0 else ""))]
        if r["co2"] < 0:
            notes.check(f"The CO₂ 'dose' comes out negative ({r['co2']:.1f} mg/L): the water already contains more CO₂ than "
                        "the design point needs, so marble would dissolve beyond it. Please check the design point, or "
                        "consider stripping CO₂ first.")
        if s2.SI > -0.05:
            notes.check(f"The design point appears to be at or above calcite saturation (SI {s2.SI:+.2f}). A marble filter "
                        "cannot reach it" + (" – please check the design point." if naoh > 0 else
                                             " – please check the design point or plan an NaOH trim after the filter."))
        eq = start.apply(cb.co2_dose(max(r["co2"], 0))).apply(cb.marble_to_equilibrium())
        r["equilibrium"] = eq
        r["share"] = (alk_t - start.alk_mmol) / max(eq.alk_mmol - start.alk_mmol, 1e-9)
        if name == "CO2 + marble filter" and not coag:
            lo_hi = table_12_1(B["raw"], v)
            if lo_hi and not lo_hi[0] - 0.5 <= r["co2"] <= lo_hi[1] + 0.5:
                notes.check(f"The CO₂ dose ({r['co2']:.1f} mg/L) lies outside the experience range of NV 212 Table 12.1 "
                            f"for this raw water ({lo_hi[0]:.0f}–{lo_hi[1]:.0f} mg/L). Please check the raw-water data "
                            "and the design point.")
        if coag and coag[0].get("metal", "Fe") == "Fe" and r["co2"] > 10:
            notes.check(f"With iron coagulation NV 212 §12.3.1 expects 0–10 mg CO₂/L (typically 4–5); your dose is "
                        f"{r['co2']:.1f} mg/L – please check.")
    # NaOH trim
    final = r["waters"][-1]
    if naoh > 0:
        final = replace(final.apply(cb.naoh(naoh)), label=f"NaOH trim {naoh:g} mg/L")
        r["waters"].append(final)
    r["final"] = final
    r["log"] = log
    # target checks
    (p0, p1), (a0, a1), (c0, c1) = ((v["pH_t_lo"], v["pH_t_hi"]), (v["alk_t_lo"], v["alk_t_hi"]),
                                    (v["ca_t_lo"], v["ca_t_hi"]))
    if not (p0 <= pH_t <= p1 and a0 <= alk_t <= a1):
        notes.check(f"The design point (pH {pH_t:g}, Alk {alk_t:g} mmol/L) lies outside your target ranges "
                    f"(pH {p0:.1f}–{p1:.1f}, Alk {a0:.1f}–{a1:.1f}). If this is intended, please justify it.")
    if name == "CO2 stripping + marble filter" and final.alk_mmol > a1:
        notes.check(f"After stripping to {E['strip']['base']:g} mg/L and marble filtration the alkalinity would be about "
                    f"{final.alk_mmol:.2f} mmol/L, above your target ({a0:.1f}–{a1:.1f} mmol/L). This is the usual trade-off of this "
                    "route (see B7) – please discuss it.")
    if not c0 <= final.ca_mg <= c1:
        notes.check(f"Calcium in the treated water appears to be {final.ca_mg:.1f} mg/L, outside your target "
                    f"({c0:g}–{c1:g} mg/L) – please check.")
    temps = {}
    for T in (v["T_min"], v["T_max"], 20.0, 60.0):
        temps[T] = replace(final, temp_C=T)
    r["temps"] = temps
    for T in (v["T_min"], v["T_max"]):
        if not p0 - 0.005 <= temps[T].pH <= p1 + 0.005:
            notes.check(f"At {T:g} °C the treated water would have pH {temps[T].pH:.2f}, outside {p0:.1f}–{p1:.1f}. "
                        "Please check at which temperature the design point should be met.")
    if temps[60.0].CCPP() > 10:
        notes.check(f"In hot-water systems (60 °C) the water could deposit about {temps[60.0].CCPP():.0f} mg CaCO₃/L – "
                    "please check whether scaling could be a problem.")
    r["notes"] = notes
    return r


def table_12_1(raw, v):
    """NV 212 Table 12.1 range for the raw water, or None if outside the table."""
    ca, pH, alk = raw.ca_mg, raw.pH, raw.alk_mmol
    if 2 <= ca <= 5 and 5.0 <= pH <= 6.0:
        a = raw.species[0] * 1000             # CO2 acidity ≈ free CO2, mmol/L
        x = [0.05, 0.15, 0.25, 0.35, 0.45]
        return float(np.interp(a, x, [15, 10, 6, 1, 0])), float(np.interp(a, x, [20, 15, 11, 7, 2]))
    if ca <= 8 and 6.3 <= pH <= 7.6 and alk <= 0.55:
        x = [0.05, 0.15, 0.25, 0.35]
        cols = {6.5: ([11, 7, 2, 0], [18, 14, 9, 4]), 7.0: ([12, 11, 10, 8], [19, 18, 16, 15]),
                7.5: ([13, 12, 12, 11], [20, 20, 19, 18])}
        ps = sorted(cols)
        p = min(max(pH, 6.5), 7.5)
        lo = [float(np.interp(alk, x, cols[q][0])) for q in ps]
        hi = [float(np.interp(alk, x, cols[q][1])) for q in ps]
        return float(np.interp(p, ps, lo)), float(np.interp(p, ps, hi))
    return None


def dose_range(B, steps, meth, chem):
    """Doses for the corners of the raw-water range (same design point and pre-treatment)."""
    v = B["v"]
    rows = []
    for pH in (v["pH_min"], v["pH_max"]):
        for alk in (v["alk_min"], v["alk_max"]):
            for ca in sorted({v["ca_min"], v["ca_max"]}):
                try:
                    w = raw_at(B, chem["Td"], pH, alk, ca)
                    ws = pretreat(w, steps)
                    rr = chemistry(B, ws, steps, meth)
                    rows.append(dict(pH=pH, alk=alk, ca=ca, co2=rr.get("co2"), marble=rr.get("marble"),
                                     lime=rr.get("lime"), pH_out=rr["final"].pH, ca_out=rr["final"].ca_mg,
                                     alk_out=rr["final"].alk_mmol))
                except (InputError, ValueError) as e:
                    rows.append(dict(pH=pH, alk=alk, ca=ca, error=str(e)))
    return rows


# ---------------------------------------------------------------------------
# C5 units
# ---------------------------------------------------------------------------
def units(B, chem, U):
    """Size the units the chosen method needs. U: dict of dicts from the unit cell."""
    notes = Notes()
    v = B["v"]
    Qmax, Qavg, Qmin = v["Q_max"], v["Q_avg"], v["Q_min"]
    name = chem["method"]
    out = {"entries": []}
    log = []

    def get(sec, key, label, unit_map=None):
        if sec not in U:
            raise InputError(f"The unit cell has no section '{sec}', which the method '{name}' needs.")
        e = entry(U[sec], key, label, unit_map)
        out["entries"].append(e)
        return e["base"]

    if name in MARBLE:
        molde = name == "coagulation + three-media filter (Molde)"
        n = int(get("marble_filter", "n", "number of marble filters"))
        ebct = get("marble_filter", "ebct_min", "minimum EBCT with one filter out")
        vel = get("marble_filter", "v", "filter rate with one filter out")
        dmin = get("marble_filter", "depth_min", "minimum marble depth")
        res = get("marble_filter", "reserve", "marble reserve above design depth")
        vbw = get("marble_filter", "v_bw", "backwash rate (water)")
        tbw = get("marble_filter", "t_bw", "backwash duration")
        nbw = get("marble_filter", "bw_per_week", "backwashes per filter and week")
        rho = get("marble_filter", "rho_bulk", "bulk density of marble")
        if n < 2:
            notes.check("Only one marble filter is given. NV 212 Table 4.16 recommends at least 2 (preferably 4), so that "
                        "the plant can run during backwash and repair.")
            n_eff = 2      # calculation needs n-1 >= 1; flagged above
        else:
            n_eff = n
        f = de.FilterDesign(Qmax, n=n_eff, ebct_min=ebct, v=vel, depth_min=dmin, reserve=res, rho_bulk=rho,
                            v_bw=vbw, t_bw=tbw, bw_per_week=nbw)
        r = f.calc(Qavg)
        if n < 2:
            # single filter: all flow through it, no redundancy
            a = np.ceil(Qmax / vel * 2) / 2
            r.update(n=1, area_each=a, area_total=a, v_one_out=float("nan"), v_all=Qmax / a,
                     ebct_one_out=float("nan"), ebct_all=r["depth"] / (Qmax / a) * 60,
                     bed_volume=a * r["depth"], bed_volume_full=a * (r["depth"] + res),
                     marble_t=a * (r["depth"] + res) * rho, Q_bw_m3h=vbw * a, V_bw_m3=vbw * a * tbw / 60)
        r["rho"] = rho
        r["ebct_min_flow"] = r["depth"] * r["area_total"] / Qmin * 60
        r["bw_share"] = n * nbw * r["V_bw_m3"] / (Qavg * 24 * 7) * 100
        out["marble_filter"] = r
        log += [("marble depth", f"H = v × EBCT = {vel:g} × {ebct:g}/60 (≥ {dmin:g} m), rounded up", f"{r['depth']:.1f} m"),
                ("area per filter", f"Q<sub>max</sub>/v/(n − 1) = {Qmax:.0f}/{vel:g}/{max(n - 1, 1)}, rounded up",
                 f"{r['area_each']:.1f} m²"),
                ("EBCT, all filters in", f"H × A<sub>total</sub>/Q<sub>max</sub> × 60", f"{r['ebct_all']:.0f} min"),
                ("marble, first fill", f"A<sub>total</sub> × (H + reserve) × {rho:g} t/m³", f"{r['marble_t']:.0f} t"),
                ("backwash flow per filter", f"{vbw:g} m/h × {r['area_each']:.1f} m²", f"{r['Q_bw_m3h']:.0f} m³/h")]
        vmax, label = (7.5, "Table 12.3 (three-media)") if molde else (10, "§12.2.2.4 (downflow 8–10 m/h)")
        if n < 2 and r["ebct_all"] < 20:
            notes.check(f"The EBCT appears to be {r['ebct_all']:.0f} min (NV 212 Table 12.3: ≥ 20 min).")
        if n >= 2 and r["ebct_one_out"] < 20:
            notes.check(f"With one filter out of service the EBCT appears to be {r['ebct_one_out']:.0f} min "
                        "(NV 212 Table 12.3: ≥ 20 min).")
        if n >= 2 and r["v_one_out"] > vmax + 1e-6:
            notes.check(f"With one filter out the filter rate appears to be {r['v_one_out']:.1f} m/h "
                        f"(NV 212 {label}: ≤ {vmax:g} m/h).")
        if r["depth"] < 2.5:
            notes.check(f"The marble depth appears to be {r['depth']:.1f} m (NV 212 Table 12.3: ≥ 2.5 m in downflow filters).")
        if r["area_each"] > 100:
            notes.check(f"The area per filter appears to be {r['area_each']:.0f} m² (NV 212 Table 4.16: < 100 m²) – "
                        "please check the number of filters.")
        if vbw < 50 or vbw > 65:
            notes.check(f"Backwash rate {vbw:g} m/h: NV 212 Table 4.19 gives 50–65 m/h (marble, water only), "
                        "Table 12.3 ≥ 60 m/h – please check.")
        if not 8 <= tbw <= 12:
            notes.check(f"Backwash duration {tbw:g} min: NV 212 Table 4.19 gives 8–12 min (marble, water only).")
        if molde:
            notes.info("For a three-media filter, the anthracite/sand layers (NV 212 §12.3.1.4: normally 60 cm anthracite "
                       "+ 40 cm sand) and their backwash must be designed with the coagulation filter.")
    if name == "CO2 stripping + marble filter":
        E_ = get("stripping_tower", "efficiency", "stripping efficiency E")
        vf = get("stripping_tower", "vf", "surface loading of the tower")
        dep = get("stripping_tower", "depth", "packing depth")
        lv = get("stripping_tower", "air_water", "air-to-water ratio L/V")
        nt = int(get("stripping_tower", "n", "number of towers"))
        s = de.stripping(chem["start"], chem["E"]["strip"]["base"], E_, vf, lv, Qmax, max(nt, 1))
        s["depth"] = dep
        out["tower"] = s
        log += [("CO₂ out of the tower", f"c<sub>in</sub> − E (c<sub>in</sub> − c<sub>sat</sub>) = {s['c_in']:.1f} − {E_:g} × "
                 f"({s['c_in']:.1f} − {s['c_sat']:.2f})", f"{s['c_tower']:.1f} mg/L"),
                ("share of the flow through the towers", "from blending (C<sub>T</sub> mass balance)",
                 f"{100 * s['fraction']:.0f} %"),
                ("area per tower", f"Q<sub>max</sub> × share / v<sub>f</sub> / n", f"{s['area_each']:.2f} m² (Ø {s['diameter']:.2f} m)"),
                ("air flow", f"L/V × Q<sub>tower</sub>", f"{s['Q_air']:.0f} m³/h"),
                ("minimum L/V", "(c<sub>in</sub> − c<sub>out</sub>)/(H c<sub>in</sub>)", f"{s['lv_min']:.2f}")]
        if E_ > 0.95:
            notes.check(f"Efficiency {E_:g} is above the 90–95 % NV 212 §8.2.3.4 gives for tower aerators.")
        if not 40 <= vf <= 100:
            notes.check(f"Surface loading {vf:g} m³/(m²·h) is outside NV 212 §8.2.4.4 (40–100).")
        if not 3 <= dep <= 5:
            notes.check(f"Packing depth {dep:g} m is outside NV 212 §8.2.4.4 (3–5 m).")
        if not 5 <= lv <= 100:
            notes.check(f"L/V = {lv:g} is outside NV 212 Table 8.2 (5–100 for towers).")
        if lv < 3 * s["lv_min"]:
            notes.check(f"L/V = {lv:g} is less than 3 × the minimum from Henry's law ({s['lv_min']:.1f}) – the margin "
                        "appears small.")
        if s["fraction"] >= 0.999:
            notes.check("All the water must pass the towers and the target residual may not be reached with this "
                        "efficiency – please check.")
        if nt < 2:
            notes.check("Only one tower: if it is out of service, the marble filter receives the unstripped water. "
                        "Please discuss redundancy.")
        if s["tower"].SI > 0:
            notes.check(f"The tower outlet appears to be supersaturated (SI {s['tower'].SI:+.1f}) – scaling of the packing "
                        "is possible.")
    if name == "lime + CO2":
        conc = get("lime_saturator", "conc", "lime-water concentration")
        load = get("lime_saturator", "surface_load", "surface loading of the lime saturator")
        tc = get("lime_saturator", "contact_time", "contact time in the lime saturator")
        n = int(get("lime_saturator", "n", "number of lime saturators"))
        pur = get("lime_saturator", "purity", "Ca(OH)₂ content of the lime product")
        q_lw = chem["lime"] * Qmax / (conc * 1000)            # m3/h lime water at Q_max
        a = q_lw / load / max(n - 1, 1)
        out["lime"] = dict(q_lw=q_lw, area_each=a, volume=q_lw * tc / 60, n=n,
                           kg_d=chem["lime"] * Qavg * 24 / 1000 / pur)
        log += [("lime water at Q<sub>max</sub>", f"dose × Q<sub>max</sub> / c = {chem['lime']:.1f} × {Qmax:.0f} / ({conc:g} × 1000)",
                 f"{q_lw:.2f} m³/h"),
                ("area per saturator", f"Q<sub>lime water</sub> / surface loading / (n − 1)", f"{a:.2f} m²"),
                ("lime consumption", "dose × Q<sub>avg</sub> × 24/1000 / purity", f"{out['lime']['kg_d']:.0f} kg/d")]
        if conc > 1.75:
            notes.check(f"A lime-water concentration of {conc:g} g Ca(OH)₂/L is above the solubility of Ca(OH)₂ at 20 °C "
                        "(about 1.7 g/L; NV 212 §12.2.3.4 states 1.5 g Ca/L) – please check.")
        if load > 8 or tc < 30:
            notes.check("Surface loading or contact time appear to be outside NV 212 Table 12.2 (2–8 m/h, 30–60 min, "
                        "depending on polymer and lamellae).")
        if n < 2:
            notes.check("Only one lime saturator: please discuss how lime is dosed during cleaning or repair.")
    # CO2 dosing (all methods with a positive dose)
    co2 = chem.get("co2", 0) or 0
    if co2 > 0:
        util = get("co2", "utilisation", "CO₂ utilisation")
        days = get("co2", "storage_days", "CO₂ storage time")
        rho = get("co2", "rho_liquid", "density of liquid CO₂")
        rng = chem.get("range") or []
        dmax = max([x["co2"] for x in rng if x.get("co2") is not None] + [co2])
        dmin = min([x["co2"] for x in rng if x.get("co2") is not None and x["co2"] > 0] + [co2])
        c = dict(kg_d=co2 * Qavg * 24 / 1000 / util, rate_max=dmax * Qmax / 1000 / util,
                 rate_min=dmin * Qmin / 1000 / util)
        c["turndown"] = c["rate_max"] / max(c["rate_min"], 1e-9)
        c["storage_t"] = c["kg_d"] * days / 1000
        c["storage_m3"] = c["storage_t"] / rho
        out["co2"] = c
        log += [("CO₂ consumption", f"{co2:.1f} mg/L × Q<sub>avg</sub> {Qavg:.0f} m³/h × 24/1000 / {util:g}",
                 f"{c['kg_d']:.0f} kg/d"),
                ("CO₂ dosing, max / min", f"highest dose × Q<sub>max</sub>; lowest dose × Q<sub>min</sub> (/ utilisation)",
                 f"{c['rate_max']:.2f} / {c['rate_min']:.2f} kg/h"),
                ("CO₂ storage", f"{days:g} d × consumption / density", f"{c['storage_t']:.1f} t ≈ {c['storage_m3']:.1f} m³")]
        if util > 1:
            notes.stop("The CO₂ utilisation must be ≤ 1.")
        if c["turndown"] > 10:
            notes.check(f"The CO₂ dosing must cover about {c['turndown']:.0f} : 1 (Q<sub>min</sub> × lowest dose to "
                        "Q<sub>max</sub> × highest dose) – please check whether one dosing line can do this.")
    if chem.get("naoh", 0) > 0:
        conc = get("naoh", "conc", "NaOH solution strength")
        dens = get("naoh", "density", "density of the NaOH solution")
        days = get("naoh", "storage_days", "NaOH storage time")
        kg_d = chem["naoh"] * Qavg * 24 / 1000
        out["naoh"] = dict(kg_d=kg_d, L_h_max=chem["naoh"] * Qmax / 1000 / (conc / 100) / dens,
                           storage_m3=kg_d * days / (conc / 100) / dens / 1000)
        log += [("NaOH consumption (100 %)", f"{chem['naoh']:g} mg/L × Q<sub>avg</sub> × 24/1000", f"{kg_d:.1f} kg/d"),
                ("NaOH solution at Q<sub>max</sub>", f"dose × Q<sub>max</sub> / ({conc:g} % × {dens:g} kg/L)",
                 f"{out['naoh']['L_h_max']:.1f} L/h")]
    if "marble" in chem and chem["marble"] and "marble_filter" in out:
        pur = get("marble_filter", "purity", "CaCO₃ content of the marble")
        loss = get("marble_filter", "loss", "extra marble lost as fines")
        kg = chem["marble"] * Qavg * 24 / 1000 / pur * (1 + loss)
        mf = out["marble_filter"]
        loss_m_d = kg / 1000 / mf.get("rho", 1.5) / mf["area_total"] if mf["area_total"] else 0
        out["marble_use"] = dict(kg_d=kg, refill_days=(mf["reserve"] / loss_m_d) if loss_m_d else float("inf"))
        log += [("marble consumption", f"{chem['marble']:.1f} mg/L × Q<sub>avg</sub> × 24/1000 / purity × (1 + loss)",
                 f"{kg:.0f} kg/d"),
                ("marble refill interval", "reserve / bed loss per day", f"{out['marble_use']['refill_days']:.0f} days")]
    source_notes(out["entries"], notes)
    out["notes"], out["log"] = notes, log
    return out


# ---------------------------------------------------------------------------
# C6 costs (estimates)
# ---------------------------------------------------------------------------
def costs(B, U_out, chem, P):
    notes = Notes()
    if not P:
        return dict(rows=[], notes=notes, total=0)
    rows = []
    year = B["v"]["Q_avg"] * 24 * 365
    items = [("co2", "CO₂", U_out.get("co2", {}).get("kg_d")),
             ("marble", "marble", U_out.get("marble_use", {}).get("kg_d")),
             ("lime", "lime", U_out.get("lime", {}).get("kg_d")),
             ("naoh", "NaOH (as 100 %)", U_out.get("naoh", {}).get("kg_d"))]
    ents = []
    for key, label, kg_d in items:
        if not kg_d:
            continue
        e = entry(P, key, f"price of {label}", required=False)
        if e is None:
            notes.check(f"No price given for {label} – the cost total is incomplete.")
            continue
        ents.append(e)
        rows.append(dict(item=label, kg_d=kg_d, price=e["base"], kNOK=kg_d * 365 * e["base"] / 1000, source=e["source"]))
    source_notes(ents, notes)
    total = sum(r["kNOK"] for r in rows)
    return dict(rows=rows, notes=notes, total=total, per_m3=total * 1000 / year if year else 0)


# ---------------------------------------------------------------------------
# C7 summary
# ---------------------------------------------------------------------------
def summary_rows(B, steps, chem, U_out, C):
    rows = []

    def add(group, label, value, unit, source, k="calculated"):
        rows.append(dict(group=group, parameter=label, value=value, unit=unit, source=source, type=k))
    for e in B["E"].values():
        add("Design basis", e["label"], e["value"], e["unit"], e["source"], e["kind"])
    for i, s in enumerate(steps, 1):
        vals = ", ".join(f"{k}={v}" for k, v in s.items() if k not in ("step", "source"))
        add("Pre-treatment", f"{i} {STEPS.get(s.get('step'), s.get('step'))}", vals, "", s.get("source", ""),
            kind(s.get("source", "")))
    add("Corrosion control", "method", chem["method"], "", "your choice (C3)", "decision")
    for e in chem["E"].values():
        if e:
            add("Corrosion control", e["label"], e["value"], e["unit"], e["source"], e["kind"])
    f = chem["final"]
    if chem.get("co2") is not None:
        add("Corrosion control", "CO₂ dose (negative = removal)", f"{chem['co2']:.1f}", "mg/L", "calculated (C4)")
    if chem.get("marble") is not None:
        add("Corrosion control", "CaCO₃ dissolved", f"{chem['marble']:.1f}", "mg/L", "calculated (C4)")
    if chem.get("lime") is not None:
        add("Corrosion control", "lime dose", f"{chem['lime']:.1f}", "mg Ca(OH)₂/L", "calculated (C4)")
    for T, w in chem["temps"].items():
        add("Treated water", f"pH / SI at {T:g} °C", f"{w.pH:.2f} / {w.SI:+.2f}", "– / –", "calculated (C4)")
    add("Treated water", "alkalinity", f"{f.alk_mmol:.2f}", "mmol/L", "calculated (C4)")
    add("Treated water", "calcium", f"{f.ca_mg:.1f}", "mg/L", "calculated (C4)")
    for e in U_out.get("entries", []):
        add("Units – design values", e["label"], e["value"], e["unit"], e["source"], e["kind"])
    for sec, lst in (("Units – results", U_out.get("log", [])),):
        for name, eq, res in lst:
            add(sec, name, res, "", "calculated: " + eq)
    for r in C.get("rows", []):
        add("Costs (ESTIMATE)", r["item"], f"{r['kNOK']:.0f}", "kNOK/year", f"price {r['price']:g} NOK/kg: {r['source']}",
            "estimate")
    return rows
