"""
design_example.py - the worked design example of Part B (TVM4173).

One place for all design values and calculations of the notebooks B1-B8,
so that every chapter uses the same numbers. Every value carries its
source: a section or table of Norsk Vann 212/2015 (NV 212), NV 274/2022,
a lecture M08 slide, a data sheet - or it is marked as an ASSUMPTION.

Example: "Example waterworks", a soft humic lake water (neutral teaching
example, not a semester-project case). Main design: CO2 dosing + marble
filter. Second example (B7): CO2-rich groundwater, stripping + marble.
"""
from dataclasses import dataclass, field, replace, asdict
import numpy as np
from scipy.optimize import brentq
import carbonate as cb

# ---------------------------------------------------------------------------
# Sources (short key -> full reference shown in the notebooks)
# ---------------------------------------------------------------------------
SRC = {
    "NV212-2": "NV 212/2015 ch. 2 'Dimensjonerende vannmengde'",
    "NV212-4.7.4": "NV 212/2015 §4.7.4 (filter design procedure)",
    "NV212-T4.1": "NV 212/2015 Table 4.1 (molar masses)",
    "NV212-T4.16": "NV 212/2015 Table 4.16 (filter layout)",
    "NV212-T4.19": "NV 212/2015 Table 4.19 (backwash rates)",
    "NV212-8.2": "NV 212/2015 §8.2.3.4 and §8.2.4.4 (tower aerators)",
    "NV212-T8.2": "NV 212/2015 Table 8.2 (L/V ratios)",
    "NV212-11.2": "NV 212/2015 Table 11.1 (C·t values for chlorine)",
    "NV212-12.2.1": "NV 212/2015 §12.2.1 (carbonation: targets and methods)",
    "NV212-T12.1": "NV 212/2015 §12.2.2.4, Table 12.1 (CO2 doses)",
    "NV212-12.2.2.4": "NV 212/2015 §12.2.2.4 (marble filter design values)",
    "NV212-T12.3": "NV 212/2015 §12.3.1.4, Table 12.3 (marble layer in filters)",
    "NV274": "NV 274/2022 (target water quality; lecture M08 slide 'Target water quality')",
    "M08-strip": "Lecture M08, slide 'Physical treatment: CO2 stripping'",
    "M08-25": "Lecture M08, slide 'Target water quality'",
    "M08-26": "Lecture M08, slide 'Targets per pipe material'",
    "M08-29": "Lecture M08, slides 'pH and disinfection' and 'Where to raise the pH?'",
    "M08-38": "Lecture M08, slide 'CO2 + marble filter: design'",
    "M08-43": "Lecture M08, slides 'Molde process: design values' and 'Molde process'",
    "M08-51": "Lecture M08, slide 'Choosing a method'",
    "PB1982": "Plummer & Busenberg (1982) constants, used in carbonate.py",
    "Visnes": "Visnes Kalk, product data sheet '1-3 mm Water treatment marble'",
    "NIST": "NIST Chemistry WebBook, CO2 saturated liquid at -20 °C",
    "IMARC": "IMARC Group, liquid CO2 price update Q2 2026 (Germany)",
    "TED-IVAR": "TED notice 263316-2020 (IVAR IKS, filter marble contract)",
    "Morris": "Morris (1966), pKa of HOCl as function of temperature",
    "A": "ASSUMPTION for this example – check for a real plant",
}


def src(*keys):
    """Text with the full references for the given keys."""
    return "; ".join(SRC[k] for k in keys)


# ---------------------------------------------------------------------------
# Design basis (B1)
# ---------------------------------------------------------------------------
@dataclass
class Basis:
    # flows (approved for the example)
    Q_avg: float = 50.0          # L/s, average day            [A]
    Q_max: float = 80.0          # L/s, max hour of max day    [A]
    Q_min: float = 15.0          # L/s, minimum (night, winter)[A]
    # temperature range of the lake water
    T_min: float = 2.0           # °C, winter                  [A]
    T_max: float = 12.0          # °C, late summer             [A]
    # raw water (lab data, measured at 5 °C)
    pH: float = 6.6
    alk: float = 0.10            # mmol/L
    ca: float = 2.5              # mg/L
    ec: float = 35.0             # µS/cm
    T_lab: float = 5.0
    # seasonal range of the raw water used for the dosing equipment [A]
    pH_range: tuple = (6.4, 6.8)
    alk_range: tuple = (0.05, 0.15)
    # treatment target (chosen inside the NV 274 zone, see B3)
    pH_target: float = 8.3
    alk_target: float = 1.0      # mmol/L
    T_design: float = 2.0        # °C at which the target pH is met (B3)

    @property
    def Q_dim(self):
        """m3/h. NV 212 ch. 2: Q_dim = max hour in max day (no equalisation
        tank in front of the treatment step is assumed)."""
        return self.Q_max * 3.6

    def raw(self, temp_C=None, pH=None, alk=None):
        w = cb.Water.from_pH_alk(pH or self.pH, self.alk if alk is None else alk, self.ca,
                                 temp_C=self.T_lab, ec_uS_cm=self.ec, name="raw water")
        # the lab sample was measured at 5 °C; at another temperature the same
        # water (same C_T and alkalinity) has another pH
        return w if temp_C is None else replace(w, temp_C=temp_C)


BASIS = Basis()


# ---------------------------------------------------------------------------
# Chemistry (B3): doses from a target point (two vectors in the Deffeyes diagram)
# ---------------------------------------------------------------------------
def doses(raw, pH_t, alk_t, temp_C):
    """CO2 dose and marble dissolution that move `raw` to the target point
    (pH_t, alk_t) at temperature temp_C.

    marble (mmol/L) = (Alk_t - Alk_raw) / 2      [each CaCO3 adds 2 meq alkalinity]
    CO2 (mmol/L)   = C_T,target - C_T,raw - marble   [each CaCO3 adds 1 mmol C_T]
    Negative CO2 = stripping needed. Returns mg/L and the list of waters.
    """
    m = (alk_t - raw.alk_mmol) / 2
    if m < 0:
        raise ValueError("Target alkalinity below raw-water alkalinity: marble cannot remove alkalinity.")
    after = replace(raw, temp_C=temp_C).apply(cb.caco3(m * cb.M["CaCO3"]))   # for I and Ca
    ct_t = 1000 * cb.CT_from_pH_alk(pH_t, alk_t / 1000, after.K)
    co2 = ct_t - raw.ct_mmol - m                                              # mmol/L
    w = replace(raw, temp_C=temp_C)
    s1 = w.apply(cb.co2_dose(co2 * cb.M["CO2"]) if co2 >= 0 else cb.co2_strip(-co2 * cb.M["CO2"]))
    s2 = s1.apply(cb.caco3(m * cb.M["CaCO3"], label=f"Marble filter {m * cb.M['CaCO3']:.0f} mg CaCO3/L"))
    return {"co2_mg": co2 * cb.M["CO2"], "marble_mg": m * cb.M["CaCO3"], "waters": [w, s1, s2]}


def treated(raw, co2_mg, marble_mg, temp_C):
    """Treated water for given doses at temperature temp_C."""
    w = replace(raw, temp_C=temp_C)
    s1 = w.apply(cb.co2_dose(co2_mg) if co2_mg >= 0 else cb.co2_strip(-co2_mg))
    return s1.apply(cb.caco3(marble_mg))


def equilibrium_limit(raw, co2_mg, temp_C):
    """What an infinitely long marble filter would give after the CO2 dose."""
    w = replace(raw, temp_C=temp_C).apply(cb.co2_dose(co2_mg))
    return w.apply(cb.marble_to_equilibrium())


def nv_table_12_1(alk):
    """NV 212 Table 12.1, right half (raw water 4-8 mg Ca/L, pH 6.5),
    interpolated in alkalinity: (low, high) CO2 dose in mg/L."""
    a = np.array([0.05, 0.15, 0.25, 0.35])
    lo = np.array([11, 7, 2, 0]); hi = np.array([18, 14, 9, 4])
    return float(np.interp(alk, a, lo)), float(np.interp(alk, a, hi))


# ---------------------------------------------------------------------------
# Marble filter (B4)
# ---------------------------------------------------------------------------
@dataclass
class FilterDesign:
    Q_dim: float                  # m3/h
    n: int = 4                    # number of filters            [NV212-T4.16: min 2, preferably 4]
    ebct_min: float = 20.0        # min EBCT with one filter out [NV212-T12.3: >= 20 min; §12.2.2.4: 20-30]
    v: float = 8.0                # m/h with one filter out      [NV212-12.2.2.4: 8-10 m/h downflow]
    depth_min: float = 2.5        # m                            [NV212-T12.3 / §12.3.1.4: >= 2.5 m]
    reserve: float = 0.3          # m marble above design depth, consumed between refills [A]
    support: float = 0.3          # m gravel support layer       [A, NV 212 §4.7.3.8 gives no number]
    water_above: float = 1.5      # m water above the bed (available head loss) [A, NV212-4.7.4: 2-3 m total]
    freeboard: float = 0.3        # m                            [NV212-T4.16: about 30 cm]
    expansion: float = 0.10       # bed expansion in backwash    [NV212-12.2.2.4: 10 %]
    rho_bulk: float = 1.5         # t/m3 marble bulk density     [Visnes: 1.45-1.55 kg/dm3]
    v_bw: float = 60.0            # m/h backwash water           [NV212-T12.3: >= 60; T4.19: 50-65]
    t_bw: float = 10.0            # min                          [NV212-T4.19: 8-12 min]
    bw_per_week: float = 1.0      # washes per filter per week   [A]

    def calc(self, Q_avg_m3h=None):
        Q = self.Q_dim
        n_on = self.n - 1                                  # one filter out (backwash / repair)
        depth = max(self.depth_min, self.v * self.ebct_min / 60)
        depth = np.ceil(depth * 10 - 1e-9) / 10            # round up to 0.1 m
        a_on = Q / self.v                                  # area needed in service
        a = np.ceil(a_on / n_on * 2 - 1e-9) / 2            # per filter, rounded up to 0.5 m2
        A = a * self.n
        v_on, v_all = Q / (a * n_on), Q / A
        ebct_on, ebct_all = depth / v_on * 60, depth / v_all * 60
        r = dict(n=self.n, depth=depth, area_each=a, area_total=A, side=np.sqrt(a),
                 v_one_out=v_on, v_all=v_all, ebct_one_out=ebct_on, ebct_all=ebct_all,
                 bed_volume=A * depth, bed_volume_full=A * (depth + self.reserve),
                 marble_t=A * (depth + self.reserve) * self.rho_bulk,
                 reserve=self.reserve, box_height=self.support + depth + self.reserve + self.water_above + self.freeboard,
                 expansion_m=self.expansion * (depth + self.reserve))
        if Q_avg_m3h:
            r["ebct_avg"] = depth * A / Q_avg_m3h * 60
        r["Q_bw_m3h"] = self.v_bw * a
        r["V_bw_m3"] = r["Q_bw_m3h"] * self.t_bw / 60
        return r


# ---------------------------------------------------------------------------
# Chemicals and costs (B5) - prices are ESTIMATES
# ---------------------------------------------------------------------------
@dataclass
class Chemicals:
    co2_mg: float                 # design dose
    co2_max_mg: float             # highest dose (worst raw water)
    co2_min_mg: float             # lowest dose (best raw water)
    marble_mg: float              # CaCO3 dissolved, mg/L
    utilisation: float = 0.90     # fraction of dosed CO2 that dissolves [A]
    purity: float = 0.98          # CaCO3 content of the marble [Visnes: >= 98 %]
    marble_loss: float = 0.05     # extra marble lost as fines in backwash [A]
    storage_days: float = 30.0    # CO2 storage at average consumption [A]
    rho_co2: float = 1.03         # t/m3 liquid CO2 at -20 °C [NIST]
    price_co2: float = 3.0        # NOK/kg delivered, incl. tank rent [ESTIMATE]
    price_marble: float = 2.0     # NOK/kg delivered in big bags [ESTIMATE]

    def calc(self, basis, filt=None):
        d = 86.4                                           # (L/s) * (mg/L) -> kg/d
        co2_avg = self.co2_mg * basis.Q_avg * d / 1000 / self.utilisation
        co2_rate_max = self.co2_max_mg * basis.Q_max * 3.6 / 1000 / self.utilisation   # kg/h
        co2_rate_min = self.co2_min_mg * basis.Q_min * 3.6 / 1000 / self.utilisation
        marble_avg = self.marble_mg * basis.Q_avg * d / 1000 / self.purity * (1 + self.marble_loss)
        V_year = basis.Q_avg * 86400 * 365 / 1000                  # m3/year
        storage_t = co2_avg * self.storage_days / 1000
        r = dict(co2_kg_d=co2_avg, co2_t_y=co2_avg * 0.365, co2_rate_max=co2_rate_max,
                 co2_rate_min=co2_rate_min, turndown=co2_rate_max / max(co2_rate_min, 1e-9),
                 co2_storage_t=storage_t, co2_storage_m3=storage_t / self.rho_co2,
                 marble_kg_d=marble_avg, marble_t_y=marble_avg * 0.365, V_year=V_year,
                 cost_co2=co2_avg * 365 * self.price_co2, cost_marble=marble_avg * 365 * self.price_marble)
        r["cost_total"] = r["cost_co2"] + r["cost_marble"]
        r["cost_per_m3"] = r["cost_total"] / V_year
        if filt is not None:
            r["first_fill_cost"] = filt["marble_t"] * 1000 * self.price_marble
            # bed depth lost per day (m) and days until the reserve is used up
            loss_m_d = marble_avg / 1000 / 1.5 / filt["area_total"]
            r["bed_loss_mm_month"] = loss_m_d * 1000 * 30
            r["refill_days"] = (filt["bed_volume_full"] - filt["bed_volume"]) / filt["area_total"] / loss_m_d
        return r


def dose_range(basis=BASIS):
    """CO2 doses for the seasonal raw-water range (same target point).
    Low alkalinity and low pH need the most CO2? - computed, not guessed."""
    out = {}
    for p in basis.pH_range:
        for a in basis.alk_range:
            r = doses(basis.raw(pH=p, alk=a), basis.pH_target, basis.alk_target, basis.T_design)
            out[(p, a)] = r["co2_mg"]
    return out


# ---------------------------------------------------------------------------
# Disinfection link (B6)
# ---------------------------------------------------------------------------
def pKa_hocl(temp_C):
    """Morris (1966): pKa = 3000/T - 10.0686 + 0.0253 T (T in K)."""
    T = temp_C + 273.15
    return 3000 / T - 10.0686 + 0.0253 * T


def hocl_fraction(pH, temp_C):
    return 1 / (1 + 10 ** (pH - pKa_hocl(temp_C)))


# ---------------------------------------------------------------------------
# Second example (B7): CO2-rich groundwater, stripping tower + marble filter
# ---------------------------------------------------------------------------
GW = dict(pH=6.4, alk=0.70, ca=12.0, ec=120.0, T=6.0)     # carbonate.example_waters()


def gw_raw():
    return cb.Water.from_pH_alk(GW["pH"], GW["alk"], GW["ca"], temp_C=GW["T"], ec_uS_cm=GW["ec"],
                                name="groundwater")


def stripping(raw, co2_target=13.0, efficiency=0.90, vf=60.0, air_water=10.0, Q_dim=288.0, n_towers=2):
    """Tower stripping of part of the flow, then blending with the bypass.

    efficiency: E = (c_in - c_out)/(c_in - c_sat)   [NV212-8.2 eq. for E; towers 90-95 %]
    vf: surface loading m3/(m2 h)                   [NV212-8.2: 40-100]
    air_water: L/V = Q_air/Q_water                  [NV212-T8.2: towers 5-100]
    co2_target: free CO2 after blending             [NV212-12.2.1: 12-15 mg/L]
    """
    c_sat = cb.co2_saturation_mg(raw.temp_C)
    c_in = raw.co2_mg
    c_tower = c_in - efficiency * (c_in - c_sat)
    tower = raw.apply(cb.strip_co2_to(raw, c_tower))

    def co2_after(f):                         # blending mixes C_T (alkalinity is the same)
        return replace(raw, CT=f * tower.CT + (1 - f) * raw.CT).co2_mg - co2_target
    if co2_target >= c_in:
        f = 0.0
    elif co2_target <= c_tower:
        f = 1.0
    else:
        f = brentq(co2_after, 0, 1)
    blend = replace(raw, CT=f * tower.CT + (1 - f) * raw.CT, label=f"stripped to {co2_target:g} mg/L CO2")
    Q_t = f * Q_dim
    area = Q_t / vf / n_towers if n_towers else 0
    # minimum air/water ratio for counter-current stripping: (c_in - c_out)/(H c_in),
    # H = dimensionless Henry constant = 1/(K_H R T)
    H = 1 / (10 ** cb.log_K(raw.temp_C)["KH"] * 0.082057 * (raw.temp_C + 273.15))
    lv_min = (c_in - c_tower) / (H * c_in)
    return dict(c_in=c_in, c_sat=c_sat, c_tower=c_tower, fraction=f, Q_tower=Q_t,
                area_each=area, diameter=np.sqrt(4 * area / np.pi) if area > 0 else 0.0,
                Q_air=air_water * Q_t, H=H, lv_min=lv_min, blend=blend, tower=tower)


# ---------------------------------------------------------------------------
# Summary register (B8): every design value with its source
# ---------------------------------------------------------------------------
def main_design(basis=BASIS, n=4, ebct=20.0, v=8.0, price_co2=3.0, price_marble=2.0):
    """The complete main design as one dict (used in B3-B8)."""
    raw = basis.raw()
    d = doses(raw, basis.pH_target, basis.alk_target, basis.T_design)
    rng = dose_range(basis)
    f = FilterDesign(basis.Q_dim, n=n, ebct_min=ebct, v=v).calc(basis.Q_avg * 3.6)
    ch = Chemicals(d["co2_mg"], max(rng.values()), min(rng.values()), d["marble_mg"],
                   price_co2=price_co2, price_marble=price_marble).calc(basis, f)
    finals = {T: treated(raw, d["co2_mg"], d["marble_mg"], T) for T in (basis.T_min, basis.T_max)}
    return dict(raw=raw, doses=d, range=rng, filter=f, chem=ch, finals=finals,
                prices=(price_co2, price_marble))


def register(basis=BASIS, design=None):
    """Rows (group, parameter, value, unit, source) for the summary table."""
    D = design or main_design(basis)
    d, f, ch, fin = D["doses"], D["filter"], D["chem"], D["finals"]
    w2, w12 = fin[basis.T_min], fin[basis.T_max]
    R = [
        ("Basis", "Average flow Q_avg", f"{basis.Q_avg:g}", "L/s", src("A")),
        ("Basis", "Maximum flow Q_max (max hour, max day)", f"{basis.Q_max:g}", "L/s", src("A")),
        ("Basis", "Minimum flow Q_min", f"{basis.Q_min:g}", "L/s", src("A")),
        ("Basis", "Design flow Q_dim = Q_max", f"{basis.Q_dim:.0f}", "m³/h", src("NV212-2")),
        ("Basis", "Water temperature", f"{basis.T_min:g}–{basis.T_max:g}", "°C", src("A")),
        ("Basis", "Raw water pH / Alk / Ca (at 5 °C)", f"{basis.pH} / {basis.alk} / {basis.ca}", "– / mmol/L / mg/L", "example water (carbonate.py); " + src("A")),
        ("Basis", "Target pH", "8.0–8.5", "–", src("NV274")),
        ("Basis", "Target alkalinity", "0.6–1.0", "mmol/L", src("NV274", "NV212-12.2.1")),
        ("Basis", "Target calcium", "15–25", "mg/L", src("NV274", "NV212-12.2.1")),
        ("Chemistry", "Design point (pH, Alk) at 2 °C", f"{basis.pH_target}, {basis.alk_target}", "–, mmol/L", "chosen in B3 inside the target zone"),
        ("Chemistry", "CO₂ dose (design)", f"{d['co2_mg']:.1f}", "mg/L", "calculated (B3); compare " + src("NV212-T12.1")),
        ("Chemistry", "CO₂ dose range (raw-water range)", f"{min(D['range'].values()):.1f}–{max(D['range'].values()):.1f}", "mg/L", "calculated (B3) for the assumed raw-water range"),
        ("Chemistry", "CaCO₃ dissolved in marble filter", f"{d['marble_mg']:.0f}", "mg/L", "calculated (B3)"),
        ("Chemistry", "Raw water C_T", f"{D['raw'].ct_mmol:.2f}", "mmol/L", "calculated from pH and alkalinity"),
        ("Chemistry", "Treated water pH at 2 / 12 °C", f"{w2.pH:.2f} / {w12.pH:.2f}", "–", "calculated, " + src("PB1982")),
        ("Chemistry", "Treated water Ca", f"{w2.ca_mg:.1f}", "mg/L", "calculated"),
        ("Chemistry", "Treated water SI at 2 / 12 °C", f"{w2.SI:+.2f} / {w12.SI:+.2f}", "–", "calculated"),
        ("Marble filter", "Number of filters", f"{f['n']}", "–", src("NV212-T4.16")),
        ("Marble filter", "Filter rate, one filter out / all in", f"{f['v_one_out']:.1f} / {f['v_all']:.1f}", "m/h", src("NV212-12.2.2.4")),
        ("Marble filter", "EBCT, one filter out / all in", f"{f['ebct_one_out']:.0f} / {f['ebct_all']:.0f}", "min", src("NV212-T12.3", "NV212-12.2.2.4")),
        ("Marble filter", "Marble depth (design) + reserve", f"{f['depth']:.1f} + {f['reserve']:.1f}", "m", src("NV212-T12.3") + "; reserve: " + src("A")),
        ("Marble filter", "Area per filter / total", f"{f['area_each']:.1f} / {f['area_total']:.0f}", "m²", "calculated"),
        ("Marble filter", "Grain size", "1–2.5", "mm", src("NV212-12.2.2.4")),
        ("Marble filter", "Backwash rate / duration", "60 / 10", "m/h / min", src("NV212-T12.3", "NV212-T4.19")),
        ("Marble filter", "Backwash flow / volume per wash", f"{f['Q_bw_m3h']:.0f} / {f['V_bw_m3']:.0f}", "m³/h / m³", "calculated"),
        ("Marble filter", "Marble in all filters (first fill)", f"{f['marble_t']:.0f}", "t", "calculated; bulk density " + src("Visnes")),
        ("Chemicals", "CO₂ consumption (average)", f"{ch['co2_kg_d']:.0f}", "kg/d", "calculated; utilisation 90 %: " + src("A")),
        ("Chemicals", "CO₂ dosing capacity max / min", f"{ch['co2_rate_max']:.1f} / {ch['co2_rate_min']:.2f}", "kg/h", "calculated (Q_max × highest dose, Q_min × lowest dose)"),
        ("Chemicals", "CO₂ storage (30 days)", f"{ch['co2_storage_t']:.1f} t ≈ {ch['co2_storage_m3']:.1f}", "m³", "30 days: " + src("A") + "; density " + src("NIST")),
        ("Chemicals", "Marble consumption", f"{ch['marble_kg_d']:.0f} kg/d ≈ {ch['marble_t_y']:.0f}", "t/year", "calculated; purity " + src("Visnes") + "; 5 % fines loss " + src("A")),
        ("Costs (ESTIMATE)", "CO₂", f"{ch['cost_co2'] / 1000:.0f}", "kNOK/year", f"price {D['prices'][0]:g} NOK/kg: ESTIMATE, see B5"),
        ("Costs (ESTIMATE)", "Marble", f"{ch['cost_marble'] / 1000:.0f}", "kNOK/year", f"price {D['prices'][1]:g} NOK/kg: ESTIMATE, see B5"),
        ("Costs (ESTIMATE)", "Chemicals per m³", f"{ch['cost_per_m3']:.2f}", "NOK/m³", "ESTIMATE"),
    ]
    return [dict(group=g, parameter=p, value=v, unit=u, source=s) for g, p, v, u, s in R]
