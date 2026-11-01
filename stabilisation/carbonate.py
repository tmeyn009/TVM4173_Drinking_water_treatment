"""
carbonate.py - carbonate system and calcite saturation for TVM4173
(Drinking water treatment, NTNU). Stabilisation and corrosion control.

What this module does
---------------------
* Equilibrium constants K1, K2, Kw, K_calcite and K_H as a function of
  temperature (Plummer & Busenberg 1982; Harned & Owen 1958 for Kw).
* Activity coefficients from the Davies equation; ionic strength from
  conductivity (Russell 1976: I = 1.6e-5 * EC[uS/cm]).
* A water is described by temperature, ionic strength, calcium,
  alkalinity and total inorganic carbon C_T. The pH follows from these.
* Treatment steps (CO2 stripping/dosing, NaOH, lime, CaCO3 dissolution,
  acid, coagulant) change alkalinity, C_T and calcium - they are the
  vectors of the Deffeyes diagram.
* Saturation: exact saturation index SI, Langelier index (slide formula)
  and the calcium carbonate precipitation potential CCPP.

Simplifications (say so in your report if they matter)
* Ion pairs (CaHCO3+, CaCO3(aq)) are ignored. In soft water this changes
  SI by a few hundredths; PHREEQC includes them.
* Only the carbonate system and water contribute to alkalinity
  (no NOM, silicate, phosphate, borate).
* Ionic strength is estimated from conductivity and then updated
  approximately for each dose.
* Everything is EQUILIBRIUM. A real marble filter only approaches the
  equilibrium; that is why its size comes from design values (contact
  time), not from this module.

Units in the user functions: mmol/L, mg/L, degrees C.
Alkalinity in mmol/L means mmol/L of monovalent charge (= meq/L), as in
Norsk Vann reports.
"""
from dataclasses import dataclass, field, replace
import numpy as np
from scipy.optimize import brentq

# molar masses, g/mol
M = {"Ca": 40.078, "CO2": 44.010, "CaCO3": 100.087, "NaOH": 39.997,
     "Ca(OH)2": 74.093, "Na2CO3": 105.989, "Fe": 55.845, "Al": 26.982,
     "HCl": 36.461}

# Target water quality, Norsk Vann 274/2022 (lecture M08, slide 25)
TARGETS = {"pH": (8.0, 8.5), "Alk": (0.6, 1.0), "Ca": (15.0, 25.0)}

P_CO2_AIR = 10 ** -3.38        # atm, about 420 ppm CO2 in air


# ---------------------------------------------------------------------------
# Constants and activity
# ---------------------------------------------------------------------------
def log_K(temp_C):
    """log10 of the thermodynamic constants at temp_C (0-50 C).

    Returns dict with K1, K2, Ks (calcite), KH (mol/L/atm), Kw.
    Plummer & Busenberg (1982), Geochim. Cosmochim. Acta 46, 1011;
    Kw: Harned & Owen (1958).
    """
    T = temp_C + 273.15
    lgT = np.log10(T)
    return {
        "K1": -356.3094 - 0.06091964 * T + 21834.37 / T + 126.8339 * lgT - 1684915 / T**2,
        "K2": -107.8871 - 0.03252849 * T + 5151.79 / T + 38.92561 * lgT - 563713.9 / T**2,
        "Ks": -171.9065 - 0.077993 * T + 2839.319 / T + 71.595 * lgT,
        "KH": 108.3865 + 0.01985076 * T - 6919.53 / T - 40.45154 * lgT + 669365 / T**2,
        "Kw": -4470.99 / T + 6.0875 - 0.01706 * T,
    }


def davies_A(temp_C):
    """Debye-Hueckel A parameter (approximation, 0-40 C)."""
    return 0.4883 + 8.074e-4 * temp_C


def log_gamma(z, I, temp_C):
    """Davies equation: log10 activity coefficient of an ion with charge z."""
    sI = np.sqrt(I)
    return -davies_A(temp_C) * z**2 * (sI / (1 + sI) - 0.3 * I)


def I_from_conductivity(ec_uS_cm):
    """Ionic strength (mol/L) from conductivity, Russell (1976)."""
    return 1.6e-5 * ec_uS_cm


class _Consts:
    """Conditional ('mixed') constants for measured pH = -log a(H+)
    and concentrations of all other species."""

    def __init__(self, temp_C, I, activity=True):
        lk = log_K(temp_C)
        g1 = 10 ** log_gamma(1, I, temp_C) if activity else 1.0
        g2 = 10 ** log_gamma(2, I, temp_C) if activity else 1.0
        self.g1, self.g2 = g1, g2
        self.K1 = 10 ** lk["K1"] / g1          # [HCO3-] aH / [CO2*]
        self.K2 = 10 ** lk["K2"] * g1 / g2     # [CO3--] aH / [HCO3-]
        self.Kw = 10 ** lk["Kw"] / g1          # [OH-] aH
        self.Ks = 10 ** lk["Ks"] / g2**2       # [Ca++][CO3--] at saturation
        self.KH = 10 ** lk["KH"]
        self.lk = lk


def alphas(pH, K):
    """Fractions of CO2*, HCO3-, CO3-- in C_T."""
    h = 10.0 ** (-np.asarray(pH, dtype=float))
    D = h * h + K.K1 * h + K.K1 * K.K2
    return h * h / D, K.K1 * h / D, K.K1 * K.K2 / D


def alk_from_pH_CT(pH, CT, K):
    """Alkalinity (eq/L) = [HCO3-] + 2[CO3--] + [OH-] - [H+]."""
    a0, a1, a2 = alphas(pH, K)
    h = 10.0 ** (-np.asarray(pH, dtype=float))
    return CT * (a1 + 2 * a2) + K.Kw / h - h / K.g1


def CT_from_pH_alk(pH, Alk, K):
    a0, a1, a2 = alphas(pH, K)
    h = 10.0 ** (-pH)
    return (Alk - K.Kw / h + h / K.g1) / (a1 + 2 * a2)


def pH_from_alk_CT(Alk, CT, K):
    f = lambda p: alk_from_pH_CT(p, CT, K) - Alk
    lo, hi = 2.0, 13.5
    if f(lo) > 0 or f(hi) < 0:
        raise ValueError("No pH for this alkalinity and C_T "
                         "(alkalinity must lie between about 0 and 2*C_T).")
    return brentq(f, lo, hi, xtol=1e-10)


# ---------------------------------------------------------------------------
# Water
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Water:
    """A water in equilibrium with itself (not necessarily with calcite).

    Internal units: mol/L and eq/L. Use the constructors and properties,
    which work in mmol/L and mg/L.
    """
    temp_C: float
    I: float          # ionic strength, mol/L
    Ca: float         # mol/L
    Alk: float        # eq/L
    CT: float         # mol/L
    name: str = ""
    activity: bool = True
    label: str = ""   # treatment step that produced this water

    # ---- constructors ----------------------------------------------------
    @classmethod
    def from_pH_alk(cls, pH, alk_mmol, ca_mg, temp_C=5.0, ec_uS_cm=None,
                    I=None, name="", activity=True):
        """Water from the usual lab data: pH, alkalinity (mmol/L),
        calcium (mg/L), temperature, and conductivity (uS/cm) or I."""
        if I is None:
            if ec_uS_cm is None:
                raise ValueError("Give the conductivity (ec_uS_cm) or the ionic strength I.")
            I = I_from_conductivity(ec_uS_cm)
        K = _Consts(temp_C, I, activity)
        Alk = alk_mmol / 1000
        CT = CT_from_pH_alk(pH, Alk, K)
        if CT <= 0:
            raise ValueError("This pH and alkalinity give C_T <= 0 - check the data.")
        return cls(temp_C, I, ca_mg / M["Ca"] / 1000, Alk, CT, name, activity, "raw water")

    # ---- derived properties ---------------------------------------------
    @property
    def K(self):
        return _Consts(self.temp_C, self.I, self.activity)

    @property
    def pH(self):
        return pH_from_alk_CT(self.Alk, self.CT, self.K)

    @property
    def species(self):
        """mol/L of CO2*, HCO3-, CO3--."""
        a0, a1, a2 = alphas(self.pH, self.K)
        return self.CT * a0, self.CT * a1, self.CT * a2

    @property
    def co2_mg(self):
        """Free (dissolved) CO2 in mg/L."""
        return self.species[0] * M["CO2"] * 1000

    @property
    def ca_mg(self):
        return self.Ca * M["Ca"] * 1000

    @property
    def alk_mmol(self):
        return self.Alk * 1000

    @property
    def ct_mmol(self):
        return self.CT * 1000

    @property
    def SI(self):
        """Exact calcite saturation index log10(IAP/Ks)."""
        co3 = self.species[2]
        return float(np.log10(self.Ca * co3 / self.K.Ks))

    def pHs(self, slide_formula=True):
        """Saturation pH (Langelier).

        slide_formula=True: pHs = pK2 - pKs - log[Ca] - log[Alk] + S
        (HCO3- approximated by alkalinity, as on slides 21 and 59).
        slide_formula=False: the actual [HCO3-] is used instead.
        S = -log(gamma_Ca * gamma_HCO3) is the ionic-strength correction.
        """
        K = self.K
        lk = K.lk
        S = -(np.log10(K.g1) + np.log10(K.g2))
        hco3 = self.Alk if slide_formula else self.species[1]
        return float(-lk["K2"] + lk["Ks"] - np.log10(self.Ca) - np.log10(hco3) + S)

    @property
    def LSI(self):
        """Langelier index pH - pHs with the slide formula."""
        return self.pH - self.pHs(slide_formula=True)

    def CCPP(self):
        """Calcium carbonate precipitation potential in mg CaCO3/L.
        > 0: would precipitate this much; < 0: would dissolve this much."""
        x = self._x_to_saturation()
        return x * M["CaCO3"] * 1000

    def _x_to_saturation(self):
        """mol/L CaCO3 to precipitate (+) or dissolve (-) to reach SI = 0."""
        def si_after(x):
            w = replace(self, Ca=self.Ca - x, Alk=self.Alk - 2 * x,
                        CT=self.CT - x, I=max(self.I - 3 * x, 1e-6))
            return w.SI
        x_hi = min(self.Ca, self.CT, self.Alk / 2) * (1 - 1e-9)
        x_lo = -0.02  # dissolve up to 20 mmol/L
        if si_after(0) == 0:
            return 0.0
        return brentq(si_after, x_lo, x_hi, xtol=1e-12)

    # ---- treatment -------------------------------------------------------
    def apply(self, step):
        """Apply a Step and return the new Water."""
        if step.to_equilibrium:
            x = -self._x_to_saturation()          # mol/L dissolved (+)
            step = caco3(x * M["CaCO3"] * 1000, label=step.label)
        new = replace(self,
                      Alk=self.Alk + step.dAlk / 1000,
                      CT=self.CT + step.dCT / 1000,
                      Ca=self.Ca + step.dCa / 1000,
                      I=max(self.I + step.dI / 1000, 1e-6),
                      label=step.label)
        if new.CT <= 0 or new.Ca < 0:
            raise ValueError(f"'{step.label}' removes more than the water contains.")
        new.pH  # raises ValueError if impossible
        return new

    def treat(self, steps):
        """Apply a list of steps; returns the list of waters (raw first)."""
        out = [self]
        for s in steps:
            out.append(out[-1].apply(s))
        return out

    def summary(self):
        """Dict with the numbers a student reports."""
        return {"pH": self.pH, "Alk (mmol/L)": self.alk_mmol,
                "C_T (mmol/L)": self.ct_mmol, "Ca (mg/L)": self.ca_mg,
                "free CO2 (mg/L)": self.co2_mg, "SI": self.SI,
                "LSI (slide)": self.LSI, "CCPP (mg CaCO3/L)": self.CCPP(),
                "T (C)": self.temp_C, "I (mmol/L)": self.I * 1000}


# ---------------------------------------------------------------------------
# Treatment steps = Deffeyes vectors (all doses in mg/L of the chemical)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    label: str
    dAlk: float = 0.0     # mmol/L (meq/L)
    dCT: float = 0.0      # mmol/L
    dCa: float = 0.0      # mmol/L
    dI: float = 0.0       # mmol/L, approximate
    to_equilibrium: bool = False


def co2_dose(mg):
    n = mg / M["CO2"]
    return Step(f"CO2 dosing {mg:g} mg/L", dCT=n)


def co2_strip(mg):
    """Remove mg/L CO2 (aeration/stripping)."""
    n = mg / M["CO2"]
    return Step(f"CO2 stripping {mg:g} mg/L", dCT=-n)


def strip_co2_to(water, target_mg):
    """Step that strips CO2 until target_mg free CO2 remains."""
    if target_mg >= water.co2_mg:
        return Step("CO2 stripping 0 mg/L")
    def f(dct):
        return replace(water, CT=water.CT - dct).co2_mg - target_mg
    dct = brentq(f, 0, water.species[0] * (1 - 1e-9))
    return co2_strip(dct * M["CO2"] * 1000)


def naoh(mg):
    n = mg / M["NaOH"]
    return Step(f"NaOH {mg:g} mg/L", dAlk=n, dI=n)


def lime(mg):
    """Ca(OH)2 in mg/L."""
    n = mg / M["Ca(OH)2"]
    return Step(f"Lime {mg:g} mg Ca(OH)2/L", dAlk=2 * n, dCa=n, dI=3 * n)


def soda(mg):
    """Na2CO3 in mg/L."""
    n = mg / M["Na2CO3"]
    return Step(f"Soda {mg:g} mg Na2CO3/L", dAlk=2 * n, dCT=n, dI=3 * n)


def caco3(mg, label=None):
    """Dissolve mg/L CaCO3 (marble, limestone, micronised marble)."""
    n = mg / M["CaCO3"]
    return Step(label or f"CaCO3 dissolved {mg:.3g} mg/L",
                dAlk=2 * n, dCT=n, dCa=n, dI=3 * n)


def marble_to_equilibrium():
    """Marble filter run long enough to reach calcite equilibrium (SI = 0).
    This is the LIMIT a real filter approaches."""
    return Step("Marble filter (equilibrium)", to_equilibrium=True)


def acid(mmol):
    """Strong acid in mmol/L (e.g. HCl)."""
    return Step(f"Acid {mmol:g} mmol/L", dAlk=-mmol)


def coagulant(mg_metal, metal="Fe", basicity=0.0):
    """Fe(III) or Al(III) coagulant, dose in mg Me/L.
    Each mol Me consumes 3(1 - basicity) eq alkalinity; the CO2 formed
    stays in the water (C_T unchanged)."""
    n = mg_metal / M[metal]
    return Step(f"{metal} coagulant {mg_metal:g} mg {metal}/L",
                dAlk=-3 * (1 - basicity) * n)


def air_equilibrium(water, p_co2=P_CO2_AIR):
    """Step to equilibrium with air at constant alkalinity
    (the limit of CO2 stripping, slide 32)."""
    K = water.K
    co2_eq = 10 ** K.lk["KH"] * p_co2            # mol/L
    # solve for pH where CO2* = co2_eq at constant Alk
    def g(pH):
        ct = CT_from_pH_alk(pH, water.Alk, K)
        a0 = alphas(pH, K)[0]
        return ct * a0 - co2_eq
    pH_eq = brentq(g, 4.0, 11.0)
    ct = CT_from_pH_alk(pH_eq, water.Alk, K)
    dct = (ct - water.CT) * 1000
    return Step("CO2 stripping to air equilibrium", dCT=dct)


def co2_saturation_mg(temp_C, p_co2=P_CO2_AIR):
    """Dissolved CO2* (mg/L) in equilibrium with a gas phase (Henry's law)."""
    return 10 ** log_K(temp_C)["KH"] * p_co2 * M["CO2"] * 1000


def gas_equilibrium(alk_mmol, p_co2=P_CO2_AIR, temp_C=5.0, I=5e-4, ca_mg=0.0,
                    name="", activity=True):
    """Water with the given alkalinity in equilibrium with a gas phase
    containing CO2 at partial pressure p_co2 (atm). Alkalinity 0 gives pure
    water (e.g. rain); p_co2 = 0.01-0.05 atm is typical soil air."""
    K = _Consts(temp_C, I, activity)
    co2 = 10 ** K.lk["KH"] * p_co2                  # mol/L CO2*
    Alk = alk_mmol / 1000

    def g(pH):
        return CT_from_pH_alk(pH, Alk, K) * alphas(pH, K)[0] - co2
    pH = brentq(g, 2.5, 11.5, xtol=1e-10)
    CT = CT_from_pH_alk(pH, Alk, K)
    return Water(temp_C, I, ca_mg / M["Ca"] / 1000, Alk, CT, name, activity,
                 f"equilibrium with {100 * p_co2:.3g} % CO2")


# ---------------------------------------------------------------------------
# Deffeyes diagram
# ---------------------------------------------------------------------------
def pH_line(pH, ct_mmol, temp_C, I, activity=True):
    """Alkalinity (mmol/L) along a pH line for C_T values (mmol/L)."""
    K = _Consts(temp_C, I, activity)
    return 1000 * alk_from_pH_CT(pH, np.asarray(ct_mmol) / 1000, K)


def saturation_line(ca_mg, ct_mmol, temp_C, I, activity=True, pH_max=10.0):
    """Alkalinity (mmol/L) at calcite saturation for a FIXED calcium.
    Points that would need pH > pH_max are returned as NaN (at very low
    C_T saturation needs a very high pH; not relevant for drinking water)."""
    K = _Consts(temp_C, I, activity)
    ca = ca_mg / M["Ca"] / 1000
    out = []
    for ct in np.atleast_1d(ct_mmol) / 1000:
        def si(pH):
            return np.log10(ca * ct * alphas(pH, K)[2] / K.Ks)
        try:
            p = brentq(si, 4.0, 13.0)
            out.append(1000 * alk_from_pH_CT(p, ct, K) if p <= pH_max else np.nan)
        except ValueError:
            out.append(np.nan)
    return np.array(out)


SAT_COLORS = ["#b5651d", "#7b3f99", "#c62828", "#00838f"]


def deffeyes(ax, temp_C=5.0, I=5e-4, ct_max=2.5, alk_max=2.0,
             pH_values=(5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 10.0),
             ca_sat=None, target=True, activity=True):
    """Draw an empty Deffeyes diagram (alkalinity vs C_T) on ax.

    pH lines are grey; ca_sat is a list of calcium concentrations (mg/L)
    for which the calcite saturation line is drawn (dashed)."""
    ct = np.linspace(1e-4, ct_max, 400)
    top = []                              # labels on the top edge
    for p in pH_values:
        alk = pH_line(p, ct, temp_C, I, activity)
        ok = (alk >= -0.05) & (alk <= alk_max)
        if not ok.any():
            continue
        ax.plot(ct[ok], alk[ok], color="0.72", lw=0.8, zorder=1)
        i = np.where(ok)[0][-1]
        if alk[i] >= alk_max * 0.98:      # line leaves through the top
            top.append((ct[i], p))
        else:                             # line leaves through the right
            ax.text(ct_max * 1.01, alk[i], f"pH {p:g}", fontsize=7.5,
                    color="0.4", ha="left", va="center")
    # spread top labels that would overlap (pH 8 and 8.5 are close)
    top.sort()
    min_gap = 0.03 * ct_max
    xs = [x for x, _ in top]
    for k in range(1, len(xs)):
        xs[k] = max(xs[k], xs[k-1] + min_gap)
    for (x_line, p), x_txt in zip(top, xs):
        ax.annotate(f"pH {p:g}", xy=(x_line, alk_max), xytext=(x_txt, alk_max * 1.015),
                    textcoords="data", fontsize=7.5, color="0.4", rotation=90,
                    ha="center", va="bottom", annotation_clip=False)
    if target:
        _target_zone(ax, temp_C, I, ct_max, alk_max, activity)
    for k, ca in enumerate(ca_sat or []):
        alk = saturation_line(ca, ct, temp_C, I, activity)
        alk[alk > alk_max] = np.nan
        ax.plot(ct, alk, ls="--", lw=1.4, color=SAT_COLORS[k % 4], zorder=2,
                label=f"calcite saturation, Ca {ca:.3g} mg/L")
    if ca_sat:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_xlim(0, ct_max)
    ax.set_ylim(0, alk_max)
    ax.set_xlabel("Total inorganic carbon $C_T$ (mmol/L)")
    ax.set_ylabel("Alkalinity (mmol/L = meq/L)")
    ax.set_title(f"Deffeyes diagram, {temp_C:g} °C", fontsize=11, pad=34)
    return ax


def _target_zone(ax, temp_C, I, ct_max, alk_max, activity=True):
    """Shade pH 8.0-8.5 and alkalinity 0.6-1.0 mmol/L (NV 274/2022)."""
    from matplotlib.patches import Polygon
    (p_lo, p_hi), (a_lo, a_hi) = TARGETS["pH"], TARGETS["Alk"]
    K = _Consts(temp_C, I, activity)
    corners = [(1000 * CT_from_pH_alk(p, a / 1000, K), a)
               for p, a in ((p_lo, a_lo), (p_hi, a_lo), (p_hi, a_hi), (p_lo, a_hi))]
    ax.add_patch(Polygon(corners, closed=True, facecolor="#2e7d32", alpha=0.25,
                         edgecolor="#2e7d32", lw=1.2, zorder=0))
    x0, y0 = corners[3]
    if x0 < ct_max and y0 < alk_max:
        ax.annotate("target zone\npH 8.0–8.5, Alk 0.6–1.0\n(NV 274/2022)",
                    xy=(x0, y0), xytext=(-12, 22), textcoords="offset points",
                    fontsize=7.5, color="#2e7d32", ha="right", va="bottom",
                    arrowprops=dict(arrowstyle="-", color="#2e7d32", lw=0.8))


def plot_path(ax, waters, color="#1f4e9a", show_labels=True):
    """Draw treatment steps as arrows between successive waters."""
    xs = [w.ct_mmol for w in waters]
    ys = [w.alk_mmol for w in waters]
    ax.plot(xs[0], ys[0], "o", color="k", ms=7, zorder=5)
    if show_labels:
        ax.annotate("raw", (xs[0], ys[0]), xytext=(6, -10),
                    textcoords="offset points", fontsize=9)
    for i in range(1, len(waters)):
        if abs(xs[i] - xs[i-1]) < 1e-9 and abs(ys[i] - ys[i-1]) < 1e-9:
            continue
        ax.annotate("", xy=(xs[i], ys[i]), xytext=(xs[i-1], ys[i-1]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2,
                                    shrinkA=0, shrinkB=0), zorder=4)
        ax.plot(xs[i], ys[i], "o", color=color, ms=4, zorder=5)
        if show_labels:
            ax.annotate(str(i), ((xs[i] + xs[i-1]) / 2, (ys[i] + ys[i-1]) / 2),
                        fontsize=8, color=color, xytext=(4, 4),
                        textcoords="offset points", weight="bold")
    ax.plot(xs[-1], ys[-1], "s", color=color, ms=7, zorder=6)


# ---------------------------------------------------------------------------
# Example waters (neutral teaching examples, NOT the semester-project cases)
# ---------------------------------------------------------------------------
def example_waters():
    return {
        "Soft surface water (example)": Water.from_pH_alk(
            pH=6.6, alk_mmol=0.10, ca_mg=2.5, temp_C=5, ec_uS_cm=35,
            name="Soft surface water (example)"),
        "CO2-rich groundwater (example)": Water.from_pH_alk(
            pH=6.4, alk_mmol=0.70, ca_mg=12.0, temp_C=6, ec_uS_cm=120,
            name="CO2-rich groundwater (example)"),
    }
