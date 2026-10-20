"""Oxygen Lab: run with `streamlit run app.py`. Python 3.11+.

Formative teaching and assessment; values are models, not patient advice.
All learner state is session-local. No persistent learner database is used.
"""
from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import math
import random
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

MMHG_PER_KPA = 7.50062
TOPICS = {"cascade": "Oxygen cascade", "transport": "Oxygen transport"}
OBJECTIVES = {
    "cascade": {1: "Label the cascade", 2: "Explain each fall", 3: "Use the alveolar gas equation"},
    "transport": {1: "Pressure, saturation and content", 2: "Dissociation curve and shifts", 3: "Content and delivery"},
}
BLUE, TEAL, ORANGE = "#005EB8", "#007F86", "#C65D13"


def to_mmhg(value, unit):
    return value * MMHG_PER_KPA if unit == "kPa" else value


def from_mmhg(value, unit):
    return value / MMHG_PER_KPA if unit == "kPa" else value


def alveolar_po2(fio2, pb=760.0, paco2=40.0, rq=0.8):
    """Simplified equation in mmHg at 37°C. Never clamp a negative result."""
    if not (0.0 < fio2 <= 1.0 and pb > 47 and paco2 >= 0 and rq > 0):
        raise ValueError("Invalid gas equation inputs")
    return fio2 * (pb - 47.0) - paco2 / rq


def barometric_pressure(altitude_m):
    """Standard atmosphere approximation; not a measured local pressure."""
    if not 0 <= altitude_m <= 5000:
        raise ValueError("Altitude must be 0–5000 m")
    return 760.0 * (1.0 - 2.25577e-5 * altitude_m) ** 5.25588


def saturation(po2, p50=26.6, n=2.7):
    """Hill teaching approximation, returned as a fraction."""
    if po2 < 0 or p50 <= 0 or n <= 0:
        raise ValueError("Invalid curve inputs")
    return po2**n / (po2**n + p50**n)


def oxygen_content(hb_g_dl, sao2_fraction, pao2_mmhg):
    if hb_g_dl < 0 or not 0 <= sao2_fraction <= 1 or pao2_mmhg < 0:
        raise ValueError("Invalid content inputs")
    return 1.34 * hb_g_dl * sao2_fraction + 0.003 * pao2_mmhg


def oxygen_delivery(cao2_ml_dl, co_l_min):
    if cao2_ml_dl < 0 or co_l_min < 0:
        raise ValueError("Invalid delivery inputs")
    return cao2_ml_dl * co_l_min * 10.0


def parse_number(text):
    try:
        value = float(str(text).strip())
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def mcq(qid, objective, prompt, options, correct, explanation):
    return dict(id=qid, objective=objective, type="mcq", prompt=prompt,
                options=options, answer=correct, explanation=explanation)


def numeric(qid, objective, prompt, answer, tolerance, unit, explanation):
    return dict(id=qid, objective=objective, type="number", prompt=prompt,
                answer=answer, tolerance=tolerance, unit=unit, explanation=explanation)


FORMS = "ABCDEFGHIJ"
BANK_VERSION = "advanced-kpa-v3-ten"
DISSOLVED_KPA = 0.0225


def question_bank(topic, seed):
    """120 distinct stems per topic; 40 per objective, pressures in kPa."""
    rng = random.Random(seed)
    questions = []
    prefix = "c2" if topic == "cascade" else "t2"

    def number(obj, n, prompt, answer, explanation, unit="kPa", tolerance=.15):
        questions.append(numeric(f"{prefix}_{obj}_{n:02}", obj, prompt, answer,
                                 tolerance, unit, explanation))

    def single(obj, n, prompt, correct, distractors, explanation):
        questions.append(mcq(f"{prefix}_{obj}_{n:02}", obj, prompt,
                             [correct] + distractors, correct, explanation))

    if topic == "cascade":
        # Objective 1: reconstruct and interpret the cascade under stated conditions.
        pb = rng.choice([68., 72., 76., 80.])
        number(1, 1, f"At altitude PB is {pb:g} kPa. Room air is fully humidified at 37°C (PH₂O 6.3 kPa). Calculate inspired PO₂ before alveolar exchange.",
               .21*(pb-6.3), f"PIO₂ = 0.21 × ({pb:g} − 6.3) = {.21*(pb-6.3):.2f} kPa.")
        fio = rng.choice([.28,.32,.36,.40])
        number(1, 2, f"At constant total pressure, fully humidifying inspired gas at 37°C adds water vapour at 6.3 kPa. FiO₂ is {fio:.2f}. By how much does PO₂ fall from its dry value?",
               fio*6.3, f"The oxygen share of the displaced dry-gas pressure is FiO₂ × PH₂O = {fio:.2f} × 6.3 = {fio*6.3:.2f} kPa.")
        single(1, 3, "A sea-level room-air cascade is labelled 21.3 → 20.0 → 13.4 → 12.7 kPa. A learner labels 13.4 kPa 'arterial blood'. What correction best explains the next point?",
               "13.4 kPa is alveolar; 12.7 kPa is arterial after venous admixture and V/Q inequality",
               ["13.4 kPa is mixed venous; 12.7 kPa is mitochondrial", "13.4 kPa is humidified inspired gas; 12.7 kPa is dry air", "Both points are arterial; the fall is caused by tissue extraction before arterial sampling"],
               "The alveolar-to-arterial step precedes systemic tissue extraction. State rest, sea level, room air and normal gas exchange.")
        single(1, 4, "A graph places mixed venous PO₂ 5.3 kPa between arterial blood and tissue PO₂ as a compulsory serial diffusion compartment. How should it be redrawn?",
               "Show arterial → tissue → mitochondria, with mixed venous blood on a separate return pathway",
               ["Put mixed venous blood before the alveoli on the inspired gas axis", "Replace all tissue points with the mixed venous value", "Keep it because every oxygen molecule must pass through venous blood before reaching cells"],
               "Mixed venous blood is the flow-weighted return after extraction, not a mandatory intermediate for diffusion from arteries into cells.")
        fio = rng.choice([.25,.30,.35,.40])
        pb = rng.choice([75.,80.,85.])
        number(1, 5, f"Dry inspired PO₂ is {fio*pb:.2f} kPa at PB {pb:g} kPa. Determine FiO₂ as a percentage. Do not apply humidification to this dry sample.",
               fio*100, f"FiO₂ = dry PO₂/PB = {fio*pb:.2f}/{pb:g} = {fio:.2f}, or {fio*100:g}%.", "%", .2)
        pb = rng.choice([65.,70.,75.])
        target = rng.choice([18.,20.,22.])
        number(1, 6, f"At PB {pb:g} kPa, what inspired oxygen percentage produces humidified PIO₂ {target:g} kPa? Assume PH₂O 6.3 kPa. Give a percentage, not a fraction.",
               100*target/(pb-6.3), f"FiO₂ = PIO₂/(PB−PH₂O); multiply {target:g}/({pb:g}−6.3) by 100.", "%", .3)
        single(1, 7, "After mitochondrial oxygen consumption stops while arterial supply initially remains unchanged, what happens to the local diffusion gradient?",
               "Local PO₂ tends to rise, reducing the gradient driving oxygen into that region",
               ["Mitochondrial PO₂ must fall to zero", "Arterial PO₂ instantly becomes zero", "The gradient increases because mitochondria produce oxygen"],
               "Consumption maintains a low downstream PO₂. Reducing the sink reduces the diffusion gradient, subject to ongoing flow and metabolism.")
        single(1, 8, "At the same arterial PO₂, two organs have different capillary density, flow and oxygen consumption. Which statement about tissue PO₂ is justified?",
               "Their tissue PO₂ can differ; one fixed tissue point cannot predict both organs",
               ["Both tissues must have exactly 2.7 kPa", "Both must equal arterial PO₂", "Mixed venous PO₂ uniquely determines each organ's mitochondrial PO₂"],
               "Delivery, extraction and diffusion distances vary locally. Tissue and mitochondrial points are illustrative, not universal normal values.")
        pb = rng.choice([70.,78.,86.,94.])
        pio = .21*(pb-6.3)
        number(1, 9, f"An oxygen cascade begins with humidified inspired PO₂ {pio:.3f} kPa on room air (FiO₂ 0.21). At 37°C, PH₂O is 6.3 kPa. Reconstruct the barometric pressure.",
               pb, f"PB = PIO₂/0.21 + 6.3 = {pb:.2f} kPa.", tolerance=.2)
        pb = rng.choice([65.,70.,75.])
        number(1, 10, f"At sea level PB is 101.3 kPa and room-air dry PO₂ is 0.21 × PB. At a second site PB is {pb:g} kPa with the same FiO₂. By what percentage has dry atmospheric PO₂ fallen?",
               (101.3-pb)/101.3*100, "With FiO₂ fixed, dry oxygen pressure is proportional to PB. Percentage fall = (101.3 − new PB)/101.3 × 100.", "%", .3)
        single(1, 11, "A learner draws only 20.0 kPa inspired gas and 12.5 kPa arterial blood, then calls their difference the A–a gradient. What is missing?",
               "The alveolar PO₂ must first be estimated; inspired-to-arterial difference is not A–a",
               ["The mitochondrial PO₂ must be subtracted from inspired PO₂", "The mixed venous saturation must replace inspired PO₂", "Nothing: inspired and alveolar PO₂ are equal at steady state"],
               "A–a means alveolar minus arterial. Alveolar exchange creates a substantial inspired-to-alveolar fall even in healthy lungs.")
        single(1, 12, "A resting cascade is quoted without FiO₂, PB or PaCO₂. Why is identifying an 'abnormal' alveolar point from it unreliable?",
               "The expected alveolar value changes with inspired oxygen, barometric pressure and ventilation",
               ["Normal PAO₂ is always 13.3 kPa irrespective of conditions", "Only haemoglobin concentration determines PAO₂", "Arterial saturation removes the need to state any conditions"],
               "Reference values require conditions. A value appropriate at altitude or during oxygen administration may differ greatly from sea-level room-air values.")
        water = rng.choice([5.5,6.,6.3])
        fio = .35
        number(1, 13, f"PB is 100 kPa and FiO₂ 0.35. Measured humidified inspired PO₂ is {fio*(100-water):.3f} kPa. Infer the water vapour partial pressure, assuming an ideal gas mixture and no oxygen uptake yet.",
               water, "Rearrange PIO₂ = FiO₂(PB−PH₂O): PH₂O = PB − PIO₂/FiO₂.")
        pb = rng.choice([70.,75.,80.])
        number(1, 14, f"At PB {pb:g} kPa, a dry room-air sample has FiO₂ 0.21. What percentage of its dry oxygen partial pressure is lost on full humidification at 37°C (PH₂O 6.3 kPa)?",
               6.3/pb*100, "Relative pressure loss = (FiO₂×6.3)/(FiO₂×PB) × 100 = 6.3/PB × 100. The oxygen fraction cancels.", "%", .2)
        single(1, 15, "A patient has normal alveolar and arterial PO₂ but much lower mixed venous PO₂ than before. Which cascade interpretation is most defensible?",
               "Increased extraction or reduced oxygen delivery may explain the downstream fall despite preserved lung oxygen tension",
               ["The inspired oxygen fraction must have increased", "A large A–a gradient is proved", "Normal arterial PO₂ excludes any problem with tissue oxygen supply"],
               "Normal arterial tension does not guarantee adequate content or flow. Mixed venous changes reflect the balance of delivery and consumption.")
        single(1, 16, "A graph shows alveolar PO₂ 13.3 kPa, arterial PO₂ 12.6 kPa and mitochondria 0.3 kPa. Which additional statement is essential?",
               "The mitochondrial point is illustrative and spatially variable, not a fixed value calculated from the A–a gradient",
               ["All mitochondria must remain at exactly 0.3 kPa", "The A–a gradient determines mitochondrial consumption directly", "Mitochondrial PO₂ equals oxygen content expressed in kPa"],
               "The gas equation and A–a gradient concern lung gas exchange; they do not determine a unique intracellular PO₂.")

        # Objective 2: interpret mechanisms rather than recall definitions.
        cases = [
            ("On room air at PB 101.3 kPa, PH₂O 6.3 kPa and R 0.8, PaCO₂ is 8.0 kPa and PaO₂ is 9.3 kPa. What best explains the hypoxaemia?",
             "Predominantly hypoventilation, with an A–a gradient of approximately 0.7 kPa",
             ["A large A–a gradient proves a major shunt", "Low haemoglobin directly lowers the calculated PAO₂", "The A–a gradient is 10.7 kPa"],
             "PAO₂ = 19.95−8/0.8 = 9.95 kPa; A–a = 0.65 kPa. Pure hypoventilation need not widen A–a."),
            ("A healthy adult at altitude has PB 70 kPa, FiO₂ 0.21, PH₂O 6.3 kPa, R 0.8, PaCO₂ 3.2 kPa and PaO₂ 8.7 kPa. Which interpretation fits?",
             "Low inspired oxygen pressure with compensatory hyperventilation and a small A–a gradient",
             ["A very large A–a gradient is demonstrated", "Ambient FiO₂ has necessarily fallen below 0.21", "Hyperventilation caused PAO₂ to fall"],
             "PAO₂ = 0.21×63.7−3.2/0.8 = 9.38 kPa; A–a ≈0.68 kPa. Low PB lowers PIO₂, while lower PaCO₂ partly offsets this."),
            ("Two lung regions have different alveolar PO₂. Why can their arithmetic mean fail to predict systemic arterial PO₂?",
             "Blood mixes by oxygen content and flow; the content–PO₂ relation is nonlinear",
             ["Partial pressures always add when equal blood volumes mix", "The lowest regional PO₂ must equal systemic arterial PO₂", "Haemoglobin concentration has no effect on regional oxygen content"],
             "Calculate regional end-capillary content, weight it by perfusion, mix contents, then relate the result to PO₂ under stated curve conditions."),
            ("A perfused lung unit has V/Q approaching zero. Why does raising inspired oxygen correct its contribution poorly?",
             "Little or no fresh gas reaches that unit, so its blood approaches mixed venous composition",
             ["Its capillary blood has no haemoglobin", "Its alveolar PO₂ must exceed inspired PO₂", "Increasing FiO₂ cannot change any ventilated lung unit"],
             "At V/Q=0 the unit is a true shunt. Oxygen reaches ventilated units, while shunted blood bypasses alveolar oxygenation."),
            ("Increased alveolar ventilation raises PAO₂ at unchanged FiO₂, PB and R. Which link explains this?",
             "PaCO₂ falls as CO₂ elimination increases, reducing the CO₂/R subtraction",
             ["PH₂O necessarily falls to zero", "FiO₂ rises whenever ventilation increases", "Oxygen consumption becomes zero"],
             "At steady state with CO₂ production fixed, PaCO₂ varies inversely with alveolar ventilation. The gas equation links this to PAO₂."),
            ("An adult has low PaO₂ and a widened A–a gradient on room air. Which mechanism alone would be least likely to account for the widened gradient?",
             "Pure reduction in alveolar ventilation with otherwise normal gas exchange",
             ["V/Q inequality", "Right-to-left shunt", "Oxygen diffusion limitation"],
             "Pure hypoventilation lowers alveolar and arterial PO₂ together. The other mechanisms impair alveolar-to-arterial transfer."),
            ("Why may oxygen diffusion limitation emerge during exercise at altitude even if equilibration was adequate at sea-level rest?",
             "The oxygen driving gradient is lower and pulmonary capillary transit time is shorter",
             ["Ambient FiO₂ necessarily becomes zero", "Exercise stops pulmonary perfusion", "Water vapour pressure rises in direct proportion to altitude"],
             "Reduced alveolar PO₂ and shortened transit time reduce the margin for equilibration, especially when diffusion capacity is impaired."),
            ("A proposed treatment doubles alveolar ventilation but oxygen consumption and CO₂ production are unchanged. Which variable is most directly expected to approximately halve at steady state?",
             "Alveolar and arterial CO₂ partial pressure",
             ["Arterial oxygen content", "Barometric pressure", "Water vapour pressure at 37°C"],
             "PaCO₂ is proportional to CO₂ production divided by alveolar ventilation. PAO₂ does not simply double because PIO₂ is finite."),
            ("Why does a high-V/Q unit often fail to fully compensate for a low-V/Q unit when blood mixes?",
             "Haemoglobin in the high-V/Q unit is already near its saturation plateau, limiting extra bound oxygen",
             ["High-V/Q units cannot receive inspired oxygen", "Blood PO₂ mixes by its arithmetic mean regardless of content", "Haemoglobin unloading is abolished in low-V/Q units"],
             "The plateau limits content gain in well-oxygenated blood, while low-V/Q blood can lose considerable content on the steep curve."),
            ("A learner uses the simplified gas equation to calculate a specific mitochondrial PO₂ from PaCO₂ and FiO₂. What is the main error?",
             "The equation estimates alveolar PO₂ and does not model tissue flow, extraction or diffusion",
             ["The equation directly estimates only haemoglobin concentration", "The equation is valid only if tissue PO₂ is zero", "Alveolar and mitochondrial PO₂ are normally identical"],
             "Lung gas balance provides alveolar pressure. An assumed A–a gradient and a separate tissue model would be needed to go further."),
            ("At rest a person develops a local rise in metabolic oxygen demand with unchanged initial local delivery. What initially supports increased uptake?",
             "Greater extraction lowers local venous/tissue PO₂ and steepens the diffusion gradient",
             ["Mitochondria increase inspired FiO₂", "Local venous PO₂ must rise", "Haemoglobin-bound oxygen directly creates gas partial pressure without dissociation"],
             "A stronger metabolic sink can increase extraction and lower venous content and PO₂. Further compensation may include increased regional flow."),
            ("A normal PaCO₂ is present despite a substantial right-to-left shunt. Why does this not exclude the shunt?",
             "Ventilated regions can increase CO₂ elimination while oxygen content compensation is constrained by the saturation plateau",
             ["Shunted blood contains no CO₂", "CO₂ cannot cross the alveolar membrane", "Normal PaCO₂ proves every lung unit is ventilated"],
             "CO₂ elimination and oxygen loading respond differently to ventilation and blood mixing. A normal PaCO₂ does not establish normal oxygen transfer."),
            ("Two otherwise identical gas-exchange measurements are made on room air and high FiO₂. Why should a single fixed normal A–a cutoff not be applied indiscriminately?",
             "The expected A–a gradient depends on inspired oxygen and other conditions, including age",
             ["The gradient is always exactly 0.7 kPa", "A–a is independent of inspired oxygen by definition", "The gradient becomes an oxygen content measurement on high FiO₂"],
             "A–a is a pressure difference interpreted in context. FiO₂, age and gas exchange affect the expected value."),
            ("A specimen is obtained immediately after a large change in ventilation. Why may a steady-state gas-equation interpretation be misleading?",
             "Gas stores and blood gases may not yet have equilibrated to the new conditions",
             ["Dalton's law stops applying during ventilation", "The respiratory quotient must instantly become zero", "Water vapour ceases to contribute to pressure"],
             "The standard interpretation assumes steady state, negligible inspired CO₂ and an appropriate R. Rapid transitions can violate these assumptions."),
            ("PaCO₂ is unchanged while R falls from 1.0 to 0.8 in an otherwise fixed simplified-equation example. Why does predicted PAO₂ fall?",
             "For the same CO₂ production, the lower R implies more oxygen consumption relative to CO₂ output",
             ["Lower R means water vapour pressure rises", "Lower R means FiO₂ must fall", "Oxygen uptake is multiplied by R rather than divided by it"],
             "R = VCO₂/VO₂. At fixed CO₂ term, dividing by a smaller R gives a larger oxygen subtraction."),
            ("Why is the simplified alveolar gas equation less accurate at high FiO₂ when R differs from 1?",
             "It omits a correction for unequal inspired and expired gas volumes associated with gas exchange",
             ["Oxygen has no partial pressure at high concentration", "Water vapour must be omitted at high FiO₂", "Arterial CO₂ can no longer diffuse through blood"],
             "A fuller steady-state expression uses PaCO₂[FiO₂+(1−FiO₂)/R]. The difference from PaCO₂/R increases as FiO₂ increases when R<1."),
        ]
        for n, (prompt, correct, distractors, explanation) in enumerate(cases, 1):
            single(2, n, prompt, correct, distractors, explanation)

        # Objective 3: multi-step and rearranged gas-equation calculations.
        co2 = rng.choice([6.4,7.2,8.0])
        pa = rng.choice([7.,7.5,8.])
        aa = .21*95-co2/.8-pa
        number(3, 1, f"On room air PB 101.3 kPa, PH₂O 6.3 kPa, R 0.8, PaCO₂ {co2:g} kPa and PaO₂ {pa:g} kPa. Calculate the A–a gradient using the simplified equation.",
               aa, f"PAO₂ = 0.21×95−{co2:g}/0.8; A–a = PAO₂−{pa:g} = {aa:.2f} kPa.")
        target = rng.choice([10.,11.,12.])
        number(3, 2, f"At PB 101.3 kPa on FiO₂ 0.21, PH₂O 6.3 kPa and R 0.8, what PaCO₂ gives PAO₂ {target:g} kPa? Rearrange the simplified equation.",
               (.21*95-target)*.8, f"PaCO₂ = R[FiO₂(PB−PH₂O)−PAO₂] = 0.8(19.95−{target:g}).")
        target = rng.choice([12.,14.,16.])
        pb = rng.choice([65.,70.,75.])
        number(3, 3, f"At PB {pb:g} kPa, PH₂O 6.3 kPa, PaCO₂ 4.0 kPa and R 0.8, estimate the FiO₂ percentage needed for PAO₂ {target:g} kPa using the simplified equation.",
               100*(target+5)/(pb-6.3), "FiO₂ = (target PAO₂ + PaCO₂/R)/(PB−PH₂O). Report ×100 as a percentage; this is a physiological estimate, not a treatment prescription.", "%", .3)
        co2 = rng.choice([5.6,6.4,7.2])
        factor = rng.choice([1.25,1.5,2.])
        number(3, 4, f"At steady state PaCO₂ is {co2:g} kPa. Alveolar ventilation increases by a factor of {factor:g} with CO₂ production fixed. At unchanged FiO₂/PB and R 0.8, by how much does simplified PAO₂ rise?",
               co2*(1-1/factor)/.8, "New PaCO₂ = old/factor. PAO₂ rise = (old PaCO₂−new PaCO₂)/R.")
        pb = rng.choice([70.,75.,80.])
        co2 = rng.choice([3.2,4.,4.8])
        number(3, 5, f"At altitude PB {pb:g} kPa, PH₂O 6.3 kPa, FiO₂ 0.21, PaCO₂ {co2:g} kPa, R 0.8 and an assumed A–a gradient 0.8 kPa, estimate arterial PO₂.",
               .21*(pb-6.3)-co2/.8-.8, "First calculate PAO₂, then subtract the stated A–a gradient. The gradient is an assumption, not produced by the equation.")
        delta = rng.choice([10.,15.,20.])
        number(3, 6, f"PB falls by {delta:g} kPa while FiO₂ stays 0.21. With R 0.8, what fall in PaCO₂ would exactly offset the pressure-related fall in PAO₂ in the simplified equation? Assume PH₂O remains 6.3 kPa.",
               .8*.21*delta, "For no PAO₂ change, FiO₂×fall in PB = fall in PaCO₂/R. Required PaCO₂ fall = R×FiO₂×PB fall.")
        fio = rng.choice([.5,.6,.8,1.])
        co2 = rng.choice([4.,5.,6.])
        difference = co2*fio*(1/.8-1)
        number(3, 7, f"At FiO₂ {fio:g}, PaCO₂ {co2:g} kPa and R 0.8, the fuller CO₂ subtraction is PaCO₂[FiO₂+(1−FiO₂)/R]. By how much does the fuller PAO₂ exceed the simplified PAO₂? PB and PH₂O are unchanged.",
               difference, f"Difference = PaCO₂×FiO₂×(1/R−1) = {co2:g}×{fio:g}×0.25 = {difference:.2f} kPa.")
        co2 = rng.choice([4.8,5.6,6.4])
        number(3, 8, f"With PaCO₂ {co2:g} kPa and all inspired conditions fixed, by how much is simplified PAO₂ lower at R 0.8 than at R 1.0? Give a positive difference.",
               co2*(1/.8-1), "Difference = PaCO₂(1/0.8−1/1.0). The smaller R causes a larger subtraction.")
        pa = rng.choice([7.5,8.,8.5])
        aa = rng.choice([.5,1.,1.5])
        number(3, 9, f"At a site of unknown PB, FiO₂ is 0.21, PH₂O 6.3 kPa, PaCO₂ 3.2 kPa, PaO₂ {pa:g} kPa and the assumed A–a gradient {aa:g} kPa. Estimate PB with the simplified equation.",
               (pa+aa+4)/.21+6.3, "PAO₂ = PaO₂ + A–a. Then PB = (PAO₂+PaCO₂/R)/FiO₂ + PH₂O, with R=0.8.", tolerance=.3)
        # Explicit R in this item's prompt is required to make the rearrangement determinate.
        questions[-1]["prompt"] += " Assume R 0.8."
        delta = rng.choice([.04,.06,.08,.10])
        number(3, 10, f"At PB 101.3 kPa and PH₂O 6.3 kPa, FiO₂ increases by {delta:.2f}. With PaCO₂ and R fixed, calculate the rise in simplified PAO₂.",
               delta*95, "ΔPAO₂ = ΔFiO₂×(PB−PH₂O) when the CO₂ term is unchanged.")
        target = rng.choice([13.,15.,17.])
        number(3, 11, f"At PB 101.3 kPa, PH₂O 6.3 kPa, PaCO₂ 6.4 kPa and R 0.8, an assumed A–a gradient is 2.0 kPa. Estimate FiO₂ as a percentage to obtain PaO₂ {target:g} kPa using the simplified equation.",
               100*(target+2+8)/95, "Required PAO₂ = target PaO₂+2; FiO₂ = (required PAO₂+6.4/0.8)/95. Give ×100 percent.", "%", .3)
        fio = rng.choice([.3,.4,.5])
        co2 = rng.choice([6.4,7.2,8.])
        number(3, 12, f"At sea level PB 101.3 kPa, FiO₂ {fio:g}, PH₂O 6.3 kPa and R 0.8, PaCO₂ rises from 4.8 to {co2:g} kPa. By how much must FiO₂ increase, in percentage points, to keep simplified PAO₂ unchanged?",
               100*(co2-4.8)/(.8*95), "Additional inspired pressure must equal the increased CO₂/R term: ΔFiO₂ = ΔPaCO₂/[R(PB−PH₂O)].", "percentage points", .2)
        over = rng.choice([1.5,2.,2.5])
        number(3, 13, f"A trainee calculates PAO₂ using PB rather than PB−PH₂O, omitting humidification. PH₂O is 6.3 kPa. Their result is {over:g} kPa too high. What FiO₂ percentage were they using?",
               over/6.3*100, "Omitting water vapour overestimates PAO₂ by FiO₂×PH₂O, so FiO₂ = error/6.3.", "%", .3)
        co2 = rng.choice([4.,4.8,5.6])
        number(3, 14, f"At FiO₂ 0.21, PB 101.3 kPa and PH₂O 6.3 kPa, the measured PAO₂ is {(.21*95-co2/.85):.3f} kPa and PaCO₂ is {co2:g} kPa. Infer R using the simplified equation.",
               .85, "R = PaCO₂/[FiO₂(PB−PH₂O)−PAO₂]. Use the given rounded PAO₂.", "ratio", .02)
        pa = rng.choice([7.,8.,9.])
        number(3, 15, f"On FiO₂ 0.30 at PB 101.3 kPa, PH₂O 6.3 kPa and R 0.8, PaCO₂ is 5.2 kPa and PaO₂ is {pa:g} kPa. Calculate the alveolar-to-arterial oxygen ratio a/A (PaO₂/PAO₂), not the pressure difference.",
               pa/(.3*95-5.2/.8), "PAO₂ = 0.30×95−5.2/0.8 = 22 kPa; a/A = PaO₂/22.", "ratio", .015)
        pb = rng.choice([65.,70.,75.])
        number(3, 16, f"Compare sea level (PB 101.3 kPa, PaCO₂ 5.3 kPa) with altitude (PB {pb:g} kPa, PaCO₂ 3.2 kPa). FiO₂ 0.21, PH₂O 6.3 kPa and R 0.8 at both. Calculate the net fall in simplified PAO₂.",
               .21*(101.3-pb)-(5.3-3.2)/.8, "Net fall = 0.21×fall in PB − fall in PaCO₂/0.8. Hyperventilation offsets part of the pressure-related loss.")
    elif topic == "transport":
        # Objective 1: pressure, saturation and content in less obvious settings.
        hb = rng.choice([75.,85.,95.,105.])
        sa = rng.choice([.94,.96,.98])
        pa = rng.choice([10.,11.,12.])
        content = 1.34*(hb/10)*sa+.0225*pa
        number(1, 1, f"Hb is {hb:g} g/L, SaO₂ {sa*100:g}% and PaO₂ {pa:g} kPa. Calculate CaO₂ in mL O₂/dL using 1.34 mL/g and dissolved coefficient 0.0225 mL/dL/kPa.",
               content, f"Convert Hb to {hb/10:g} g/dL, use saturation as a fraction: 1.34×{hb/10:g}×{sa:g}+0.0225×{pa:g} = {content:.2f}.", "mL O₂/dL", .15)
        hb = rng.choice([8.,10.,12.])
        pa = rng.choice([40.,50.,60.])
        total = 1.34*hb+.0225*pa
        number(1, 2, f"Hb {hb:g} g/dL, SaO₂ 100%, PaO₂ {pa:g} kPa. Using Hb capacity 1.34 and dissolved coefficient 0.0225, what percentage of total CaO₂ is dissolved?",
               .0225*pa/total*100, "Dissolved fraction = 0.0225×PaO₂ / (1.34×Hb+0.0225×PaO₂). Multiply by 100.", "%", .2)
        single(1, 3, "In carbon monoxide exposure, PaO₂ can be normal while oxygen transport is severely impaired. Which explanation best links pressure and content?",
               "Dissolved oxygen tension may be preserved while CO occupies Hb sites and impairs unloading from remaining oxyhaemoglobin",
               ["PaO₂ directly counts all Hb-bound oxygen molecules", "Normal PaO₂ proves that all Hb binding sites contain oxygen", "CO only affects the dissolved oxygen component"],
               "PaO₂ reflects dissolved oxygen. CO reduces available oxygen-carrying capacity and increases oxygen affinity at remaining sites; conventional pulse oximetry may mislead.")
        single(1, 4, "Severe anaemia is present with PaO₂ 13 kPa and SaO₂ 98%. Which statement best describes what an arterial blood gas PO₂ alone misses?",
               "It does not quantify the haemoglobin available to carry oxygen or the resulting oxygen content",
               ["It must be falsely low in any anaemic patient", "It is a direct measurement of whole-body oxygen delivery", "It measures the fraction of oxygen extracted by each organ"],
               "Normal tension and saturation can coexist with low content. Delivery additionally depends on cardiac output and its distribution.")
        hb = rng.choice([90.,100.,110.])
        pa = rng.choice([11.,12.,13.])
        sa = rng.choice([.90,.94,.98])
        ca = 1.34*(hb/10)*sa+.0225*pa
        number(1, 5, f"CaO₂ is {ca:.3f} mL/dL at Hb {hb:g} g/L and PaO₂ {pa:g} kPa. Infer SaO₂ as a percentage, using capacity 1.34 and dissolved coefficient 0.0225.",
               sa*100, "SaO₂ fraction = (CaO₂−0.0225×PaO₂)/(1.34×Hb in g/dL); then multiply by 100.", "%", .2)
        hb = rng.choice([8.,9.,10.,11.])
        sa = .96
        ca = 1.34*hb*sa+.0225*12
        number(1, 6, f"CaO₂ is {ca:.3f} mL/dL, SaO₂ 96% and PaO₂ 12 kPa. Infer Hb in g/L using capacity 1.34 and dissolved coefficient 0.0225.",
               hb*10, "Hb in g/dL = (CaO₂−0.0225×12)/(1.34×0.96). Multiply by 10 for g/L.", "g/L", .3)
        single(1, 7, "Two patients have identical PaO₂ and SaO₂; patient A has Hb 150 g/L and patient B has Hb 75 g/L. Which component is unchanged between them?",
               "Dissolved oxygen content, assuming the same temperature and solubility coefficient",
               ["Hb-bound oxygen content", "Total arterial oxygen content", "Oxygen delivery regardless of cardiac output"],
               "Dissolved content depends on PO₂ and solubility, not Hb concentration. The Hb-bound component approximately halves.")
        single(1, 8, "Why can Hb unload a substantial amount of oxygen while the dissolved compartment remains a small fraction of total blood oxygen?",
               "Bound oxygen dissociates as dissolved oxygen diffuses into tissues, replenishing the dissolved pool",
               ["Bound oxygen produces a partial pressure without ever dissolving", "Haemoglobin diffuses intact through the normal capillary wall", "Dissolved oxygen is unrelated to diffusion into cells"],
               "Only dissolved oxygen directly determines tension and diffuses; equilibrium with the large bound reservoir sustains transfer.")
        hb = rng.choice([12.,14.,16.])
        drop = rng.choice([.08,.10,.12])
        number(1, 9, f"Hb {hb:g} g/dL is unchanged. SaO₂ falls from 98% by {drop*100:g} percentage points. Neglect dissolved oxygen. Calculate the fall in Hb-bound oxygen content, using capacity 1.34.",
               1.34*hb*drop, "Δbound content = 1.34×Hb×absolute change in saturation fraction. A percentage-point fall is not a relative percentage fall.", "mL O₂/dL", .08)
        rise = rng.choice([20.,30.,40.])
        number(1, 10, f"PaO₂ rises by {rise:g} kPa while Hb and SaO₂ are held constant. How much does total CaO₂ rise using dissolved coefficient 0.0225 mL/dL/kPa?",
               .0225*rise, "The bound component is fixed, so ΔCaO₂ = 0.0225×ΔPaO₂.", "mL O₂/dL", .03)
        single(1, 11, "A report gives functional oxygen saturation 98% in significant methaemoglobinaemia. Why can using total Hb × 0.98 overestimate oxygen content?",
               "Functional saturation excludes dysfunctional Hb from its denominator; total Hb is not all available to carry oxygen",
               ["Functional saturation includes every dysfunctional Hb molecule as deoxygenated Hb", "Methaemoglobin carries more oxygen than normal Hb", "PaO₂ must always be zero in methaemoglobinaemia"],
               "Functional saturation is oxyHb/(oxyHb+deoxyHb). Fractional oxyHb includes dysHb in the total denominator and is needed with total Hb for this calculation.")
        single(1, 12, "Which measurement combination is most useful for calculating content when dyshemoglobins may be present?",
               "Total Hb, co-oximetry fractional oxyhaemoglobin and measured PaO₂, with the appropriate solubility coefficient",
               ["PaO₂ alone", "SpO₂ alone without Hb", "A standard dissociation curve alone without Hb or co-oximetry"],
               "A standard curve or calculated saturation may not reflect dyshemoglobin fractions. Pressure and saturation do not substitute for Hb concentration.")
        hb = rng.choice([10.,12.,14.])
        fraction = rng.choice([.70,.75,.80])
        number(1, 13, f"For a hypothetical CO-exposed sample, total Hb is {hb:g} g/dL and co-oximetry fractional oxyHb is {fraction*100:g}% of total Hb. PaO₂ is 13 kPa. Calculate CaO₂ using 1.34 and dissolved coefficient 0.0225; do not use functional saturation.",
               1.34*hb*fraction+.0225*13, "Bound content = capacity×total Hb×fractional oxyHb. Add dissolved oxygen. This is content, not a prediction of effective tissue unloading.", "mL O₂/dL", .12)
        hb = rng.choice([10.,12.,14.])
        number(1, 14, f"Hb is {hb:g} g/dL and SaO₂ 100%. At what theoretical PaO₂ would dissolved oxygen equal Hb-bound oxygen? Use capacity 1.34 and dissolved coefficient 0.0225. This is a mathematical comparison, not a physiological target.",
               1.34*hb/.0225, "Set 0.0225×PaO₂ = 1.34×Hb and solve. The very high value illustrates how dominant Hb carriage normally is.", tolerance=1.)
        single(1, 15, "Patient A has PaO₂ 13 kPa, Hb 8 g/dL and SaO₂ 98%; B has PaO₂ 10 kPa, Hb 15 g/dL and SaO₂ 95%. Who has higher CaO₂ using the same capacity and solubility constants?",
               "B, because the larger Hb-bound component more than offsets the lower PaO₂ and saturation",
               ["A, because PO₂ alone ranks oxygen content", "They must be identical because both saturations exceed 90%", "It cannot be compared even with all these values"],
               "CaO₂ ≈10.80 mL/dL in A and 19.32 in B using 1.34 and 0.0225. A higher pressure does not imply greater content.")
        single(1, 16, "An arterial saturation improves from 95% to 100%, but Hb simultaneously halves at unchanged PaO₂. What is the most likely change in CaO₂?",
               "A substantial fall, because the small saturation gain cannot offset loss of half the Hb",
               ["It necessarily doubles", "It is unchanged because 100% saturation guarantees normal content", "It rises in direct proportion to PaO₂ despite fixed PaO₂"],
               "The bound component changes by a factor 0.5×1.00/0.95 ≈0.526. The dissolved component is unchanged.")

        # Objective 2: curve interpretation, competing effects and supplied Hill-model problems.
        cases = [
            ("During exercise, tissue temperature and CO₂ rise and pH falls. At a fixed local PO₂, how does the dissociation curve facilitate extraction?",
             "A right shift raises P₅₀ and lowers Hb saturation at that PO₂",
             ["A left shift raises P₅₀ and increases saturation", "A right shift lowers P₅₀ and traps oxygen on Hb", "A shift changes Hb concentration rather than affinity"],
             "Heat, CO₂ and H⁺ reduce oxygen affinity. At fixed tissue PO₂, lower saturation represents more oxygen unloaded from a given arterial load."),
            ("At very low alveolar PO₂, why can a right shift have competing effects on oxygen supply?",
             "It can aid tissue unloading but impair pulmonary loading on the steep portion of the curve",
             ["It always increases arterial saturation and tissue affinity simultaneously", "It always leaves arterial saturation unchanged", "It makes Hb-bound oxygen independent of PO₂"],
             "Whole-body delivery depends on arterial loading and flow as well as unloading. A right shift is not automatically beneficial in severe hypoxia."),
            ("A left-shifted curve reaches 50% saturation at 2.8 kPa; the original reaches 50% at 3.6 kPa. At tissue PO₂ 4 kPa, which inference is best?",
             "The left-shifted Hb retains more oxygen at that PO₂, potentially reducing unloading",
             ["The left-shifted Hb necessarily has less oxygen affinity", "P₅₀ has increased", "Both curves must have identical saturation at every PO₂ above P₅₀"],
             "A lower P₅₀ indicates higher affinity. Compare curves at the same PO₂, while keeping the incoming arterial oxygen load in mind."),
            ("A plot uses oxygen content on the y-axis rather than saturation. Hb halves while affinity, PO₂ and dissolved coefficient remain unchanged. What happens to the plotted curve?",
             "Its Hb-bound content is approximately halved vertically, without an intrinsic change in P₅₀",
             ["It must shift right because anaemia always raises P₅₀ immediately", "Its maximum content is unchanged", "Saturation at each PO₂ must halve"],
             "Content depends on Hb concentration; saturation describes occupancy of available sites. Anaemia alone need not change affinity."),
            ("A rise in H⁺ decreases oxygen affinity. Which paired description correctly distinguishes the Bohr and Haldane effects?",
             "Bohr: CO₂/H⁺ alter oxygen binding; Haldane: oxygenation alters blood CO₂ carrying capacity",
             ["Bohr: oxygenation alters CO₂ capacity; Haldane: H⁺ alters oxygen affinity", "Both names describe only dissolved oxygen solubility", "Both effects mean Hb concentration doubles"],
             "Bohr concerns the effect of CO₂/H⁺ on oxygen affinity. Haldane concerns the effect of oxygenation on CO₂ carriage."),
            ("Why does fetal haemoglobin support placental oxygen transfer at relatively low PO₂?",
             "Its lower interaction with 2,3-DPG produces higher oxygen affinity and a lower P₅₀",
             ["It has no cooperative oxygen binding", "It requires a higher PO₂ than adult Hb for 50% saturation", "It carries oxygen only in dissolved form"],
             "Reduced 2,3-DPG binding contributes to HbF's left shift. Maternal/fetal gas gradients and acid–base effects also matter."),
            ("A recently stored red-cell sample is depleted of 2,3-DPG. Under otherwise identical conditions, what curve change is expected?",
             "A lower P₅₀ with higher affinity and potentially less unloading at fixed tissue PO₂",
             ["A higher P₅₀ with lower affinity", "No Hb-bound oxygen at any PO₂", "A mandatory doubling of Hb concentration"],
             "2,3-DPG stabilises the lower-affinity deoxygenated state. Depletion favours a left shift; in-vivo recovery and other factors modify the effect."),
            ("A learner concludes that a P₅₀ measured during acidosis proves an intrinsic Hb variant. What should be checked first?",
             "Whether pH, temperature, CO₂ and 2,3-DPG conditions match the reference conditions",
             ["Only the patient's height", "Whether PaO₂ is expressed as saturation", "Whether all normal Hb has a P₅₀ of zero"],
             "P₅₀ is condition-dependent. Standardised conditions are needed before inferring a change intrinsic to Hb."),
            ("If Hb had a simple hyperbolic rather than cooperative sigmoid oxygen-binding curve with the same P₅₀, what feature would be lost?",
             "The characteristic steep middle region coupled to a high-PO₂ loading plateau produced by cooperativity",
             ["All dissolved oxygen in plasma", "Every oxygen binding site", "Any dependence of saturation on PO₂"],
             "Cooperativity changes the shape, allowing loading at lung PO₂ and substantial unloading across the physiological tissue range."),
            ("Myoglobin has a lower P₅₀ and a hyperbolic oxygen-binding curve compared with adult Hb. Which interpretation is most appropriate?",
             "It has high oxygen affinity and lacks Hb's multi-subunit cooperative binding",
             ["It requires four oxygen molecules to bind at once", "It has lower affinity because its P₅₀ is lower", "It is the main carrier of oxygen in circulating plasma"],
             "Myoglobin is a monomer with high affinity, supporting intracellular oxygen handling. It is not the principal blood oxygen carrier."),
            ("At lung PO₂ near the plateau, a modest pressure fall causes little saturation change. At tissue PO₂ on the steep region, the same pressure fall causes much more unloading. What explains this?",
             "The local slope dS/dPO₂ differs along the sigmoid curve",
             ["PaO₂ and oxygen content are identical quantities", "The solubility coefficient becomes zero in the lungs", "Hb concentration changes whenever PO₂ falls"],
             "The steep region has a larger saturation change per unit PO₂. The plateau provides relative loading stability."),
            ("CO exposure both removes oxygen-binding sites and left-shifts binding at remaining sites. Which two changes must be represented to explain its impact?",
             "Reduced oxygen-carrying capacity and impaired oxygen unloading",
             ["Raised Hb concentration and easier unloading", "Increased dissolved oxygen and a right shift", "Only a fall in barometric pressure"],
             "A saturation-versus-PO₂ plot alone may hide loss of functional capacity. Fractional oxyHb and total content are needed."),
        ]
        for n, (prompt, correct, distractors, explanation) in enumerate(cases, 1):
            single(2, n, prompt, correct, distractors, explanation)
        p50 = rng.choice([3.,3.5,4.])
        po = rng.choice([4.,5.,6.])
        s = 100*po**2.7/(po**2.7+p50**2.7)
        number(2, 13, f"Use the supplied Hill model S = PO₂^2.7/(PO₂^2.7+P₅₀^2.7). At PO₂ {po:g} kPa and P₅₀ {p50:g} kPa, calculate saturation as a percentage. A calculator is allowed.",
               s, "Use the same pressure units for PO₂ and P₅₀, evaluate the supplied expression and multiply by 100. This is a teaching approximation.", "%", .5)
        s = rng.choice([.6,.7,.8])
        p50 = rng.choice([3.,3.5,4.])
        po = p50*(s/(1-s))**(1/2.7)
        number(2, 14, f"In a Hill model with n 2.7, saturation is {s*100:g}% at PO₂ {po:.3f} kPa. Infer P₅₀ using P₅₀ = PO₂×[(1−S)/S]^(1/2.7), with S as a fraction.",
               p50, "Use fractional saturation and the supplied inverse expression. A lower inferred P₅₀ means higher affinity.", tolerance=.05)
        po = rng.choice([4.,4.5,5.])
        difference = 100*(po**2.7/(po**2.7+3.0**2.7)-po**2.7/(po**2.7+4.2**2.7))
        number(2, 15, f"At tissue PO₂ {po:g} kPa, compare Hill curves with P₅₀ 3.0 and 4.2 kPa, n 2.7. Using S = PO₂^n/(PO₂^n+P₅₀^n), calculate how many percentage points lower saturation is on the 4.2 kPa curve.",
               difference, "Calculate each saturation at the same PO₂ and subtract right-shifted saturation from left-shifted saturation; report percentage points, not relative percent.", "percentage points", .5)
        arterial = rng.choice([.96,.98,1.])
        venous = rng.choice([.55,.60,.65])
        hb = rng.choice([12.,14.,16.])
        number(2, 16, f"A hypothetical shifted curve gives arterial saturation {arterial*100:g}% and tissue-end capillary saturation {venous*100:g}%. Hb is {hb:g} g/dL. Neglect dissolved oxygen. Calculate oxygen unloaded per dL, using Hb capacity 1.34.",
               1.34*hb*(arterial-venous), "Unloaded Hb-bound content = 1.34×Hb×(arterial−end-capillary saturation), using fractions.", "mL O₂/dL", .1)

        # Objective 3: integrated content, delivery, extraction and Fick calculations.
        hb = rng.choice([8.,9.,10.,11.])
        co = rng.choice([3.5,4.,4.5])
        sa = .96
        number(3, 1, f"Hb {hb:g} g/dL, SaO₂ 96%, PaO₂ 12 kPa and cardiac output {co:g} L/min. Calculate DO₂ using capacity 1.34 and dissolved coefficient 0.0225. Include the dL-to-L conversion.",
               (1.34*hb*sa+.0225*12)*co*10, "First calculate CaO₂ in mL/dL, then multiply by cardiac output and 10.", "mL O₂/min", 8.)
        hb = rng.choice([7.,8.,9.])
        oldhb = 14.
        number(3, 2, f"Initially Hb is 14 g/dL, SaO₂ 98%, PaO₂ 13 kPa and cardiac output 5 L/min. Hb falls to {hb:g} g/dL; SaO₂ and PaO₂ remain unchanged. What cardiac output would preserve the original DO₂? Use 1.34 and 0.0225.",
               5*(1.34*oldhb*.98+.0225*13)/(1.34*hb*.98+.0225*13), "Set old CaO₂×old CO = new CaO₂×new CO. Include the unchanged dissolved component in both contents.", "L/min", .08)
        ca = rng.choice([16.,18.,20.])
        vo = rng.choice([240.,270.,300.])
        co = rng.choice([4.,5.,6.])
        number(3, 3, f"CaO₂ is {ca:g} mL/dL, cardiac output {co:g} L/min and VO₂ {vo:g} mL/min. Calculate whole-body oxygen extraction as a percentage of delivery.",
               vo/(ca*co*10)*100, "DO₂ = CaO₂×CO×10; extraction ratio = VO₂/DO₂. Report a percentage.", "%", .3)
        ca = rng.choice([16.,18.,20.])
        vo = rng.choice([200.,240.,280.])
        co = rng.choice([4.,5.,6.])
        number(3, 4, f"CaO₂ {ca:g} mL/dL, cardiac output {co:g} L/min and VO₂ {vo:g} mL/min. Infer mixed venous oxygen content from VO₂ = CO×10×(CaO₂−CvO₂).",
               ca-vo/(co*10), "CvO₂ = CaO₂−VO₂/(CO×10). Use content differences rather than PO₂ differences in the Fick equation.", "mL O₂/dL", .1)
        vo = rng.choice([240.,270.,300.])
        ca = rng.choice([18.,19.,20.])
        cv = ca-rng.choice([4.,5.,6.])
        number(3, 5, f"VO₂ is {vo:g} mL/min, CaO₂ {ca:g} mL/dL and mixed venous CvO₂ {cv:g} mL/dL. Calculate cardiac output by the Fick principle.",
               vo/(10*(ca-cv)), "CO = VO₂/[10(CaO₂−CvO₂)]. The factor 10 converts L of blood to dL.", "L/min", .08)
        hb = rng.choice([10.,12.,14.])
        co = rng.choice([3.,4.,5.])
        vo = rng.choice([220.,250.,280.])
        sv = .98-vo/(10*co*1.34*hb)
        number(3, 6, f"Hb {hb:g} g/dL, SaO₂ 98%, cardiac output {co:g} L/min and VO₂ {vo:g} mL/min. Neglect dissolved oxygen and assume no dyshemoglobins. Infer mixed venous saturation as a percentage using Hb capacity 1.34.",
               sv*100, "VO₂ = CO×10×1.34×Hb×(SaO₂−SvO₂). Rearrange for SvO₂, using saturation fractions before multiplying by 100.", "%", .4)
        target = rng.choice([600.,700.,800.])
        co = rng.choice([4.,5.,6.])
        number(3, 7, f"In a mathematical model, what Hb in g/L gives DO₂ {target:g} mL/min at cardiac output {co:g} L/min, SaO₂ 95% and PaO₂ 10 kPa? Use 1.34 and 0.0225. This is not a transfusion threshold.",
               10*(target/(co*10)-.0225*10)/(1.34*.95), "Required CaO₂ = DO₂/(CO×10). Subtract dissolved oxygen, divide by 1.34×0.95 to obtain g/dL, then ×10 for g/L.", "g/L", .5)
        drop = rng.choice([.2,.25,.3])
        number(3, 8, f"Cardiac output falls by {drop*100:g}% while CaO₂ rises by 10%. By what percentage does DO₂ fall relative to baseline?",
               (1-(1-drop)*1.1)*100, "Delivery ratio = flow ratio×content ratio = (1−flow fall)×1.10. Subtract this ratio from 1 and multiply by 100.", "%", .2)
        co = rng.choice([4.,5.,6.])
        rise = rng.choice([30.,40.,50.])
        number(3, 9, f"At fixed Hb, SaO₂ and cardiac output {co:g} L/min, PaO₂ rises by {rise:g} kPa. Calculate the rise in DO₂ attributable solely to dissolved oxygen, coefficient 0.0225 mL/dL/kPa.",
               .0225*rise*co*10, "ΔDO₂ = 0.0225×ΔPaO₂×CO×10. The Hb-bound component is held fixed.", "mL O₂/min", 1.)
        ca = rng.choice([16.,18.,20.])
        ci = rng.choice([2.,2.5,3.])
        number(3, 10, f"CaO₂ is {ca:g} mL/dL and cardiac index {ci:g} L/min/m². Calculate indexed oxygen delivery DO₂I.",
               ca*ci*10, "DO₂I = CaO₂×cardiac index×10. The body-surface-area denominator is already included in the index.", "mL O₂/min/m²", 3.)
        ca = rng.choice([16.,18.,20.])
        vo = rng.choice([240.,270.,300.])
        maximum = rng.choice([.35,.4,.45])
        number(3, 11, f"In an illustrative model, VO₂ is {vo:g} mL/min, CaO₂ {ca:g} mL/dL, and the specified maximum extraction fraction is {maximum:g}. What minimum cardiac output meets consumption without exceeding that assumed limit?",
               vo/(ca*10*maximum), "Set VO₂ = extraction limit×CaO₂×CO×10 and rearrange. The stated limit is a modelling assumption, not a universal physiological cutoff.", "L/min", .08)
        ca = rng.choice([16.,18.,20.])
        co = rng.choice([4.,5.,6.])
        rise = rng.choice([80.,100.,120.])
        number(3, 12, f"CaO₂ remains {ca:g} mL/dL and CO remains {co:g} L/min. VO₂ rises by {rise:g} mL/min. By how many mL O₂/dL must mixed venous content fall, according to Fick?",
               rise/(co*10), "At fixed CaO₂ and flow, the increase in extraction content equals ΔVO₂/(CO×10).", "mL O₂/dL", .08)
        ca = rng.choice([16.,18.,20.])
        cv = ca-rng.choice([4.,5.,6.])
        number(3, 13, f"Arterial content is {ca:g} mL/dL and mixed venous content is {cv:g} mL/dL. Calculate extraction as a percentage, without needing cardiac output.",
               (ca-cv)/ca*100, "Extraction ratio = (CaO₂−CvO₂)/CaO₂ when both samples represent the same steady-state circulation.", "%", .3)
        hb = rng.choice([8.,10.,12.])
        sa = rng.choice([.9,.94,.98])
        co = rng.choice([4.,5.,6.])
        number(3, 14, f"Hb {hb:g} g/dL and cardiac output {co:g} L/min are fixed. SaO₂ rises from {sa*100:g}% to 100%. Neglect dissolved oxygen. Calculate the increase in DO₂ using capacity 1.34.",
               1.34*hb*(1-sa)*co*10, "ΔDO₂ = 1.34×Hb×absolute saturation-fraction increase×CO×10. A high initial saturation limits the gain.", "mL O₂/min", 2.)
        co = rng.choice([4.,5.,6.])
        ca = rng.choice([15.,16.,18.])
        number(3, 15, f"CaO₂ is initially {ca:g} mL/dL and CO {co:g} L/min. Content falls by 20%. What percentage increase in CO is required to preserve DO₂?",
               25., "New CO/old CO = 1/0.8 = 1.25. A 20% content fall requires a 25% flow rise, not a 20% rise.", "%", .2)
        hb = rng.choice([10.,12.,14.])
        sa, sv = .98, .68
        co = rng.choice([4.,5.,6.])
        number(3, 16, f"Hb {hb:g} g/dL, SaO₂ 98%, SvO₂ 68%, PaO₂ 13 kPa, PvO₂ 5 kPa and CO {co:g} L/min. Calculate VO₂ by Fick, including dissolved oxygen. Use 1.34 and 0.0225 for arterial and venous content.",
               (1.34*hb*(sa-sv)+.0225*(13-5))*co*10, "CaO₂−CvO₂ = 1.34×Hb×(0.98−0.68)+0.0225×(13−5). Multiply this content difference by CO×10.", "mL O₂/min", 4.)
    else:
        raise ValueError("Unknown topic")
    for question in questions:
        if question["type"] == "number":
            question["working"] = original_working(question)
    return questions + additional_questions(topic)



def arithmetic(expression):
    """Evaluate author-written numerical substitutions, never learner input."""
    import ast
    import operator
    operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                 ast.Div: operator.truediv, ast.Pow: operator.pow}
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        raise ValueError("Unsupported author calculation")
    return visit(ast.parse(expression, mode="eval").body)


def additional_questions(topic):
    """72 further distinct stems per topic, with explicit worked substitutions."""
    out = []
    prefix = "c2" if topic == "cascade" else "t2"
    def N(obj, n, prompt, formula, expression, unit="kPa", tolerance=.15, note=""):
        answer = arithmetic(expression)
        q = numeric(f"{prefix}_{obj}_{n:02}", obj, prompt, answer, tolerance, unit,
                    note or "Use the stated assumptions and the values in this question; round only the final answer.")
        q["working_expression"] = expression
        q["working"] = f"Relationship: {formula}\nSubstitution: {pretty_expression(expression)} = {answer:.6g} {unit}."
        out.append(q)
    def C(obj, n, prompt, correct, wrong, why):
        out.append(mcq(f"{prefix}_{obj}_{n:02}", obj, prompt, [correct]+wrong, correct, why))
    if topic == "cascade":
        # Objective 1: two numerical reconstructions and two interpretations per new paper.
        N(1,17,"A room-air cascade has PB 90 kPa and PH₂O 6.3 kPa. Calculate the combined partial pressure of all dry gases after full humidification, before isolating oxygen's share.","Dry-gas pressure = PB−PH₂O","90-6.3",note="This is total dry-gas pressure, not PIO₂. Only its oxygen share is multiplied by FiO₂.")
        N(1,18,"An airway model contains fully humidified inspired gas with total dry-gas pressure 83.7 kPa. Oxygen contributes 25.11 kPa. Calculate inspired oxygen as a percentage of the dry gases.","FiO₂ (%) = 100×PIO₂/(PB−PH₂O)","100*25.11/83.7","%",.2)
        C(1,19,"A diagram labels 21% oxygen as though it were 21 kPa at every altitude. Which correction preserves the distinction between fraction and pressure?","Keep FiO₂ approximately 0.21 but calculate dry PO₂ as 0.21×local PB",["Replace FiO₂ with PB expressed as a percentage","Keep dry PO₂ fixed at 21 kPa and reduce nitrogen alone","Subtract arterial PaCO₂ from FiO₂ before applying Dalton's law"],"FiO₂ is a dimensionless fraction, while PO₂ is a pressure. A constant fraction gives a lower oxygen pressure when total barometric pressure falls. The dry-air stage precedes humidification and alveolar exchange.")
        C(1,20,"A candidate draws the cascade with 'oxygen content, mL/dL' on its y-axis and labels atmospheric air 21.3. Which change is necessary?","Use PO₂ in kPa for the cascade pressure axis; blood content requires a separate plot",["Keep the label because kPa and mL/dL are interchangeable","Multiply every point by haemoglobin concentration, including atmospheric gas","Relabel the entire axis as percentage saturation"],"The conventional oxygen cascade compares partial pressures in gas and tissue compartments. Oxygen content measures amount per blood volume and depends on Hb and saturation; atmospheric air has no haemoglobin saturation.")
        N(1,21,"Dry inspired PO₂ at a remote site is 18.9 kPa on FiO₂ 0.21. What will PIO₂ be after full humidification at PH₂O 6.3 kPa?","PIO₂ = dry PO₂−FiO₂×PH₂O","18.9-0.21*6.3")
        N(1,22,"A diagram gives dry inspired PO₂ 30.0 kPa and humidified PIO₂ 28.11 kPa. If PH₂O is 6.3 kPa, infer PB.","FiO₂ = (dry PO₂−PIO₂)/PH₂O; PB = dry PO₂/FiO₂","30/((30-28.11)/6.3)",tolerance=.3)
        C(1,23,"For a pressure cascade during increased FiO₂, why should the tissue and mitochondrial points not simply be multiplied by the FiO₂ ratio?","Local flow, extraction and diffusion regulate these points; they are not fixed fractions of inspired pressure",["All tissue PO₂ must remain zero regardless of supply","Every cascade compartment has a linear FiO₂ relationship through the same origin","Only nitrogen determines mitochondrial PO₂"],"The inspired gas pressure is directly proportional to FiO₂ at fixed PB, but blood loading and tissue oxygenation also involve a nonlinear binding curve, flow and metabolism. The gas equation alone does not predict intracellular pressure.")
        C(1,24,"A resting cascade identifies arterial PO₂ as 12.5 kPa and mixed venous PO₂ as 5.3 kPa. What does their pressure difference fail to quantify on its own?","The amount of oxygen removed per dL of blood",["The direction of the pressure fall","Which listed pressure is greater","Whether the two pressures differ numerically"],"Extraction is calculated from arterial minus venous oxygen CONTENT, not the PO₂ difference alone. Hb, saturation and dissolved oxygen link pressure to content; the relation is nonlinear.")
        N(1,25,"In two airway models PB is 100 kPa and FiO₂ 0.30. Water vapour pressure is 4.0 kPa in model A and 6.3 kPa in B. Calculate PIO₂(A) minus PIO₂(B).","ΔPIO₂ = FiO₂×[PH₂O(B)−PH₂O(A)]","0.30*(6.3-4.0)")
        N(1,26,"At PB 85 kPa a fully humidified gas has PIO₂ 23.61 kPa and PH₂O 6.3 kPa. Calculate the dry inspired PO₂ before humidification.","FiO₂ = PIO₂/(PB−PH₂O); dry PO₂ = FiO₂×PB","23.61/(85-6.3)*85")
        C(1,27,"A teacher uses a mixed venous PO₂ as a surrogate for every tissue's PO₂. Why is this interpretation too strong?","Mixed venous blood combines returns from organs with different flow and extraction",["Mixed venous blood contains only blood from skeletal muscle","Every organ has identical capillary transit time","Mixing blood makes all mitochondrial pressures identical"],"A mixed venous sample represents the systemic balance of delivery and consumption. Organs and local tissue regions differ; a global return value cannot specify every intracellular PO₂.")
        C(1,28,"A sample has PaO₂ 13 kPa. Is that sufficient to locate all downstream points on an individual's oxygen cascade?","No; Hb, flow, extraction and local diffusion can differ despite the same arterial tension",["Yes; mitochondrial PO₂ is always arterial PO₂ divided by 40","Yes; mixed venous PO₂ is always exactly 40% of PaO₂","No; arterial blood has no dissolved oxygen"],"The arterial pressure is one compartment measurement. Content and regional oxygen supply require additional variables, and tissue consumption maintains downstream gradients. There is no universal downstream pressure ratio.")
        N(1,29,"At PB 101.3 kPa and PH₂O 6.3 kPa, humidified inspired PO₂ is to be maintained at 30 kPa. PB then falls to 80 kPa. Calculate the required increase in FiO₂ in percentage points.","ΔFiO₂ (%) = 100×[PIO₂/(new PB−PH₂O)−PIO₂/(old PB−PH₂O)]","100*(30/(80-6.3)-30/(101.3-6.3))","percentage points",.2)
        N(1,30,"At FiO₂ 0.28, humidified PIO₂ decreases from 26.6 to 23.8 kPa with PH₂O unchanged. What is the fall in PB?","Fall in PB = fall in PIO₂/FiO₂","(26.6-23.8)/0.28")
        C(1,31,"A candidate places inspired PO₂ and PaO₂ on a saturation-only graph with no Hb curve or gas pressure scale. Why cannot inspired PO₂ be read as an Hb saturation?","Inspired gas has a PO₂ but no haemoglobin binding-site occupancy",["Gas PO₂ can only be expressed in percentages","Every inspired oxygen molecule is bound to haemoglobin","Inspired pressure and saturation are always numerically identical"],"Partial pressure is defined for a gas mixture and dissolved oxygen. Saturation describes occupied Hb binding sites in blood. A dissociation curve relates blood pressure and saturation under stated conditions; it does not give atmospheric Hb saturation.")
        C(1,32,"Which diagram feature best prevents a learner from mistaking the normal alveolar–arterial fall for systemic tissue extraction?","Show alveolar and arterial compartments before the separate tissue and venous return stages",["Put tissue extraction between inspired gas and alveoli","Omit the arterial compartment entirely","Join mitochondria directly to the atmospheric gas point without labels"],"Alveolar-to-arterial differences concern lung gas exchange and venous admixture. Tissue extraction occurs after arterial delivery to the systemic circulation and determines returning venous content.")
        N(1,33,"In a dry gas mixture at PB 92 kPa, nitrogen and other non-oxygen gases contribute 64.4 kPa in total. Calculate oxygen partial pressure before humidification.","Dalton's law: oxygen pressure = total−other-gas pressures","92-64.4")
        N(1,34,"At PB 90 kPa, inspired oxygen percentage rises from 25% to 35%. With PH₂O 6.3 kPa, calculate the increase in humidified inspired PO₂.","ΔPIO₂ = (new FiO₂−old FiO₂)×(PB−PH₂O)","(0.35-0.25)*(90-6.3)")
        C(1,35,"A cascade graph shows a greater pressure drop in one step than another. Why does this not by itself identify the step transporting the greatest oxygen quantity?","Pressure differences and oxygen flux are different quantities; conductance, binding, flow and consumption also matter",["The largest pressure drop always implies the greatest blood oxygen content","All steps have identical conductance and no binding effects","Oxygen flux is measured in kPa"],"A pressure gradient can drive diffusion, but flux additionally depends on conductance and geometry. Blood carries oxygen mainly on Hb. A pressure-only graph does not quantify oxygen amount per minute.")
        C(1,36,"A reference mitochondrial PO₂ is labelled 0.3 kPa. What is the best way to use this in a viva?","State it is illustrative and explain how consumption and local diffusion maintain a low, variable pressure",["Treat it as a diagnostic cutoff valid for all organs","Use it to calculate cardiac output without any other measurements","Assume it equals oxygen saturation 0.3%"],"Intracellular pressure varies with tissue, microcirculation and metabolic state. It is useful as a conceptual low-pressure endpoint, but does not provide a universal reference limit or a direct measure of saturation.")
        N(1,37,"A fully humidified inspired sample at PB 100 kPa has PIO₂ 18.74 kPa. PH₂O is 6.3 kPa. What is the inspired oxygen percentage?","FiO₂ (%) = 100×PIO₂/(PB−PH₂O)","100*18.74/(100-6.3)","%",.2)
        N(1,38,"At PB 75 kPa, which FiO₂ percentage makes dry inspired PO₂ equal the sea-level room-air dry value calculated with PB 101.3 kPa and FiO₂ 0.21?","FiO₂ (%) = 100×0.21×sea-level PB/local PB","100*0.21*101.3/75","%",.2)
        C(1,39,"Which annotation most clearly distinguishes a resting room-air reference cascade from an individual patient prediction?","State FiO₂, PB, temperature and PaCO₂, and label downstream tissue points as schematic",["Label every point 'normal' without conditions","Use pressure, saturation and content interchangeably","Remove units to avoid differences between patients"],"Inspired and alveolar pressures depend on stated conditions. Arterial pressure additionally depends on gas exchange; tissue points require further assumptions. Clear units and labels prevent overinterpretation.")
        C(1,40,"An anaemic patient and a non-anaemic patient have identical pressure cascades from atmosphere to arterial blood. Which conclusion remains possible?","Their arterial oxygen contents and deliveries may still differ substantially",["Their Hb concentrations must be identical","Their cardiac outputs must be identical","Their organs must have identical oxygen consumption"],"The upstream pressure cascade does not specify Hb concentration or flow. Identical arterial PO₂ can coexist with different Hb-bound oxygen content and different DO₂.")

        # Objective 2: six sets of four distinct mechanism questions.
        data = [
          ("An increase in dead-space ventilation occurs while total minute ventilation stays constant and CO₂ production is unchanged. What mechanism can lower PAO₂?","Effective alveolar ventilation falls, raising PaCO₂ and the CO₂/R subtraction",["FiO₂ must increase","PH₂O must become zero","Barometric pressure must fall"],"Minute ventilation includes dead space. At fixed total ventilation, a larger wasted component reduces effective alveolar ventilation. PaCO₂ then rises at steady state, reducing PAO₂ through the gas equation."),
          ("Minute ventilation is unchanged but tidal volume falls and respiratory rate rises proportionately. Anatomical dead space per breath is unchanged. Why can oxygenation worsen?","More minute ventilation is spent ventilating dead space, so alveolar ventilation may fall",["Alveolar ventilation must remain identical whenever minute ventilation is identical","The respiratory quotient automatically becomes 1","Every breath now has a larger oxygen fraction"],"VA=(VT−VD)×rate. With smaller breaths, dead space occupies a greater fraction of each breath, and increasing rate does not necessarily preserve VA. The subsequent PaCO₂ rise can lower PAO₂."),
          ("A room-air PaO₂ is low while calculated PAO₂ is also low and only slightly above PaO₂. What is the key next distinction?","Separate low inspired pressure or hypoventilation from impaired alveolar-to-arterial transfer",["Conclude severe shunt solely from the low PaO₂","Assume the inspired oxygen fraction is 100%","Infer mitochondrial failure from PaO₂ alone"],"A small A–a gradient suggests arterial blood is relatively close to the available alveolar pressure. Low PB and hypoventilation can lower that alveolar pressure. PaO₂ alone does not establish the mechanism."),
          ("A lung unit has ventilation but virtually no perfusion. What does its alveolar gas tend towards?","Humidified inspired gas, because little blood extracts oxygen or delivers CO₂",["Mixed venous composition despite absent perfusion","An oxygen pressure of zero","A water-vapour pressure equal to the entire PB at 37°C"],"As V/Q tends to infinity, alveolar gas approaches inspired gas. This is the dead-space limit, distinct from the shunt limit where ventilation is absent and perfusion persists."),
          ("Why does increasing FiO₂ tend to improve low-V/Q hypoxaemia more readily than a true zero-ventilation shunt?","Some fresh gas still reaches low-V/Q units, allowing their alveolar PO₂ to rise",["Low-V/Q blood has no Hb","A true shunt has higher ventilation than low-V/Q units","FiO₂ changes PB rather than oxygen fraction"],"Low V/Q is not the same as V/Q=0. Oxygen can reach partly ventilated units, while true shunted blood bypasses gas exchange. The response depends on the degree of mismatch and shunt."),
          ("A positive A–a gradient is reported. Why does it not alone distinguish diffusion limitation from V/Q inequality?","Both mechanisms can reduce arterial oxygenation relative to the calculated alveolar pressure",["Only diffusion limitation ever widens A–a","Only V/Q inequality ever widens A–a","A–a measures the Hb concentration directly"],"A–a identifies an alveolar-to-arterial oxygen difference under stated conditions, not a unique diagnosis. Shunt, V/Q inequality and diffusion limitation require contextual interpretation."),
          ("Why is normal end-capillary PO₂ in one well-ventilated unit insufficient to guarantee normal whole-lung PaO₂?","Other perfused units or shunted blood can have lower oxygen content before blood mixes",["All regional blood has identical content by definition","Only the highest regional PO₂ determines PaO₂","Arterial PO₂ ignores blood mixing"],"Whole-lung arterial blood combines regional returns. Mixing is weighted by perfusion and content; a normal sampled or ideal unit does not exclude substantial heterogeneity elsewhere."),
          ("During acute hypoventilation with no new V/Q abnormality, PaO₂ falls. Why is subtracting its fall from the old A–a gradient misleading?","The alveolar pressure also falls and must be recalculated using the new PaCO₂",["PAO₂ stays fixed whenever FiO₂ is fixed","PH₂O must rise by the same amount as PaO₂ falls","PaCO₂ has no role in the alveolar gas equation"],"A–a uses simultaneous alveolar and arterial conditions. Holding the old PAO₂ while PaCO₂ changes can falsely suggest a new transfer defect."),
          ("At altitude, why does hyperventilation only partly restore the oxygen cascade despite unchanged ambient FiO₂?","Lower PaCO₂ reduces the alveolar subtraction but does not restore the lost barometric pressure",["Hyperventilation raises PB back to sea-level pressure","It replaces nitrogen with oxygen automatically","It eliminates water vapour completely"],"The two influences enter different terms of the gas equation. Lower PaCO₂ helps, but PIO₂ remains constrained by the lower PB and by humidification."),
          ("A candidate assumes PH₂O falls in direct proportion to PB at altitude. What correction is needed for saturated gas at the same temperature?","Use the saturated water vapour pressure determined by temperature, not a fixed fraction of PB",["Set PH₂O equal to 21% of PB","Set PH₂O equal to PaCO₂","Assume fully saturated gas contains no water"],"At 37°C the saturated water vapour pressure is approximately 6.3 kPa over the usual physiological pressure range. Its fraction of total pressure therefore increases as PB falls."),
          ("Why should a falling PaCO₂ not always be interpreted as improving lung oxygen transfer?","It can reflect hyperventilation while a significant A–a gradient persists",["A lower PaCO₂ always proves shunt has resolved","PaCO₂ determines Hb concentration","Lower PaCO₂ forces FiO₂ to fall"],"Hyperventilation raises calculated PAO₂ by reducing the CO₂ term. Arterial pressure may still remain well below that alveolar value if transfer is impaired. Recalculate A–a under the new conditions."),
          ("A calculated PAO₂ is negative with extreme inputs. What is the sound interpretation?","The assumed steady state is not physically sustainable under those inputs; check assumptions and inputs",["Negative alveolar oxygen pressure is a normal altitude adaptation","Record zero without flagging the inconsistency","A negative answer means arterial blood produces oxygen"],"The algebra can produce an impossible value when oxygen supply cannot sustain the assumed CO₂ and metabolism. The result flags an invalid model state, rather than an actual negative gas pressure."),
          ("If pulmonary capillary transit time increases while membrane properties and alveolar PO₂ stay fixed, what generally happens to the opportunity for diffusion equilibration?","There is more time for equilibration, although disease may still prevent completion",["There is less time for equilibration","The membrane becomes infinitely thick","Inspired PO₂ must fall"],"Diffusion depends on the driving gradient and conductance over the available transit time. A longer time can reduce diffusion limitation; it does not correct shunt or guarantee normal whole-lung oxygenation."),
          ("Oxygen diffusion distance increases due to membrane thickening, with surface area and pressure gradient held fixed. What does Fick's diffusion law predict?","Reduced diffusive oxygen flux",["Increased flux proportional to thickness","No change because gradient alone determines flux","An automatic increase in FiO₂"],"Diffusive flux is proportional to area and pressure gradient and inversely proportional to thickness. This is Fick's diffusion law, distinct from the circulatory Fick principle used to calculate cardiac output."),
          ("A learner claims increasing FiO₂ corrects a low tissue PO₂ in every case. What limitation must be explained?","Oxygen content, regional flow, extraction and utilisation may limit tissue oxygenation despite high arterial PO₂",["Tissues never require blood flow","FiO₂ directly fixes mitochondrial PO₂","Increasing arterial PO₂ always doubles Hb concentration"],"Once Hb is nearly saturated, additional pressure adds relatively little bound oxygen. Low flow or impaired utilisation can remain limiting, so inspired oxygen is not a complete tissue model."),
          ("Which observation most directly supports the idea that arterial pressure alone is an incomplete measure of tissue supply?","A major fall in cardiac output with unchanged PaO₂ can reduce DO₂",["PO₂ has pressure units","Dry inspired oxygen follows Dalton's law","Water vapour is present in airways"],"Delivery equals arterial content multiplied by flow. Even if content and pressure are preserved, a flow reduction lowers oxygen supplied per minute and may increase extraction."),
          ("Oxygen extraction rises in skeletal muscle while cerebral flow adjusts differently. Why need not their venous PO₂ changes be identical?","Regional supply–demand balances and vascular responses differ",["All organs share one identical local V/Q ratio","Every regional venous return must equal pulmonary arterial blood exactly","Hb loses its binding properties outside the lungs"],"Each organ has its own flow, metabolism and microvascular distribution. Mixed venous blood combines these returns and cannot impose identical local venous pressures."),
          ("Can a normal room-air A–a gradient exclude anaemic tissue hypoxia?","No; A–a describes lung pressure transfer and does not measure Hb oxygen-carrying capacity",["Yes; A–a directly measures total Hb","Yes; all tissue hypoxia requires a widened A–a gradient","No; normal lungs contain no alveolar oxygen"],"An anaemic person can have normal alveolar and arterial tensions yet low CaO₂. Oxygen delivery also depends on cardiac output, so a normal gas-exchange gradient does not rule out a supply problem."),
          ("Why is the respiratory quotient an assumption that should be stated in a gas-equation calculation?","Changing R changes the oxygen subtraction corresponding to a given CO₂ term",["R is identical to the inspired oxygen fraction","R is always exactly 1 in every metabolic state","R is measured in kPa"],"R is the dimensionless ratio VCO₂/VO₂. The commonly assumed 0.8 is a useful mixed-substrate approximation, but changing it changes the estimated PAO₂."),
          ("Under what condition does the standard alveolar gas equation need modification for the inspired CO₂ term?","When inspired CO₂ is no longer negligible, such as rebreathing",["Whenever Hb is below 150 g/L","Whenever PB is expressed in kPa","Whenever SaO₂ exceeds 95%"],"The usual expression assumes negligible inspired CO₂. Rebreathing violates the gas-balance assumption; simply applying the standard equation without recognising this may mislead."),
          ("Two perfused units have equal ventilation but different blood flows. Why need their alveolar PO₂ differ?","Their ventilation-to-perfusion ratios differ, changing gas exchange per unit ventilation",["Equal ventilation forces identical V/Q regardless of flow","Blood flow cannot influence alveolar gas composition","Only inspired nitrogen affects alveolar PO₂"],"Alveolar gas reflects ventilation relative to perfusion. A more heavily perfused unit extracts more oxygen per unit ventilation and tends towards a lower alveolar oxygen pressure, other conditions equal."),
          ("A poorly ventilated lung region also receives little perfusion because of local hypoxic pulmonary vasoconstriction. What is the immediate matching benefit?","Blood is redirected away from a low-oxygen unit towards better-ventilated regions",["Its inspired oxygen fraction becomes 100%","All pulmonary shunt is abolished","The systemic Hb concentration doubles"],"Reducing flow to a poorly ventilated region can improve V/Q matching and reduce venous admixture. The response is partial and can be impaired; it does not create ventilation in a completely unventilated unit."),
          ("At steady state a person's metabolic CO₂ production rises but alveolar ventilation does not. Why may PAO₂ fall even if FiO₂ stays constant?","PaCO₂ rises, increasing the CO₂/R term, assuming R remains appropriate",["PB must decrease whenever metabolism rises","PH₂O is consumed by metabolism","FiO₂ becomes numerically equal to VO₂"],"PaCO₂ tracks CO₂ production divided by alveolar ventilation. The alveolar gas equation then predicts a lower PAO₂ for the same inspired pressure and R."),
          ("A clinician compares A–a gradients calculated using an estimated mask FiO₂ on two different breathing patterns. What uncertainty should be recognised?","Actual inspired oxygen fraction may differ from the estimate and change the calculated PAO₂",["FiO₂ is irrelevant to A–a","The patient's arterial PO₂ equals the mask label","FiO₂ can be read directly from haemoglobin concentration"],"An uncertain FiO₂ makes the inspired-pressure term and hence the calculated alveolar pressure uncertain. The result should not be interpreted with more precision than the input supports."),
        ]
        for n,(p,c,w,e) in enumerate(data,17): C(2,n,p,c,w,e)
        # Objective 3: all data specified, pressure answers in kPa or dimensionless ratios.
        N(3,17,"FiO₂ 0.25, PB 100 kPa, PH₂O 6.3 kPa, PaCO₂ 6 kPa and R 0.8. Calculate PAO₂ with the simplified equation.","PAO₂ = FiO₂(PB−PH₂O)−PaCO₂/R","0.25*(100-6.3)-6/0.8")
        N(3,18,"FiO₂ 0.35, PB 90 kPa, PH₂O 6.3 kPa, PaCO₂ 5.6 kPa, R 0.8 and PaO₂ 17 kPa. Calculate A–a.","A–a = FiO₂(PB−PH₂O)−PaCO₂/R−PaO₂","0.35*(90-6.3)-5.6/0.8-17")
        N(3,19,"At FiO₂ 0.30 and PB 95 kPa, PH₂O 6.3 kPa and R 0.8, PAO₂ is 20 kPa. Calculate PaCO₂ by rearranging the simplified equation.","PaCO₂ = R[FiO₂(PB−PH₂O)−PAO₂]","0.8*(0.30*(95-6.3)-20)")
        N(3,20,"PaCO₂ is initially 6 kPa. Metabolic CO₂ production rises by 20%, while alveolar ventilation rises by 50%. At steady state calculate the new PaCO₂.","New PaCO₂ = old PaCO₂×production ratio/ventilation ratio","6*1.2/1.5")
        N(3,21,"Minute ventilation is 8 L/min at 20 breaths/min, anatomical dead space is 0.15 L/breath and PaCO₂ is 6 kPa. Ventilation changes to 10 L/min at 20 breaths/min with the same dead space and CO₂ production. Calculate new PaCO₂ at steady state.","VA = VE−rate×VD; new PaCO₂ = old PaCO₂×old VA/new VA","6*(8-20*0.15)/(10-20*0.15)")
        N(3,22,"At FiO₂ 0.21, PB 101.3 kPa, PH₂O 6.3 kPa and R 0.8, PaCO₂ falls from 6.4 to 4.0 kPa while PaO₂ rises from 9 to 13 kPa. Calculate the fall in A–a gradient (old minus new).","Fall in A–a = rise in PaO₂−rise in PAO₂","(13-9)-(6.4-4)/0.8")
        N(3,23,"The measured inspired humidified PO₂ is 18 kPa, PAO₂ is 11 kPa and PaCO₂ is 5.6 kPa. Infer R from the simplified gas equation.","R = PaCO₂/(PIO₂−PAO₂)","5.6/(18-11)","ratio",.02)
        N(3,24,"At FiO₂ 0.21, PH₂O 6.3 kPa, R 0.8 and PaCO₂ 4 kPa, find the PB at which simplified PAO₂ equals 8 kPa.","PB = (PAO₂+PaCO₂/R)/FiO₂+PH₂O","(8+4/0.8)/0.21+6.3",tolerance=.3)
        N(3,25,"At PB 101.3 kPa, PH₂O 6.3 kPa, PaCO₂ 5 kPa and R 1.0, calculate PAO₂ while breathing FiO₂ 0.60. Here the simplified and fuller equations agree.","PAO₂ = FiO₂(PB−PH₂O)−PaCO₂/R","0.60*(101.3-6.3)-5/1")
        N(3,26,"At FiO₂ 0.80, PB 101.3 kPa, PH₂O 6.3 kPa, PaCO₂ 5.6 kPa and R 0.8, use the FULLER equation PAO₂ = FiO₂(PB−PH₂O)−PaCO₂[FiO₂+(1−FiO₂)/R].","Fuller gas equation with unequal-volume correction","0.80*(101.3-6.3)-5.6*(0.8+(1-0.8)/0.8)")
        N(3,27,"Using the simplified equation, at PB 101.3 kPa, PH₂O 6.3 kPa, PaCO₂ 8 kPa and R 0.8, calculate the ratio PAO₂ at FiO₂ 0.40 divided by PAO₂ at FiO₂ 0.20.","Ratio = [0.40(PB−PH₂O)−PaCO₂/R]/[0.20(PB−PH₂O)−PaCO₂/R]","(0.40*95-8/0.8)/(0.20*95-8/0.8)","ratio",.03,note="Doubling FiO₂ does not simply double PAO₂ because the same CO₂ term is subtracted from both inspired pressures.")
        N(3,28,"PAO₂ is 14 kPa and PaO₂ 10 kPa. If PAO₂ rises to 22 kPa and the A–a difference remains fixed as an explicit model assumption, estimate the new PaO₂.","New PaO₂ = new PAO₂−old A–a","22-(14-10)",note="The unchanged gradient is given for this calculation; it should not be assumed across changing FiO₂ in real patients without justification.")
        N(3,29,"At FiO₂ 0.21, PH₂O 6.3 kPa, PaCO₂ 5.3 kPa and R 0.8, compare PB 101.3 kPa with PB 80 kPa. Calculate the percentage fall in PAO₂ using the simplified equation, with PaCO₂ unchanged.","Percentage fall = 100×(old PAO₂−new PAO₂)/old PAO₂","100*(0.21*(101.3-80))/(0.21*(101.3-6.3)-5.3/0.8)","%",.3)
        N(3,30,"At PB 85 kPa, PH₂O 6.3 kPa, PaCO₂ 4.8 kPa and R 0.8, a model requires PaO₂ 12 kPa with A–a 1 kPa. Calculate required FiO₂ as a percentage with the simplified equation.","FiO₂ (%) = 100×(PaO₂+A–a+PaCO₂/R)/(PB−PH₂O)","100*(12+1+4.8/0.8)/(85-6.3)","%",.3)
        N(3,31,"At PB 101.3 kPa and PH₂O 6.3 kPa, FiO₂ falls from 0.35 to 0.30 while PaCO₂ falls from 6.4 to 4.0 kPa, with R 0.8. Calculate the net FALL in simplified PAO₂.","Net fall = FiO₂ fall×dry-gas pressure−PaCO₂ fall/R","(0.35-0.30)*(101.3-6.3)-(6.4-4)/0.8")
        N(3,32,"Using the fuller equation at FiO₂ 1.0, PB 101.3 kPa, PH₂O 6.3 kPa and PaCO₂ 5.3 kPa, calculate PAO₂. The R-dependent term cancels at this FiO₂.","At FiO₂ 1: PAO₂ = PB−PH₂O−PaCO₂","101.3-6.3-5.3")
        N(3,33,"In a model R changes from 0.8 to 0.7 while PaCO₂ stays 4.9 kPa and inspired conditions are fixed. Calculate the fall in simplified PAO₂.","Fall = PaCO₂×(1/new R−1/old R)","4.9*(1/0.7-1/0.8)")
        N(3,34,"At PB 70 kPa, PH₂O 6.3 kPa, FiO₂ 0.21, R 0.8 and A–a 0.6 kPa, what PaCO₂ would give arterial PO₂ 8 kPa in the simplified model?","PaCO₂ = R[FiO₂(PB−PH₂O)−(PaO₂+A–a)]","0.8*(0.21*(70-6.3)-(8+0.6))")
        N(3,35,"A candidate uses PaCO₂ 5 kPa without dividing by R 0.8. How much too high is their PAO₂ estimate relative to the simplified equation? Inspired conditions are otherwise correct.","Error = correct CO₂ subtraction−incorrect subtraction","5/0.8-5")
        N(3,36,"At FiO₂ 0.30, PB 101.3 kPa, PH₂O 6.3 kPa and R 0.8, alveolar ventilation must rise from its current value to lower PaCO₂ from 8 to 5 kPa. With CO₂ production fixed, calculate the required percentage increase in alveolar ventilation.","VA ratio = old PaCO₂/new PaCO₂; increase (%) = 100×(ratio−1)","100*(8/5-1)","%",.3)
        N(3,37,"At FiO₂ 0.21, PB 101.3 kPa, PH₂O 6.3 kPa and R 0.8, PaCO₂ doubles from 4 to 8 kPa. Calculate the percentage fall in PAO₂, rather than assuming it halves.","Percentage fall = 100×[(new CO₂−old CO₂)/R]/old PAO₂","100*((8-4)/0.8)/(0.21*(101.3-6.3)-4/0.8)","%",.3)
        N(3,38,"PAO₂ is 18 kPa, PaO₂ 12 kPa and FiO₂ 0.30. Calculate the A–a gradient divided by PAO₂ as a percentage. This is a model ratio, not a shunt fraction.","Normalised pressure difference (%) = 100×(PAO₂−PaO₂)/PAO₂","100*(18-12)/18","%",.3)
        N(3,39,"In an altitude model PB falls from 90 to 80 kPa while PH₂O stays 6.3 kPa. To keep PIO₂ unchanged, by what percentage must FiO₂ rise relative to its original fraction?","FiO₂ ratio = old dry-gas pressure/new dry-gas pressure","100*((90-6.3)/(80-6.3)-1)","%",.3)
        N(3,40,"FiO₂ is 0.40, PB 100 kPa, PH₂O 6.3 kPa, R 0.8 and PaCO₂ 6.4 kPa. Calculate the fraction of humidified inspired PO₂ lost in the CO₂/R subtraction, as a percentage.","Alveolar subtraction fraction (%) = 100×(PaCO₂/R)/PIO₂","100*(6.4/0.8)/(0.40*(100-6.3))","%",.3)
    else:
        # Transport objective 1: capacity, content and pressure; two calculations/two concepts per paper.
        N(1,17,"Hb is 120 g/L, fractional oxyHb 92% and PaO₂ 9 kPa. Calculate Hb-bound oxygen per litre of blood using capacity 1.34 mL/g; exclude dissolved oxygen.","Bound content (mL/L) = 1.34×Hb (g/L)×fractional oxyHb","1.34*120*0.92","mL O₂/L",1.)
        N(1,18,"Blood has CaO₂ 15 mL/dL, Hb 110 g/L and PaO₂ 12 kPa. Using capacity 1.34 and solubility 0.0225 mL/dL/kPa, calculate the fractional oxyHb percentage.","Fractional oxyHb (%) = 100×(CaO₂−0.0225×PaO₂)/(1.34×Hb in g/dL)","100*(15-0.0225*12)/(1.34*11)","%",.3)
        C(1,19,"A blood gas analyser calculates saturation from PaO₂ using a normal curve in suspected CO poisoning. What is its main limitation?","The normal-curve calculation does not measure COHb or available oxygen-carrying sites",["It directly measures every Hb species","It proves oxygen content is normal when calculated saturation is high","It measures cardiac output from PaO₂"],"Calculated saturation is model-dependent. Dyshemoglobins require co-oximetry; normal dissolved oxygen tension can coexist with reduced fractional oxyHb and impaired oxygen transport.")
        C(1,20,"Why should fractional oxyHb and functional SaO₂ not be used interchangeably in a sample with substantial dyshemoglobins?","Their denominators differ, so the same percentage does not necessarily represent the same fraction of total Hb carrying oxygen",["Both always include only COHb in the denominator","Neither depends on oxyHb","Both are pressure measurements in kPa"],"Fractional oxyHb uses total measured Hb species in the denominator. Functional saturation uses oxyHb plus deoxyHb. With abnormal Hb species, using the wrong denominator can overestimate oxygen-carrying content.")
        N(1,21,"Hb is 10 g/dL and SaO₂ 90%. A second sample has Hb 12 g/dL with the same Hb-bound oxygen content. Neglect dissolved oxygen and assume identical capacity. Calculate the second saturation percentage.","Equal bound content: Hb₁×S₁ = Hb₂×S₂","100*10*0.90/12","%",.2)
        N(1,22,"At PaO₂ 12 kPa, total CaO₂ is 18 mL/dL. Using 0.0225 mL/dL/kPa, calculate the Hb-bound component.","Bound oxygen = total content−dissolved content","18-0.0225*12","mL O₂/dL",.08)
        C(1,23,"A report changes Hb units from g/dL to g/L while leaving the numeric value and content formula unchanged. What error results if the true g/L value is entered directly into a formula requiring g/dL?","The Hb-bound term is overestimated tenfold",["It is underestimated tenfold","Only the dissolved term changes","There is no effect because the units are interchangeable"],"One g/dL equals 10 g/L. The standard 1.34×Hb expression for mL/dL requires Hb in g/dL. Convert g/L by dividing by 10 before calculating that content term.")
        C(1,24,"A learner inserts SaO₂ 97 rather than 0.97 into 1.34×Hb×SaO₂. Which component is affected?","The Hb-bound component is overestimated by a factor of 100",["Only the dissolved component is overestimated","The bound component is underestimated by 100","No component changes because percent is already a fraction"],"The binding term requires a fraction from 0 to 1. Percent means per hundred, so 97% must be divided by 100. The dissolved term depends on PaO₂ and is separate.")
        N(1,25,"Hb falls from 14 to 10 g/dL at SaO₂ 97% and unchanged PaO₂. Calculate the loss of Hb-bound oxygen content using capacity 1.34.","Lost bound content = 1.34×Hb fall×SaO₂ fraction","1.34*(14-10)*0.97","mL O₂/dL",.08)
        N(1,26,"At fixed Hb-bound oxygen content 16 mL/dL, PaO₂ rises from 10 to 50 kPa. Using coefficient 0.0225, calculate the percentage increase in TOTAL CaO₂ from the initial value.","Total content ratio = (bound+new dissolved)/(bound+old dissolved)","100*((16+0.0225*50)/(16+0.0225*10)-1)","%",.2)
        C(1,27,"Why does an isolated low cardiac output not necessarily lower the oxygen pressure measured in a systemic arterial sample?","The lungs may still oxygenate arterial blood normally even though less oxygen is delivered per minute",["Low flow always forces arterial PaO₂ to zero","Cardiac output and PaO₂ are the same variable","Hb cannot bind oxygen at low cardiac output"],"Pressure reflects the oxygen tension achieved by pulmonary gas exchange. Delivery additionally requires adequate flow. Venous oxygen content may fall through increased extraction while arterial pressure remains normal.")
        C(1,28,"A patient has a high PaO₂ after oxygen administration. Which extra measurement is indispensable for converting that pressure into a complete CaO₂ estimate?","Hb concentration plus an appropriate measured oxygen saturation or fractional oxyHb",["PaCO₂ alone without Hb","Respiratory rate alone","The inspired oxygen label alone"],"The dissolved term follows PaO₂, but normally most oxygen is Hb-bound. That term requires Hb and appropriate saturation. Pressure alone cannot establish the total amount carried.")
        N(1,29,"CaO₂ is 160 mL/L. Express it in mL/dL before using the standard DO₂ formula with its factor of 10.","One litre = 10 decilitres; mL/dL = mL/L÷10","160/10","mL O₂/dL",.05)
        N(1,30,"Hb is 90 g/L, SaO₂ 100% and PaO₂ 20 kPa. Calculate total CaO₂ in mL/L using capacity 1.34 and dissolved coefficient 0.0225 mL/dL/kPa.","CaO₂ (mL/L) = 10×[1.34×Hb(g/dL)×S+0.0225×PaO₂]","10*(1.34*9+0.0225*20)","mL O₂/L",.5)
        C(1,31,"What does a normal SpO₂ fail to establish without additional data?","Normal arterial oxygen content or adequate tissue oxygen delivery",["That the device has reported a percentage","That a saturation estimate is available","That saturation differs conceptually from pressure"],"SpO₂ estimates saturation, with measurement limitations. It does not measure Hb or cardiac output, so anaemia and low flow can reduce oxygen transport despite a high displayed value.")
        C(1,32,"Why is the statement 'PaO₂ is the amount of oxygen in arterial blood' imprecise?","PaO₂ is tension; amount per blood volume is content, dominated by Hb binding",["PaO₂ measures only CO₂","Content and pressure have identical dimensions","All oxygen is dissolved and none is Hb-bound"],"Pressure and content are related but different variables. The dissolved component creates tension, while the much larger Hb reservoir contributes content according to Hb concentration and saturation.")
        N(1,33,"Hb is 130 g/L, SaO₂ 96% and PaO₂ 11 kPa. Calculate the ratio Hb-bound oxygen divided by dissolved oxygen, using capacity 1.34 and solubility 0.0225 mL/dL/kPa.","Bound/dissolved ratio = (1.34×Hb in g/dL×S)/(0.0225×PaO₂)","(1.34*13*0.96)/(0.0225*11)","ratio",.5)
        N(1,34,"PaO₂ is 14 kPa and the measured dissolved oxygen content is 0.315 mL/dL. Calculate the implied oxygen solubility coefficient.","Solubility coefficient = dissolved content/PaO₂","0.315/14","mL O₂/dL/kPa",.0002)
        C(1,35,"A study compares oxygen transport at equal Hb concentration but different oxygen-carrying capacity constants. Why must the constant be stated in a calculation question?","Different accepted capacity assumptions can give different content estimates even with identical Hb and saturation",["The constant is measured in kPa","All capacity constants must numerically equal FiO₂","The constant replaces the saturation term"],"Clinical teaching commonly uses about 1.34 mL O₂/g, while theoretical or empirical conventions differ. A stated constant makes the numerical key reproducible and avoids penalising a different convention.")
        C(1,36,"Why can a low arterial oxygen content be present in both anaemia and hypoxaemia by different routes?","Anaemia reduces available Hb; hypoxaemia can reduce its saturation",["Both must reduce barometric pressure","Neither changes the Hb-bound term","Both always produce identical PaO₂"],"The bound term is capacity×Hb×saturation. Lowering either Hb or saturation lowers content; their pressure measurements and compensatory responses can differ.")
        N(1,37,"In a hypothetical sample, total Hb is 12 g/dL, fractional oxyHb 80% and PaO₂ 12 kPa. Calculate the content overestimate if functional saturation 100% is incorrectly substituted for fractional oxyHb. Use capacity 1.34 and the same dissolved term in both estimates.","Overestimate = 1.34×Hb×(wrong−correct fraction)","1.34*12*(1-0.80)","mL O₂/dL",.08)
        N(1,38,"At Hb 8 g/dL and PaO₂ 10 kPa, measured CaO₂ is 10 mL/dL. Infer the Hb-bound oxygen per gram of TOTAL Hb in this sample, subtracting dissolved content with coefficient 0.0225.","Bound O₂ per gram = (CaO₂−dissolved content)/Hb","(10-0.0225*10)/8","mL O₂/g Hb",.015,note="This measured bound amount per total gram includes the sample's saturation state; it is not necessarily the fully saturated oxygen capacity constant.")
        C(1,39,"An investigator reports oxygen content without stating whether it is per dL or per L. Why is this consequential for delivery calculations?","The missing volume unit can introduce a tenfold error when combined with cardiac output in L/min",["Both units always give the same numerical value","Only the saturation percentage is affected","Pressure units are interchangeable with blood volume units"],"Ten decilitres make a litre. CaO₂ in mL/dL needs ×10 when multiplied by L/min, while content already in mL/L does not. Always carry units through the calculation.")
        C(1,40,"A patient with impaired cellular oxygen utilisation has high mixed venous saturation. Which interpretation is safest?","High venous saturation can reflect poor extraction and does not prove adequate cellular oxygen use",["High venous saturation always proves normal metabolism","Arterial oxygen content must be zero","Mitochondria must be consuming more oxygen than is delivered"],"Mixed venous saturation reflects the delivery–consumption balance. It may rise with high delivery, low demand or impaired extraction; it is not a standalone measure of successful cellular metabolism.")

        # Transport objective 2: three concepts plus one worked curve calculation per paper.
        curve_sets = [
          [("Two curves have the same P₅₀ but different Hill coefficients. What differs?","The steepness/cooperativity of the saturation–pressure relation",["Necessarily the 50%-saturation pressure","Necessarily the total Hb concentration","The units of pressure"],"P₅₀ locates the 50% point; the Hill coefficient controls the shape in this approximation. Equal P₅₀ does not guarantee equal saturation at every other pressure."),
           ("At a pressure far above P₅₀, why does further oxygen loading produce a diminishing rise in Hb saturation?","Most binding sites are already occupied, so the curve approaches its upper limit",["Hb concentration increases without bound","Oxygen cannot dissolve at high pressure","The maximum saturation exceeds 200%"],"Saturation is bounded at 100% of available sites. Near the plateau additional pressure chiefly increases dissolved oxygen, rather than adding many occupied Hb sites."),
           ("A candidate calls any fall in SaO₂ a right shift. What must be distinguished?","Movement down an unchanged curve from a change in the curve's affinity",["Any pressure fall always changes P₅₀","A shift can be diagnosed without knowing PO₂","Saturation and affinity are synonyms"],"A lower PO₂ can lower saturation without changing P₅₀. A right shift means lower saturation at the same PO₂ under the compared conditions.")],
          [("Hypothermia and alkalosis occur together. At fixed PO₂, what combined affinity effect is expected?","Higher affinity with a lower P₅₀ and greater oxygen retention",["Lower affinity with a higher P₅₀","No possible effect on binding","A mandatory fall in Hb concentration"],"Both factors favour a left shift. This can support loading but reduce unloading at a given tissue PO₂; delivery still depends on content and flow."),
           ("Why can hyperventilation at altitude have opposing effects on tissue oxygen availability?","It raises alveolar PO₂ by lowering CO₂, while alkalosis can increase Hb affinity",["It raises PB while abolishing Hb binding","It always reduces alveolar PO₂ and lowers affinity","It makes Hb concentration independent of blood volume"],"The gas-equation effect improves alveolar oxygen pressure. Acute respiratory alkalosis left-shifts Hb, which can assist loading at low pressure but alter unloading. Net supply requires both processes to be considered."),
           ("Does moving from g/dL to g/L on a laboratory report change the underlying dissociation curve?","No; it changes numerical concentration units, not oxygen affinity",["Yes; P₅₀ increases tenfold","Yes; every Hb molecule gains ten sites","Yes; saturation becomes a pressure"],"A unit conversion is not a physiological intervention. It changes the value used in content arithmetic but does not itself change binding equilibrium or P₅₀.")],
          [("A reduced P₅₀ is reported alongside a lower total Hb concentration. Which two independent changes should be separated?","Higher affinity and reduced carrying capacity",["Lower affinity and increased carrying capacity","Higher PB and lower PH₂O","Identical affinity and identical capacity"],"P₅₀ describes affinity; Hb concentration determines the number of available binding sites per blood volume. A left shift cannot restore missing Hb capacity."),
           ("A red-cell preparation has increasing 2,3-DPG at otherwise fixed conditions. Which state does this favour?","The lower-affinity deoxygenated state, increasing P₅₀",["The higher-affinity state, decreasing P₅₀","A state with no Hb molecules","A change only in dissolved oxygen solubility"],"2,3-DPG binds preferentially to deoxyHb and stabilises its lower-affinity conformation. This favours unloading at fixed tissue pressure, while the effect on pulmonary loading depends on lung PO₂."),
           ("Which comparison best demonstrates cooperative rather than independent oxygen binding?","Affinity for subsequent O₂ increases after initial O₂ binding to Hb",["Every site has exactly the same fixed affinity regardless of occupancy","Oxygen binds only to plasma water","PaO₂ alone fixes total Hb mass"],"Interactions between subunits alter affinity as occupancy changes, producing the sigmoid relation. Independent identical sites would produce a simpler hyperbolic binding relation.")],
          [("Why is 'a right shift always improves DO₂' an incomplete statement?","DO₂ uses arterial content and flow; a shift may also change lung loading",["DO₂ is defined only by tissue P₅₀","Arterial saturation never changes with affinity","Blood flow is irrelevant to delivery"],"A right shift can improve unloading but lower saturation when arterial PO₂ is on the steep curve. Delivery and extraction are related but distinct, so net benefit depends on conditions."),
           ("At fixed lung PO₂ on the plateau, Hb concentration falls while P₅₀ is unchanged. Which graph most directly reveals the lost reserve?","Oxygen content versus PO₂",["Saturation versus PO₂ alone","Pressure versus altitude alone","Temperature versus time alone"],"A saturation curve can appear unchanged because the fraction of occupied sites is preserved. A content curve shows fewer Hb sites per blood volume and a lower bound-oxygen plateau."),
           ("At the same saturation, how does PO₂ compare between otherwise similarly shaped left- and right-shifted curves?","A higher PO₂ is required on the right-shifted curve",["A lower PO₂ is required on the right-shifted curve","PO₂ must be identical at all saturations","The curves cannot have a 50% point"],"A right shift means reduced affinity and increased P₅₀. More oxygen pressure is needed to produce the same occupancy over the relevant curve range.")],
          [("Why should a Hill-model saturation not be presented as an exact replacement for measured co-oximetry?","It approximates a binding curve and does not identify dyshemoglobin fractions or all physiological modifiers",["It directly counts every COHb molecule","Its exponent measures total Hb concentration","It measures tissue blood flow"],"The Hill equation is useful for teaching relationships but does not incorporate all determinants of measured saturation. Actual blood measurements and Hb species may differ from the simple model."),
           ("A change in temperature shifts a curve, but Hb mass and saturation at a chosen point remain known. How should content be calculated?","Use the known Hb and appropriate saturation; do not add a separate P₅₀ multiplier to the content equation",["Multiply content by P₅₀ regardless of units","Replace Hb with temperature","Ignore saturation and use PO₂ alone"],"Affinity affects how saturation relates to PO₂. Once the appropriate saturation is given, the bound amount remains capacity×Hb×saturation; adding another affinity factor double-counts the effect."),
           ("What would a nearly horizontal saturation curve through the tissue range imply for unloading after a small pressure fall?","Little change in saturation, limiting the amount unloaded for that pressure change",["Complete unloading for every small pressure fall","A mandatory increase in Hb concentration","No dissolved oxygen diffusion"],"The local slope determines saturation change for a small PO₂ change. A steep region supports substantial unloading; a flat region gives a smaller occupancy change, other factors equal.")],
          [("During venous-to-arterial passage, CO₂ and H⁺ fall. Which oxygen-binding effect can help pulmonary loading?","A leftward affinity change through the Bohr effect",["A mandatory reduction in Hb binding sites","A right shift caused by lower H⁺ alone","A decrease in total atmospheric pressure"],"Lower CO₂ and H⁺ favour higher oxygen affinity. This complements the high lung oxygen pressure; the reverse changes in active tissues favour unloading."),
           ("Why is P₅₀ not the pressure at which half the body's oxygen delivery occurs?","It is a binding-curve point at 50% Hb saturation, not a blood-flow or delivery measurement",["It is defined as cardiac output divided by two","It includes no oxygen-binding information","It equals half of PB in every patient"],"P₅₀ measures affinity under stated conditions. Whole-body delivery requires arterial oxygen content and cardiac output and cannot be inferred from this single curve point."),
           ("A curve is shifted left, but tissue PO₂ falls sufficiently. Can oxygen still be unloaded?","Yes; reduced affinity-related unloading at a fixed PO₂ does not mean oxygen can never dissociate",["No; a left shift makes all oxygen binding irreversible","No; P₅₀ becomes infinite","Yes, because Hb concentration must double"],"A left shift is a change in reversible affinity, not irreversible binding. Lower tissue PO₂ can still drive dissociation; the amount depends on the full pressure and saturation differences.")],
        ]
        for group, rows in enumerate(curve_sets):
            for j,(p,c,w,e) in enumerate(rows): C(2,17+group*4+j,p,c,w,e)
        N(2,20,"For a supplied Hill model n=2.7, PO₂ is exactly twice P₅₀. Calculate saturation as a percentage using S = PO₂^n/(PO₂^n+P₅₀^n); both pressures are in kPa.","At PO₂/P₅₀=2: saturation (%) = 100×2^n/(2^n+1)","100*2**2.7/(2**2.7+1)","%",.4)
        N(2,24,"In a Hill model n=2.7 and P₅₀ 3.5 kPa, what PO₂ gives 75% saturation? Use PO₂ = P₅₀×[S/(1−S)]^(1/n).","Inverse Hill relation","3.5*(0.75/(1-0.75))**(1/2.7)",tolerance=.06)
        N(2,28,"Two similarly shaped Hill curves have P₅₀ values 3.0 and 4.0 kPa. At a fixed saturation of 70%, what is the ratio of required PO₂ on the 4.0 curve to that on the 3.0 curve?","At fixed S and n, PO₂ is proportional to P₅₀","4/3","ratio",.02)
        N(2,32,"In a supplied Hill model n=2, P₅₀ is 4 kPa and PO₂ 8 kPa. Calculate the deoxygenated fraction as a percentage using 1−S = P₅₀^n/(PO₂^n+P₅₀^n).","Unoccupied sites (%) = 100×P₅₀^n/(PO₂^n+P₅₀^n)","100*4**2/(8**2+4**2)","%",.3)
        N(2,36,"A model curve has P₅₀ 3.5 kPa and n=2.7. Calculate the saturation percentage at PO₂ 3.5 kPa without estimating from a plotted image.","When PO₂=P₅₀, the two denominator terms are equal","100*3.5**2.7/(3.5**2.7+3.5**2.7)","%",.2,note="P₅₀ is defined by this 50% point. The result does not depend on the chosen positive Hill coefficient.")
        N(2,40,"At identical tissue PO₂ two model curves yield saturations 78% and 62%. Hb is 13 g/dL. Neglect dissolved oxygen and calculate the difference in retained Hb-bound oxygen per dL, using capacity 1.34.","Retained-content difference = 1.34×Hb×saturation-fraction difference","1.34*13*(0.78-0.62)","mL O₂/dL",.08)
        # Objective 3: distinct integrated transport scenarios, all explicit units.
        N(3,17,"Hb is 110 g/L, SaO₂ 92%, PaO₂ 9 kPa and CO 4.2 L/min. Calculate DO₂, using 1.34 and solubility 0.0225 mL/dL/kPa.","DO₂ = [1.34×Hb(g/dL)×S+0.0225×PaO₂]×CO×10","(1.34*11*0.92+0.0225*9)*4.2*10","mL O₂/min",5.)
        N(3,18,"At CaO₂ 18 mL/dL, a patient has cardiac index 2.4 L/min/m² and body surface area 1.8 m². Calculate TOTAL oxygen delivery, not its index.","CO = cardiac index×BSA; DO₂ = CaO₂×CO×10","18*2.4*1.8*10","mL O₂/min",4.)
        N(3,19,"DO₂ is 900 mL/min and extraction rises from 25% to 35% at unchanged delivery. Calculate the increase in VO₂.","ΔVO₂ = DO₂×change in extraction fraction","900*(0.35-0.25)","mL O₂/min",2.)
        N(3,20,"VO₂ is 240 mL/min and oxygen extraction is 30% of delivery. At CaO₂ 16 mL/dL, calculate cardiac output.","DO₂ = VO₂/extraction; CO = DO₂/(CaO₂×10)","240/0.30/(16*10)","L/min",.06)
        N(3,21,"At steady state VO₂ is 250 mL/min and CaO₂ 20 mL/dL. CO falls from 5 to 4 L/min with VO₂ and CaO₂ unchanged. Calculate the fall in mixed venous content.","CvO₂ = CaO₂−VO₂/(CO×10); fall = old CvO₂−new CvO₂","(20-250/(5*10))-(20-250/(4*10))","mL O₂/dL",.06)
        N(3,22,"Initially CaO₂ is 18 mL/dL and CO 5 L/min. CaO₂ falls to 15 mL/dL. What absolute increase in CO preserves DO₂?","New CO = old content×old CO/new content; increase = new−old","18*5/15-5","L/min",.06)
        N(3,23,"Hb is 10 g/dL and SaO₂ 95%. Neglect dissolved oxygen. If CO is 6 L/min, calculate the DO₂ loss when Hb decreases by 2 g/dL at unchanged saturation and CO. Use capacity 1.34.","Lost DO₂ = 1.34×Hb fall×SaO₂×CO×10","1.34*2*0.95*6*10","mL O₂/min",3.)
        N(3,24,"VO₂ rises from 250 to 400 mL/min while DO₂ remains 1000 mL/min. Calculate the rise in extraction in percentage points.","Δextraction (percentage points) = 100×ΔVO₂/DO₂","100*(400-250)/1000","percentage points",.2)
        N(3,25,"Arterial content is 180 mL/L and mixed venous content 130 mL/L. VO₂ is 250 mL/min. Calculate CO, noting content is already per litre.","CO = VO₂/(CaO₂−CvO₂), when content is mL/L","250/(180-130)","L/min",.06,note="Do not add another factor of 10 when contents are already in mL/L; that would create a tenfold error.")
        N(3,26,"DO₂I is 500 mL/min/m² and body surface area is 1.9 m². VO₂ is 285 mL/min. Calculate whole-body extraction as a percentage.","Total DO₂ = DO₂I×BSA; extraction (%) = 100×VO₂/DO₂","100*285/(500*1.9)","%",.3)
        N(3,27,"Patient A has CaO₂ 14 mL/dL and CO 7 L/min; B has CaO₂ 20 mL/dL and CO 4 L/min. Calculate DO₂(A) minus DO₂(B).","Difference in delivery = 10×(contentA×flowA−contentB×flowB)","10*(14*7-20*4)","mL O₂/min",3.)
        N(3,28,"For a model with DO₂ 750 mL/min and VO₂ 300 mL/min, calculate the delivery-to-consumption ratio DO₂/VO₂, not the extraction fraction.","Delivery/consumption = DO₂÷VO₂","750/300","ratio",.03)
        N(3,29,"A patient has CaO₂ 17 mL/dL, CO 4.5 L/min and VO₂ 270 mL/min. Calculate mixed venous content from the Fick principle.","CvO₂ = CaO₂−VO₂/(CO×10)","17-270/(4.5*10)","mL O₂/dL",.08)
        N(3,30,"Hb is 120 g/L, SaO₂ 97%, SvO₂ 72% and CO 5 L/min. Neglect dissolved oxygen and calculate VO₂ using capacity 1.34.","VO₂ = 1.34×Hb(g/dL)×(SaO₂−SvO₂)×CO×10","1.34*12*(0.97-0.72)*5*10","mL O₂/min",4.)
        N(3,31,"A model initially has CaO₂ 20 mL/dL, CO 5 L/min and extraction 25%. Flow then falls to 3 L/min at unchanged content and VO₂. Calculate the required new extraction percentage.","Initial VO₂ = CaO₂×old CO×10×old extraction; new extraction = VO₂/new DO₂","100*(20*5*10*0.25)/(20*3*10)","%",.3)
        N(3,32,"At CO 4 L/min and SaO₂ 95%, Hb rises from 9 to 11 g/dL with PaO₂ fixed. Calculate the increase in DO₂ from Hb-bound oxygen using capacity 1.34.","ΔDO₂ = 1.34×ΔHb×SaO₂×CO×10","1.34*(11-9)*0.95*4*10","mL O₂/min",3.)
        N(3,33,"At Hb 150 g/L and CO 5 L/min, SaO₂ rises from 90% to 98%, while PaO₂ rises from 8 to 13 kPa. Calculate the total increase in DO₂, using capacity 1.34 and solubility 0.0225 mL/dL/kPa.","ΔDO₂ = [1.34×Hb(g/dL)×ΔS+0.0225×ΔPaO₂]×CO×10","(1.34*15*(0.98-0.90)+0.0225*(13-8))*5*10","mL O₂/min",3.)
        N(3,34,"CaO₂ is 19 mL/dL and CvO₂ 14 mL/dL. CO is 4.8 L/min. Calculate VO₂ from the content difference.","VO₂ = (CaO₂−CvO₂)×CO×10","(19-14)*4.8*10","mL O₂/min",3.)
        N(3,35,"In an illustrative exercise model, VO₂ is 1200 mL/min, CaO₂ 20 mL/dL and CvO₂ 8 mL/dL. Calculate CO by Fick.","CO = VO₂/[10×(CaO₂−CvO₂)]","1200/(10*(20-8))","L/min",.08)
        N(3,36,"DO₂ is 800 mL/min and CO 4 L/min. Calculate CaO₂ in mL/dL, then give the single final content value.","CaO₂ = DO₂/(CO×10)","800/(4*10)","mL O₂/dL",.08)
        N(3,37,"CaO₂ falls by 10% while cardiac output rises by 20%. Calculate the resulting percentage INCREASE in DO₂ relative to baseline.","DO₂ ratio = content ratio×flow ratio","100*(0.90*1.20-1)","%",.2)
        N(3,38,"A patient has VO₂ 225 mL/min and arterial–venous content difference 4.5 mL/dL. Calculate CO by Fick.","CO = VO₂/(content difference×10)","225/(4.5*10)","L/min",.06)
        N(3,39,"In a steady-state model with CaO₂ 18 mL/dL and CO 5 L/min, the specified maximum extraction is 45%. Calculate the largest VO₂ supported by those assumptions.","Model VO₂ ceiling = CaO₂×CO×10×specified extraction limit","18*5*10*0.45","mL O₂/min",4.,note="The extraction limit is stipulated for this model and is not a universal organ or whole-body threshold.")
        N(3,40,"Blood flow to an isolated organ is 0.8 L/min. Arterial content is 20 mL/dL and organ venous content 15 mL/dL. Calculate organ oxygen consumption, assuming steady state.","Organ VO₂ = organ flow×10×arterial–organ venous content difference","0.8*10*(20-15)","mL O₂/min",1.,note="Use the regional venous content for this organ, not an unrelated mixed venous sample.")
    return out


def pretty_expression(expression):
    return expression.replace("**", "^").replace("*", " × ").replace("/", " ÷ ")


def original_working(q):
    """Reconstruct substitutions from the displayed givens, also used to audit keys."""
    d = re.findall(r"[0-9]+(?:\.[0-9]+)?", q["prompt"])
    expressions = {
        "c2_1_01": lambda: f"0.21*({d[0]}-6.3)",
        "c2_1_02": lambda: f"{d[2]}*6.3",
        "c2_1_05": lambda: f"100*{d[0]}/{d[1]}",
        "c2_1_06": lambda: f"100*{d[1]}/({d[0]}-6.3)",
        "c2_1_09": lambda: f"{d[0]}/0.21+6.3",
        "c2_1_10": lambda: f"100*(101.3-{d[2]})/101.3",
        "c2_1_13": lambda: f"100-{d[2]}/0.35",
        "c2_1_14": lambda: f"100*6.3/{d[0]}",
        "c2_3_01": lambda: f"0.21*(101.3-6.3)-{d[3]}/0.8-{d[4]}",
        "c2_3_02": lambda: f"0.8*(0.21*95-{d[4]})",
        "c2_3_03": lambda: f"100*({d[4]}+4/0.8)/({d[0]}-6.3)",
        "c2_3_04": lambda: f"({d[0]}-{d[0]}/{d[1]})/0.8",
        "c2_3_05": lambda: f"0.21*({d[0]}-6.3)-{d[3]}/0.8-0.8",
        "c2_3_06": lambda: f"0.8*0.21*{d[0]}",
        "c2_3_07": lambda: f"{d[1]}*{d[0]}*(1/0.8-1)",
        "c2_3_08": lambda: f"{d[0]}*(1/0.8-1/1.0)",
        "c2_3_09": lambda: f"({d[3]}+{d[4]}+3.2/0.8)/0.21+6.3",
        "c2_3_10": lambda: f"{d[2]}*(101.3-6.3)",
        "c2_3_11": lambda: f"100*({d[5]}+2+6.4/0.8)/95",
        "c2_3_12": lambda: f"100*({d[5]}-4.8)/(0.8*95)",
        "c2_3_13": lambda: f"100*{d[1]}/6.3",
        "c2_3_14": lambda: f"{d[4]}/(0.21*(101.3-6.3)-{d[3]})",
        "c2_3_15": lambda: f"{d[5]}/(0.30*(101.3-6.3)-5.2/0.8)",
        "c2_3_16": lambda: f"0.21*(101.3-{d[2]})-(5.3-3.2)/0.8",
        "t2_1_01": lambda: f"1.34*({d[0]}/10)*({d[1]}/100)+0.0225*{d[2]}",
        "t2_1_02": lambda: f"100*(0.0225*{d[2]})/(1.34*{d[0]}+0.0225*{d[2]})",
        "t2_1_05": lambda: f"100*({d[0]}-0.0225*{d[2]})/(1.34*({d[1]}/10))",
        "t2_1_06": lambda: f"10*({d[0]}-0.0225*12)/(1.34*0.96)",
        "t2_1_09": lambda: f"1.34*{d[0]}*({d[2]}/100)",
        "t2_1_10": lambda: f"0.0225*{d[0]}",
        "t2_1_13": lambda: f"1.34*{d[0]}*({d[1]}/100)+0.0225*13",
        "t2_1_14": lambda: f"1.34*{d[0]}/0.0225",
        "t2_2_13": lambda: f"100*{d[3]}**2.7/({d[3]}**2.7+{d[4]}**2.7)",
        "t2_2_14": lambda: f"{d[2]}*((1-{d[1]}/100)/({d[1]}/100))**(1/2.7)",
        "t2_2_15": lambda: f"100*({d[0]}**2.7/({d[0]}**2.7+3.0**2.7)-{d[0]}**2.7/({d[0]}**2.7+4.2**2.7))",
        "t2_2_16": lambda: f"1.34*{d[2]}*({d[0]}/100-{d[1]}/100)",
        "t2_3_01": lambda: f"(1.34*{d[0]}*0.96+0.0225*12)*{d[3]}*10",
        "t2_3_02": lambda: f"5*(1.34*14*0.98+0.0225*13)/(1.34*{d[4]}*0.98+0.0225*13)",
        "t2_3_03": lambda: f"100*{d[2]}/({d[0]}*{d[1]}*10)",
        "t2_3_04": lambda: f"{d[0]}-{d[2]}/({d[1]}*10)",
        "t2_3_05": lambda: f"{d[0]}/(10*({d[1]}-{d[2]}))",
        "t2_3_06": lambda: f"100*(0.98-{d[3]}/(10*{d[2]}*1.34*{d[0]}))",
        "t2_3_07": lambda: f"10*({d[0]}/({d[1]}*10)-0.0225*10)/(1.34*0.95)",
        "t2_3_08": lambda: f"100*(1-(1-{d[0]}/100)*1.10)",
        "t2_3_09": lambda: f"0.0225*{d[1]}*{d[0]}*10",
        "t2_3_10": lambda: f"{d[0]}*{d[1]}*10",
        "t2_3_11": lambda: f"{d[0]}/({d[1]}*10*{d[2]})",
        "t2_3_12": lambda: f"{d[2]}/({d[1]}*10)",
        "t2_3_13": lambda: f"100*({d[0]}-{d[1]})/{d[0]}",
        "t2_3_14": lambda: f"1.34*{d[0]}*(1-{d[2]}/100)*{d[1]}*10",
        "t2_3_15": lambda: "100*(1/0.80-1)",
        "t2_3_16": lambda: f"(1.34*{d[0]}*(0.98-0.68)+0.0225*(13-5))*{d[5]}*10",
    }
    expression = expressions[q["id"]]()
    calculated = arithmetic(expression)
    if abs(calculated-q["answer"]) > q["tolerance"]*.2:
        raise ValueError("Answer key does not match the displayed question")
    q["working_expression"] = expression
    convention = "Use saturation as a fraction and Hb in the units requested." if q["id"].startswith("t2") else "Use kPa consistently for every partial pressure."
    return f"Use this relationship: {q['explanation']}\nSubstitute the stated values: {pretty_expression(expression)} = {calculated:.6g} {q['unit']}.\n{convention} Round only the final answer."


def feedback_text(item):
    unit = item.get("unit", "")
    expected = item["expected"]
    expected_text = f"{expected:.6g} {unit}" if isinstance(expected, (int, float)) else str(expected)
    parts = [f"Question: {item['prompt']}", f"Your answer: {item['response']}",
             f"Correct answer: {expected_text}", f"Mark: {item['earned']}/1"]
    if item.get("working"):
        parts += ["Worked solution:", item["working"]]
    parts += ["Reasoning and interpretation:", item["explanation"]]
    if item.get("tolerance") is not None:
        parts += [f"Accepted numerical tolerance: ±{item['tolerance']:g} {unit}. Round at the end."]
        if unit in ("%", "percentage points"):
            parts += ["A percentage point change is not a relative percentage change; follow the quantity requested in the question."]
    return "\n\n".join(parts)


def feedback_report(records):
    sections = []
    for record in records:
        sections.append(f"OXYGEN LAB — {record['learner_code']} — {TOPICS[record['topic']]} — paper {record.get('form','Legacy')}\nScore {record['score']}/{record['total']}\nAll new assessment pressures are in kPa.")
        sections.extend(feedback_text(item) for item in record["items"])
    return "\n\n" + ("\n\n" + "─"*50 + "\n\n").join(sections)

def learner_form_index(learner):
    """One allocated paper for each code S01–S10."""
    match = re.fullmatch(r"S(0[1-9]|10)", learner.strip().upper())
    if not match:
        raise ValueError("Use your allocated candidate code S01–S10")
    return int(match.group(1))-1


def make_questions(topic, seed, unit="kPa", form=None):
    """Ten disjoint 12-item forms, each with four items per objective."""
    if unit != "kPa":
        raise ValueError("All new assessments use kPa only")
    form = seed % 10 if form is None else form
    if form not in range(10):
        raise ValueError("Unknown assessment form")
    bank = question_bank(topic, seed)
    selected = []
    for obj in (1, 2, 3):
        pool = sorted((q for q in bank if q["objective"] == obj), key=lambda q: q["id"])
        slots = ((1,4,9,12), (2,3,10,11), (5,8,13,16), (6,7,14,15)) if obj == 1 else tuple(tuple(range(i,17,4)) for i in range(1,5))
        if form < 4:
            selected.extend(pool[n-1] for n in slots[form])
        else:
            selected.extend(pool[16+(form-4)*4:20+(form-4)*4])
    rng = random.Random(seed)
    for item in selected:
        if item["type"] == "mcq":
            rng.shuffle(item["options"])
    rng.shuffle(selected)
    return selected


LEGACY_IDS = {
    "cascade": {**{k: 1 for k in ("c_order", "c_air", "c_inspired", "c_baseline")},
                **{k: 2 for k in ("c_water", "c_alveoli", "c_gradient", "c_tissue")},
                **{k: 3 for k in ("c_altitude", "c_vent", "c_high", "c_oxygen")}},
    "transport": {**{k: 1 for k in ("t_pressure", "t_anaemia", "t_dissolved")},
                  **{k: 2 for k in ("t_cooperative", "t_shift", "t_factors", "t_plateau", "t_p50", "t_anchor")},
                  **{k: 3 for k in ("t_units", "t_content", "t_delivery")}},
}



def grade(question, response):
    if question["type"] == "mcq":
        return int(response == question["answer"])
    v = parse_number(response)
    return int(v is not None and abs(v-question["answer"]) <= question["tolerance"] + 1e-9)


def csv_text(rows):
    out = io.StringIO()
    if rows:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        # Protect spreadsheet opening from formula injection in learner text.
        for row in rows:
            writer.writerow({k: ("'"+v if isinstance(v,str) and v.lstrip().startswith(("=","+","-","@")) else v) for k,v in row.items()})
    return out.getvalue()


def summary_row(record):
    return {**{k:record[k] for k in ("learner_code","topic","attempt_id","submitted_utc","score","total","percentage","duration_seconds","pressure_unit")}, "paper": record.get("form", "Legacy"), "question_bank": record.get("bank_version", "Original")}


def validate_report(payload):
    """Strictly validate portable reports before teacher aggregation."""
    if not isinstance(payload,dict) or payload.get("schema_version") != 1:
        raise ValueError("Not an Oxygen Lab version 1 report")
    records = payload.get("attempts")
    if not isinstance(records,list) or not 1 <= len(records) <= 200:
        raise ValueError("Report must contain 1–200 attempts")
    for r in records:
        if not isinstance(r,dict) or r.get("topic") not in TOPICS:
            raise ValueError("Unknown topic")
        uuid.UUID(r["attempt_id"])
        if not isinstance(r.get("learner_code"),str) or not 1 <= len(r["learner_code"]) <= 80:
            raise ValueError("Invalid learner code")
        datetime.fromisoformat(r["submitted_utc"])
        if r.get("pressure_unit") not in ("kPa","mmHg"):
            raise ValueError("Invalid pressure unit")
        if not isinstance(r.get("items"),list) or len(r["items"]) != 12:
            raise ValueError("Expected 12 marked items")
        if r.get("bank_version") in (BANK_VERSION, "advanced-kpa-v2"):
            if r.get("pressure_unit") != "kPa" or r.get("form") not in tuple(FORMS):
                raise ValueError("Invalid advanced paper or unit")
            expected = {q["id"]: q["objective"] for q in make_questions(r["topic"], 0, form=FORMS.index(r["form"]))}
        elif "bank_version" not in r:
            expected = LEGACY_IDS[r["topic"]]
        else:
            raise ValueError("Unknown question-bank version")
        if {x.get("id") for x in r["items"]} != set(expected):
            raise ValueError("Question IDs are missing or duplicated")
        if any(type(x.get("earned")) is not int or x["earned"] not in (0,1) or x.get("objective") != expected[x["id"]] for x in r["items"]):
            raise ValueError("Invalid item marks or objectives")
        score = sum(x["earned"] for x in r["items"])
        if r.get("score") != score or r.get("total") != 12 or r.get("percentage") != round(100*score/12,1):
            raise ValueError("Inconsistent score")
        if not isinstance(r.get("duration_seconds"),(int,float)) or not math.isfinite(r["duration_seconds"]) or r["duration_seconds"] < 0:
            raise ValueError("Invalid duration")
    return records


def layout(fig, ytitle, xtitle=None, height=430):
    fig.update_layout(template="plotly_white", height=height,
                      margin=dict(l=35,r=20,t=30,b=45),
                      font=dict(family="Arial",size=14),
                      yaxis_title=ytitle, xaxis_title=xtitle,
                      legend=dict(orientation="h",y=1.15,x=0))
    return fig


def cascade_plot(unit, values, reference=None):
    labels = ["Dry air","Humidified gas","Alveoli","Arterial blood","Tissue*","Mitochondria*"]
    fig = go.Figure()
    if reference:
        fig.add_trace(go.Scatter(x=labels,y=[from_mmhg(x,unit) for x in reference],mode="lines+markers",name="Sea-level reference",line=dict(color="#909DA8",dash="dash",shape="hv")))
    vals = [from_mmhg(x,unit) for x in values]
    fig.add_trace(go.Scatter(x=labels,y=vals,mode="lines+markers+text",name="Selected conditions",text=[f"{x:.1f}" for x in vals],textposition="top center",line=dict(color=BLUE,width=3,shape="hv"),marker=dict(size=9)))
    fig.update_yaxes(rangemode="tozero")
    return layout(fig,f"PO₂ ({unit})")


def cascade_lab(unit):
    st.subheader("Follow the pressure gradient")
    st.caption("Healthy adult reference: rest, room air, sea level, 37°C, R = 0.8. Change one factor and predict before reading the graph.")
    presets = {"Sea-level reference":(.21,0,40,5),"Hypoventilation":(.21,0,60,5),"Altitude, fixed PaCO₂":(.21,2500,40,5),"Altitude with hyperventilation":(.21,2500,30,5),"Supplemental oxygen":(.40,0,40,5),"Widened A–a gradient":(.21,0,40,30)}
    preset = st.selectbox("Scenario",list(presets)+["Custom"],key="c_preset")
    fio, alt, co2, aa = presets.get(preset,presets["Sea-level reference"])
    if preset == "Custom":
        left,right = st.columns(2)
        with left:
            fio = st.slider("Inspired oxygen (FiO₂)",.21,1.,.21,.01,key="c_fio")
            alt = st.slider("Altitude (m)",0,5000,0,100,key="c_alt")
        with right:
            co2 = to_mmhg(st.slider(f"PaCO₂ ({unit})",2.0 if unit=="kPa" else 15.,12.0 if unit=="kPa" else 90.,5.3 if unit=="kPa" else 40.,.1 if unit=="kPa" else 1.,key=f"c_co2_{unit}"),unit)
            aa = to_mmhg(st.slider(f"Assumed A–a gradient ({unit})",0.,8. if unit=="kPa" else 60.,.7 if unit=="kPa" else 5.,.1 if unit=="kPa" else 1.,key=f"c_aa_{unit}"),unit)
    rq = st.slider("Respiratory quotient (R)",.7,1.,.8,.05,key="c_rq")
    pb = barometric_pressure(alt)
    pio, pao = fio*(pb-47), alveolar_po2(fio,pb,co2,rq)
    pa = pao-aa
    st.write(f"FiO₂ **{fio:.2f}** · altitude **{alt} m** · estimated PB **{from_mmhg(pb,unit):.1f} {unit}** · PaCO₂ **{from_mmhg(co2,unit):.1f} {unit}**")
    st.latex(r"P_{A O_2}\approx F_{I O_2}(P_B-P_{H_2O})-\frac{P_{a CO_2}}{R}")
    st.write(f"PAO₂ ≈ {fio:.2f} × ({from_mmhg(pb,unit):.2f} − {from_mmhg(47,unit):.2f}) − {from_mmhg(co2,unit):.2f}/{rq:.2f} = **{from_mmhg(pao,unit):.2f} {unit}**")
    if pao <= 0 or pa <= 0:
        st.error("These inputs yield a non-positive alveolar or arterial estimate. They cannot represent the assumed physiological steady state. Change the inputs; no cascade is plotted.")
        return
    reference = [.21*760,.21*713,alveolar_po2(.21),alveolar_po2(.21)-5,20,2]
    # Tissue is an illustrative node, only drawn if below the selected arterial PO2.
    tissue = min(20., pa*.5)
    mito = min(2., tissue*.25)
    st.plotly_chart(cascade_plot(unit,[fio*pb,pio,pao,pa,tissue,mito],reference),width="stretch")
    st.caption("*Tissue 2.7 kPa and mitochondrial 0.27 kPa are illustrative resting teaching points. At low arterial estimates these nodes are reduced only to preserve the schematic gradient; they are NOT a calculated tissue oxygenation model. Tissue/mitochondrial PO₂ varies widely.")
    st.info("Mixed venous PO₂ ≈ 5.3 kPa is a resting reference after extraction. Venous blood returns from tissues; it is not a serial diffusion step between arterial blood and mitochondria.")
    st.markdown("**Explain the falls**\n\n- **Humidification:** water vapour takes a share of total pressure.\n- **Alveoli:** oxygen uptake and carbon dioxide entry; replenishment depends on ventilation.\n- **Arterial blood:** V/Q inequality and physiological venous admixture; disease may add diffusion limitation.\n- **Tissues:** diffusion plus extraction and mitochondrial consumption maintain the gradient.")
    st.caption("The selected A–a gradient is an assumption, not a prediction of the gas equation. It varies with age, FiO₂ and disease. PaCO₂ approximates PACO₂. High FiO₂ reduces the accuracy of the simplified equation.")
    if fio > .4:
        full = fio*(pb-47)-co2*(fio+(1-fio)/rq)
        st.write(f"For comparison, a fuller steady-state equation gives PAO₂ ≈ **{from_mmhg(full,unit):.2f} {unit}**; the main plot deliberately uses the requested simplified equation.")


def transport_lab(unit):
    st.subheader("Pressure is not the amount carried")
    st.markdown("**PO₂:** oxygen tension · **SO₂:** occupied Hb binding sites · **Content:** oxygen per blood volume · **Delivery:** oxygen per minute.")
    st.markdown("**1 · Move the dissociation curve**")
    shifts = {"Normal adult":26.6,"Left shift":20.,"Right shift":33.,"Custom P₅₀":26.6}
    choice = st.selectbox("Curve",list(shifts),key="t_shift")
    p50 = shifts[choice]
    if choice == "Custom P₅₀":
        p50 = to_mmhg(st.slider(f"P₅₀ ({unit})",2. if unit=="kPa" else 15.,6. if unit=="kPa" else 45.,3.5 if unit=="kPa" else 26.6,.1,key=f"t_p50_{unit}"),unit)
    probe = to_mmhg(st.slider(f"Inspect PO₂ ({unit})",0.,20. if unit=="kPa" else 150.,5.3 if unit=="kPa" else 40.,.1 if unit=="kPa" else 1.,key=f"t_probe_{unit}"),unit)
    x = [i*.5 for i in range(301)]
    fig = go.Figure()
    for label,p,c in [("Left: P₅₀ 2.7 kPa",20.,TEAL),("Normal: P₅₀ 3.5 kPa",26.6,BLUE),("Right: P₅₀ 4.4 kPa",33.,ORANGE)]:
        fig.add_trace(go.Scatter(x=[from_mmhg(v,unit) for v in x],y=[100*saturation(v,p) for v in x],name=label,mode="lines",line=dict(color=c,width=2)))
    if choice == "Custom P₅₀":
        fig.add_trace(go.Scatter(x=[from_mmhg(v,unit) for v in x],y=[100*saturation(v,p50) for v in x],name="Custom",line=dict(color="#5938A0",dash="dash")))
    fig.add_trace(go.Scatter(x=[from_mmhg(probe,unit)],y=[100*saturation(probe,p50)],name="Selected point",mode="markers",marker=dict(size=12,color="black")))
    fig.add_hline(y=50,line_dash="dot",line_color="#888")
    fig.add_vline(x=from_mmhg(p50,unit),line_dash="dot",line_color="#888")
    fig.update_yaxes(range=[0,102])
    st.plotly_chart(layout(fig,"Saturation (%)",f"PO₂ ({unit})"),width="stretch")
    a,b = st.columns(2)
    a.metric("Selected P₅₀",f"{from_mmhg(p50,unit):.2f} {unit}")
    b.metric("Model saturation at probe",f"{100*saturation(probe,p50):.1f}%")
    st.caption("Hill approximation: S = PO₂ⁿ/(PO₂ⁿ + P₅₀ⁿ), n = 2.7. This is a teaching curve, not a validated estimator of measured SaO₂. Shifts are qualitative P₅₀ illustrations, not calculations from a patient's pH or temperature.")
    st.markdown("- **Sigmoid:** cooperative binding. **Plateau:** supports pulmonary loading. **Steep part:** supports tissue unloading.\n- **Right:** ↑ temperature, ↑ CO₂, ↓ pH, ↑ 2,3-DPG → ↑ P₅₀, lower affinity, more unloading at the same PO₂.\n- **Left:** converse factors; fetal Hb → ↓ P₅₀, higher affinity, less unloading at the same PO₂.")
    st.markdown("**2 · Calculate oxygen content and delivery**")
    scenario = st.selectbox("Delivery scenario",["Normal resting example","Anaemia","Low cardiac output","Hypoxaemia","Hyperoxia","Custom"],key="t_case")
    cases = {"Normal resting example":(15.,98.,100.,5.),"Anaemia":(7.5,98.,100.,5.),"Low cardiac output":(15.,98.,100.,2.5),"Hypoxaemia":(15.,80.,50.,5.),"Hyperoxia":(15.,100.,500.,5.)}
    hb,sa,pa,co = cases.get(scenario,cases["Normal resting example"])
    if scenario == "Custom":
        a,b = st.columns(2)
        with a:
            hb = st.slider("Haemoglobin (g/dL)",3.,20.,15.,.5,key="t_hb")
            co = st.slider("Cardiac output (L/min)",1.,30.,5.,.5,key="t_co")
        with b:
            pa = to_mmhg(st.slider(f"PaO₂ ({unit})",1. if unit=="kPa" else 7.5,80. if unit=="kPa" else 600.,13.3 if unit=="kPa" else 100.,.1 if unit=="kPa" else 1.,key=f"t_pa_{unit}"),unit)
            linked = st.checkbox("Use the selected teaching curve to estimate saturation",key="t_link")
            if linked:
                sa = 100*saturation(pa,p50)
                st.write(f"Estimated saturation: {sa:.1f}%")
            else:
                sa = st.slider("Measured/assumed SaO₂ (%)",30.,100.,98.,1.,key="t_sa")
                st.caption("SaO₂ is entered independently; confirm its consistency with PaO₂ and conditions. Dyshemoglobins are not modelled.")
    content = oxygen_content(hb,sa/100,pa)
    delivery = oxygen_delivery(content,co)
    bound,dissolved = 1.34*hb*sa/100,.003*pa
    st.write(f"Hb **{hb:g} g/dL** · SaO₂ **{sa:g}%** · PaO₂ **{from_mmhg(pa,unit):.1f} {unit}** · CO **{co:g} L/min**")
    st.latex(r"C_{a O_2}=1.34\times Hb\times\frac{S_{a O_2}(\%)}{100}+0.0225\times P_{a O_2}\;(kPa)")
    st.caption("CaO₂ in mL O₂/dL; Hb in g/dL; PaO₂ in kPa. Dissolved coefficient ≈0.0225 mL O₂/dL/kPa. Divide Hb in g/L by 10; divide saturation percent by 100.")
    st.write(f"CaO₂ = 1.34 × {hb:g} × {sa/100:.3f} + 0.0225 × {from_mmhg(pa,unit):.2f} = **{content:.2f} mL O₂/dL**")
    st.write(f"DO₂ = {content:.2f} × {co:g} × 10 = **{delivery:.0f} mL O₂/min**")
    a,b,c = st.columns(3)
    a.metric("Hb-bound oxygen",f"{bound:.2f} mL/dL")
    b.metric("Dissolved oxygen",f"{dissolved:.2f} mL/dL")
    c.metric("Delivery",f"{delivery:.0f} mL/min")
    fig = go.Figure()
    for label,h,s,p,flow in [("Reference",15.,.98,100.,5.),("Selected",hb,sa/100,pa,co)]:
        fig.add_trace(go.Bar(x=[label],y=[1.34*h*s*flow*10],name=f"{label}: Hb bound",marker_color=BLUE))
        fig.add_trace(go.Bar(x=[label],y=[.003*p*flow*10],name=f"{label}: dissolved",marker_color=TEAL))
    fig.update_layout(barmode="stack")
    st.plotly_chart(layout(fig,"DO₂ (mL O₂/min)",height=340),width="stretch")
    st.info("A normal PaO₂ or saturation can coexist with poor delivery when Hb or cardiac output is low. Raising PaO₂ adds relatively little content once Hb is nearly fully saturated. DO₂ is delivery, not VO₂ consumption.")


CANVAS = """<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{font:14px Arial;color:#24364b;margin:4px}canvas{width:100%;height:290px;border:1px solid #bdcbd9;background:white;touch-action:none;border-radius:8px}button{padding:9px 16px;background:#005eb8;color:white;border:0;border-radius:6px;margin:6px 8px 0 0;cursor:pointer}button:focus{outline:3px solid #f3a626}</style></head><body>
<canvas id="pad" width="1000" height="390" aria-label="Draw your graph with a mouse or touchscreen"></canvas>
<button id="clear">Clear drawing</button><button id="save">Download sketch PNG</button>
<span>Draw with your mouse or finger. Label both axes and your reference conditions.</span>
<script>const c=document.getElementById('pad'),x=c.getContext('2d');let down=false;
function blank(){x.fillStyle='white';x.fillRect(0,0,c.width,c.height);x.strokeStyle='#536678';x.lineWidth=2;x.beginPath();x.moveTo(60,20);x.lineTo(60,340);x.lineTo(970,340);x.stroke()};blank();
function point(e){const r=c.getBoundingClientRect();return[(e.clientX-r.left)*c.width/r.width,(e.clientY-r.top)*c.height/r.height]}
c.onpointerdown=e=>{down=true;c.setPointerCapture(e.pointerId);let p=point(e);x.beginPath();x.moveTo(...p);x.strokeStyle='#005eb8';x.lineWidth=3;x.lineCap='round';e.preventDefault()};
c.onpointermove=e=>{if(down){x.lineTo(...point(e));x.stroke();e.preventDefault()}};c.onpointerup=c.onpointercancel=()=>{down=false};
document.getElementById('clear').onclick=blank;document.getElementById('save').onclick=()=>{let a=document.createElement('a');a.download='oxygen-sketch.png';a.href=c.toDataURL('image/png');a.click()};</script></body></html>"""


def sketch_viva(topic):
    st.subheader("Sketch it, then explain it")
    st.caption("This task requires tutor review; it is not included in the automatic quiz score. The canvas is temporary—download your sketch before leaving this page.")
    if topic == "cascade":
        st.write("Draw the cascade for a healthy adult at rest on room air at sea level. Label approximate values, state your units and conditions, and explain each fall. Add mixed venous PO₂ as a separate returning-blood reference.")
        rubric = ["State room air, sea level, rest and 37°C","Label axes and units; order air → humidified → alveoli → arterial → tissue → mitochondria","Air ≈21.3, humidified ≈20.0, alveolar ≈13.3, arterial ≈12.6 kPa","Recognise mixed venous ≈5.3 kPa; tissue/mitochondrial values vary","Explain water vapour, alveolar exchange, A–a gradient and tissue extraction","Write the gas equation and predict hypoventilation, altitude and FiO₂ effects"]
        viva = ["On room air, why does 6.3 kPa water vapour reduce inspired PO₂ by only about 1.3 kPa?","Why does hypoventilation lower PAO₂ without necessarily widening the A–a gradient?","Which assumptions must be stated before quoting a mitochondrial PO₂?"]
    else:
        st.write("Sketch the normal adult dissociation curve, mark P₅₀ and the 5.3/8.0/13.3 kPa anchor points, then overlay a left and a right shift. Explain loading and unloading. Use the second calculation to explain why normal saturation can be misleading in anaemia.")
        rubric = ["PO₂ on x-axis, saturation (%) on y-axis; sigmoid curve","P₅₀ ≈3.5–3.6 kPa at 50%","Anchors in kPa: 5.3≈75%, 8.0≈90%, 13.3≈97–98%; steep and plateau regions","Explain cooperative binding","Right shift: increased P₅₀, lower affinity; list heat/CO₂/H⁺/2,3-DPG","Distinguish pressure/saturation/content and calculate CaO₂ and DO₂ with correct units"]
        viva = ["Can SaO₂ be 98% while oxygen delivery is very low?","Why is the ×10 needed in the delivery equation?","Does a right shift always improve whole-body oxygen delivery at altitude?"]
    components.html(CANVAS,height=390)
    with st.expander("Tutor/self-review checklist and viva prompts"):
        for i,item in enumerate(rubric):
            st.checkbox(item,key=f"rubric_{topic}_{i}")
        for question in viva:
            st.write("• "+question)
    st.text_area("Your explanation or tutor feedback (optional, temporary)",key=f"viva_{topic}")
    notes = st.session_state.get(f"viva_{topic}","")
    st.download_button("Download viva notes",notes,file_name=f"{topic}_viva_notes.txt",mime="text/plain")


def assessment(topic, unit, learner):
    st.subheader("Individual formative assessment")
    st.caption("Advanced applied physiology · 12 items · 1 mark each · four questions per objective. Calculator allowed. All pressures use kPa. Numerical tolerances are shown. These are formative papers, not validated equivalent examinations.")
    key = f"attempt_{topic}"
    attempt = st.session_state.get(key)
    if attempt and attempt.get("bank_version") != BANK_VERSION:
        if not attempt.get("result"):
            st.info("The question bank has been updated. Please start a new kPa assessment.")
        del st.session_state[key]
        attempt = None
    try:
        form_index = learner_form_index(learner)
        valid_code = True
    except ValueError:
        form_index, valid_code = 0, False
    if not attempt:
        st.caption("Enter your allocated code S01–S10. Each candidate receives a separate paper. Complete both topics, download your results and send them privately to the tutor. Wait for your individual 15-minute viva call.")
        if st.button("Start assessment",type="primary",key=f"start_{topic}"):
            if not valid_code:
                st.error("Enter your allocated candidate code S01–S10 in the sidebar first.")
                return
            attempt = {"id":str(uuid.uuid4()),"learner":learner.strip().upper(),"unit":"kPa","started":time.time(),"questions":make_questions(topic,uuid.uuid4().int,form=form_index),"result":None,"bank_version":BANK_VERSION,"form":FORMS[form_index]}
            st.session_state[key] = attempt
            st.rerun()
        return
    st.write(f"Learner: **{attempt['learner']}** · paper **{attempt['form']}** · pressure unit: **kPa**")
    if not attempt["result"]:
        answers = {}
        with st.form(f"quiz_{attempt['id']}"):
            for i,q in enumerate(attempt["questions"],1):
                st.markdown(f"**{i}. {q['prompt']}**")
                if q["type"] == "mcq":
                    answers[q["id"]] = st.radio("Choose one",q["options"],index=None,key=f"{attempt['id']}_{q['id']}",label_visibility="collapsed")
                else:
                    answers[q["id"]] = st.text_input(f"Answer ({q['unit']})",key=f"{attempt['id']}_{q['id']}",help=f"Accepted tolerance ±{q['tolerance']:g} {q['unit']}. Enter a number only, with a decimal point.")
            submitted = st.form_submit_button("Submit and show feedback",type="primary")
        if submitted:
            missing = [q for q in attempt["questions"] if answers[q["id"]] is None or str(answers[q["id"]]).strip()==""]
            invalid = [q for q in attempt["questions"] if q["type"]=="number" and str(answers[q["id"]]).strip() and parse_number(answers[q["id"]]) is None]
            if missing or invalid:
                st.error(f"Complete all items with valid answers before submitting. Missing: {len(missing)}; invalid numerical answers: {len(invalid)}.")
                return
            items = []
            for q in attempt["questions"]:
                response = answers[q["id"]]
                items.append(dict(id=q["id"],objective=q["objective"],prompt=q["prompt"],response=response,expected=q["answer"],earned=grade(q,response),explanation=q["explanation"],working=q.get("working",""),unit=q.get("unit",""),tolerance=q.get("tolerance"),type=q["type"]))
            score = sum(x["earned"] for x in items)
            result = dict(learner_code=attempt["learner"],topic=topic,attempt_id=attempt["id"],submitted_utc=datetime.now(timezone.utc).isoformat(),score=score,total=len(items),percentage=round(100*score/len(items),1),duration_seconds=round(time.time()-attempt["started"],1),pressure_unit="kPa",items=items,bank_version=BANK_VERSION,form=attempt["form"])
            attempt["result"] = result
            st.session_state.setdefault("results",[]).append(result)
            st.rerun()
        return
    result = attempt["result"]
    st.success(f"{result['score']}/{result['total']} — {result['percentage']:.1f}%")
    st.caption("Use objective-level feedback to choose revision. No validated pass/fail threshold is applied.")
    breakdown = []
    for obj,label in OBJECTIVES[topic].items():
        subset = [x for x in result["items"] if x["objective"]==obj]
        breakdown.append({"Objective":label,"Score":f"{sum(x['earned'] for x in subset)}/{len(subset)}"})
    st.table(breakdown)
    for i,item in enumerate(result["items"],1):
        with st.expander(f"{i}. {'✓' if item['earned'] else 'Review'} — {item['prompt']}"):
            st.markdown(feedback_text(item))
    st.download_button("Download this attempt (JSON)",json.dumps({"schema_version":1,"attempts":[result]},ensure_ascii=False,indent=2),file_name=f"oxygen_{topic}_{attempt['id'][:8]}.json",mime="application/json")
    st.download_button("Download worked feedback (TXT)", feedback_report([result]), file_name=f"{result['learner_code']}_{topic}_feedback.txt", mime="text/plain")
    st.info("This paper is complete. Download your report, complete the other topic, then wait for your tutor to call you for the 15-minute viva. Answers are frozen for this session.")


def reports():
    st.header("Your results")
    records = st.session_state.get("results",[])
    if not records:
        st.info("Complete an assessment to create your individual report.")
        return
    rows = [summary_row(x) for x in records]
    st.dataframe(rows,hide_index=True,width="stretch")
    st.download_button("Download all my results (JSON)",json.dumps({"schema_version":1,"attempts":records},ensure_ascii=False,indent=2),file_name="oxygen_individual_results.json",mime="application/json")
    st.download_button("Download my scores (CSV)",csv_text(rows),file_name="oxygen_individual_scores.csv",mime="text/csv")
    st.download_button("Download all worked feedback (TXT)", feedback_report(records), file_name="oxygen_worked_feedback.txt", mime="text/plain")
    st.info("Send the JSON results file privately to your tutor before your viva. Keep it downloaded before refreshing or closing. There is no automatic central collection.")


def tutor_dashboard():
    st.header("Tutor review")
    st.write("Ask each learner to download their results JSON and submit it through your usual teaching channel. Upload those files here to compare results and identify learning needs.")
    st.caption("Files are self-reported and unauthenticated. This panel reads only files you upload; it cannot access other learners' live sessions.")
    uploads = st.file_uploader("Upload individual Oxygen Lab JSON reports",type=["json"],accept_multiple_files=True)
    seen = {}
    for f in uploads:
        try:
            if f.size > 2_000_000:
                raise ValueError("File exceeds 2 MB")
            for r in validate_report(json.loads(f.getvalue())):
                if r["attempt_id"] in seen and seen[r["attempt_id"]] != r:
                    raise ValueError("Conflicting duplicate attempt ID")
                seen[r["attempt_id"]] = r
        except (ValueError,KeyError,TypeError,AttributeError,OverflowError) as e:
            st.error(f"{f.name}: {e}")
    if not seen:
        return
    records = list(seen.values())
    st.caption("Duplicate identical attempts are removed. Retakes are listed separately.")
    st.dataframe([summary_row(x) for x in records],hide_index=True,width="stretch")
    by_learner = defaultdict(list)
    for r in records:
        by_learner[(r["learner_code"],r["topic"])].append(r)
    latest = [max(group,key=lambda x:datetime.fromisoformat(x["submitted_utc"]).timestamp()) for group in by_learner.values()]
    st.subheader("Latest attempt for each learner and topic")
    st.dataframe([summary_row(x) for x in latest],hide_index=True,width="stretch")
    objective_rows = []
    for topic in TOPICS:
        for obj,label in OBJECTIVES[topic].items():
            subset = [item for r in latest if r["topic"]==topic for item in r["items"] if item["objective"]==obj]
            if subset:
                objective_rows.append({"Topic":TOPICS[topic],"Objective":label,"Marks earned":sum(x["earned"] for x in subset),"Marks available":len(subset),"Percent":round(100*sum(x["earned"] for x in subset)/len(subset),1)})
    st.subheader("Where the group needs revision")
    st.dataframe(objective_rows,hide_index=True,width="stretch")
    with st.expander("Individual submitted answers and worked feedback"):
        chosen = st.selectbox("Choose submitted report", list(range(len(records))), format_func=lambda i: f"{records[i]['learner_code']} · {TOPICS[records[i]['topic']]} · paper {records[i].get('form','Legacy')} · {records[i]['submitted_utc']}")
        st.markdown(feedback_report([records[chosen]]))
        st.download_button("Download this learner's worked feedback", feedback_report([records[chosen]]), file_name=f"{records[chosen]['learner_code']}_worked_feedback.txt", mime="text/plain")
    st.download_button("Download latest group scores",csv_text([summary_row(x) for x in latest]),file_name="oxygen_group_latest_scores.csv",mime="text/csv")
    st.download_button("Download objective summary",csv_text(objective_rows),file_name="oxygen_group_objectives.csv",mime="text/csv")


def lesson(topic):
    st.header(TOPICS[topic]+" · 15 minutes")
    rows = [("0–3 min","Draw and label typical values"),("3–6 min","Explain the fall at each stage"),("6–10 min","Gas equation and worked example"),("10–15 min","Individual assessment; sketch/viva discussion")] if topic=="cascade" else [("0–2 min","Pressure, saturation and content"),("2–6 min","Dissociation curve, P₅₀ and shifts"),("6–10 min","Content and delivery worked example"),("10–15 min","Individual assessment; sketch/viva discussion")]
    with st.expander("Learning objectives and teaching sequence"):
        for i,label in OBJECTIVES[topic].items():
            st.write(f"{i}. {label}")
        st.table([{"Time":a,"Focus":b} for a,b in rows])
        st.caption("Use a few selected questions live; the full 12-item test can be completed after teaching if five minutes is insufficient.")


def references():
    st.header("Equations, assumptions and sources")
    st.markdown("""- All displayed pressures and new assessments use **kPa only**.
- Simplified alveolar gas equation: **PAO₂ ≈ FiO₂ × (PB − PH₂O) − PaCO₂/R**, with all pressures in kPa. At 37°C use PH₂O ≈6.3 kPa; assume steady state, negligible inspired CO₂ and PACO₂ ≈ PaCO₂. At high FiO₂ the fuller equation differs: **PAO₂ = FiO₂(PB−PH₂O) − PaCO₂[FiO₂+(1−FiO₂)/R]**.
- CaO₂ = **1.34 × Hb (g/dL) × SaO₂ (fraction) + 0.0225 × PaO₂ (kPa)**, yielding mL O₂/dL.
- DO₂ = **CaO₂ (mL/dL) × cardiac output (L/min) × 10**, yielding mL O₂/min.
- Hill curve uses n=2.7 and normal P₅₀≈3.5 kPa. Do not interpret model saturation as a measured blood-gas value. Fetal Hb and dyshemoglobins need additional models.
- Arterial PO₂ is estimated using an explicitly assumed A–a gradient. Tissue points are illustrative; venous blood is shown separately. No patient-specific gas exchange or tissue perfusion is predicted.

Sources for further reading:

1. [Collins et al. — Relating oxygen partial pressure, saturation and content (Breathe, 2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4666443/).
2. [Popel — Theory of oxygen transport to tissue](https://pmc.ncbi.nlm.nih.gov/articles/PMC5445261/).
3. [Oxygen transport: basic principles (BMJ)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1114207/).
4. [Neethling — The Oxygen Cascade, anaesthesia refresher course](https://anaesthetics.ukzn.ac.za/wp-content/uploads/2024/09/Oxygen-cascade-Ref-2021.pdf).
5. [Cruickshank & Hirschauer — The alveolar gas equation (CEACCP, 2004)](https://e-safe-anaesthesia.org/e_library/02/The_alveolar_gas_equation_CEACCP_2004.pdf).
6. [Oxygen transport in normal and pathological situations — Fick principle and extraction](https://www.ncbi.nlm.nih.gov/books/NBK54113/).
7. [Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

Built for Prof Madapura Shashidhara's MMed/DM and Primary FRCA teaching. AI-assisted educational resource: tutor review before use. Formative feedback, not a certified examination or clinical decision tool.
""")



def viva_case(code):
    i = learner_form_index(code)
    pb, fio, pco, aa = 101.3-i*2.5, .21+.01*(i%4), 4.2+.2*(i%5), .6+.2*(i%4)
    pio = fio*(pb-6.3)
    pao = pio-pco/.8
    pa = pao-aa
    hb, co = 8.+.5*i, 3.5+.3*i
    ca = 1.34*hb*.97+.0225*12
    shifts = [
        ("fever with acidosis", "Right shift: increased P₅₀, lower affinity and more unloading at fixed tissue PO₂."),
        ("hypothermia with alkalosis", "Left shift: lower P₅₀, higher affinity and greater retention at fixed tissue PO₂."),
        ("increased red-cell 2,3-DPG", "Right shift: 2,3-DPG stabilises lower-affinity deoxyHb, increasing P₅₀."),
        ("fetal haemoglobin compared with adult Hb", "Left shift: reduced 2,3-DPG interaction supports higher affinity and lower P₅₀."),
        ("carbon monoxide exposure", "Available oxygen-carrying capacity falls and remaining-site affinity increases; normal PaO₂ does not exclude impaired transport."),
        ("acute hyperventilation with respiratory alkalosis", "Lower CO₂/alkalosis favour a left shift; lower PaCO₂ also raises PAO₂. Consider loading and unloading separately."),
        ("stored red cells depleted of 2,3-DPG", "A left shift is expected initially; low 2,3-DPG increases affinity. Do not assume the same effect persists after in-vivo recovery."),
        ("a rise in exercising-muscle temperature and local CO₂", "Right shift from heat and CO₂/H⁺ facilitates unloading at fixed tissue PO₂; lung loading remains relevant."),
        ("lower pH at fixed temperature", "A rightward Bohr effect lowers oxygen affinity and raises P₅₀; it need not change the number of Hb molecules."),
        ("a fall in total Hb with unchanged affinity", "A saturation-versus-PO₂ curve can remain unchanged while the content curve falls. Anaemia alone does not require a P₅₀ change."),
    ]
    scenario, shift_answer = shifts[i]
    return [
        ("0–4 min · Oxygen cascade",
         f"Draw the cascade at PB {pb:.1f} kPa, FiO₂ {fio:.2f}, PH₂O 6.3 kPa, rest and 37°C. Explain each fall. Where should mixed venous blood appear? Calculate PIO₂.",
         f"PIO₂ = {fio:.2f}×({pb:.1f}−6.3) = {pio:.3f} kPa. Show dry air → humidified gas → alveoli → arterial blood → tissue → mitochondria. Explain humidification, gas exchange, venous admixture/VQ inequality and tissue consumption. Mixed venous blood belongs on a separate return pathway; tissue and mitochondrial values are variable."),
        ("4–7 min · Gas equation and interpretation",
         f"For that independent gas-exchange example, PaCO₂ is {pco:.1f} kPa, R 0.8 and PaO₂ {pa:.3f} kPa. Calculate PAO₂ and A–a. What would happen if PaCO₂ increased by 1.6 kPa with inspired conditions fixed?",
         f"PAO₂ = {pio:.3f}−{pco:.1f}/0.8 = {pao:.3f} kPa. A–a = {pao:.3f}−{pa:.3f} = {aa:.3f} kPa, allowing rounding of the stated PaO₂. Increasing PaCO₂ by 1.6 kPa lowers simplified PAO₂ by 2.0 kPa. Pure hypoventilation need not widen A–a. State steady state, negligible inspired CO₂ and assumed R."),
        ("7–11 min · Dissociation curve",
         f"Sketch the adult saturation curve, label axes and normal P₅₀, then discuss {scenario}. Distinguish affinity from capacity and explain why a shift does not alone predict whole-body delivery.",
         "PO₂ axis in kPa, saturation axis in %. Normal P₅₀ approximately 3.5–3.6 kPa at standard conditions. Explain cooperativity, lung loading plateau and tissue unloading region. "+shift_answer+" DO₂ depends on arterial content and flow; at low lung PO₂ affinity changes may affect loading as well as unloading."),
        ("11–15 min · Content, delivery and response",
         f"In a SEPARATE transport example, Hb is {hb*10:g} g/L, SaO₂ 97%, PaO₂ 12 kPa and CO {co:.1f} L/min. Using 1.34 and dissolved coefficient 0.0225, calculate CaO₂ and DO₂. What happens to DO₂ if CO falls 20% with content unchanged?",
         f"Hb = {hb:.1f} g/dL and saturation = 0.97. CaO₂ = 1.34×{hb:.1f}×0.97+0.0225×12 = {ca:.3f} mL/dL. DO₂ = {ca:.3f}×{co:.1f}×10 = {ca*co*10:.1f} mL/min. A 20% CO fall causes a 20% delivery fall to {ca*co*10*.8:.1f} mL/min if content is fixed. Explain ×10, normal saturation in anaemia, and the need for adequate flow."),
    ]


def viva_station():
    st.header("Individual viva · 15 minutes")
    st.write("While you examine one candidate, the others complete both written topics independently. Ask each to download and send their JSON results before joining their private viva. Call candidates one at a time; use a private Teams call or room so waiting candidates do not hear the viva.")
    codes = [f"S{i:02}" for i in range(1,11)]
    selected = st.selectbox("Candidate for viva", codes, key="viva_candidate")
    sessions = st.session_state.setdefault("viva_records", {})
    prior = sessions.get(selected, {})
    st.caption("Four candidates take 60 minutes of viva time; ten take 150 minutes. Add any handover or break time separately. This page does not monitor remote screens or automatically receive written results.")
    for heading, prompt, answer in viva_case(selected):
        st.subheader(heading)
        st.write(prompt)
        with st.expander("Examiner model answer and follow-up points"):
            st.write(answer)
    st.markdown("**Record performance:** 0 = not demonstrated; 1 = substantial prompting; 2 = mostly independent with minor gaps; 3 = accurate explanation and reasoning. This is a local formative rubric, with no validated pass mark.")
    domains = ("Cascade", "Gas equation", "Dissociation curve", "Content and delivery")
    with st.form("viva_marks_"+selected):
        scores = {domain: st.selectbox(domain, [0,1,2,3], index=prior.get(domain,0), key="viva_score_"+selected+domain) for domain in domains}
        strength = st.text_area("Strengths", value=prior.get("Strengths",""), key="viva_strength_"+selected)
        improve = st.text_area("Learning points and agreed next step", value=prior.get("Next step",""), key="viva_next_"+selected)
        if st.form_submit_button("Save this candidate's viva"):
            sessions[selected] = {"Candidate":selected, **scores, "Total /12":sum(scores.values()), "Strengths":strength, "Next step":improve, "Saved UTC":datetime.now(timezone.utc).isoformat()}
            st.success("Viva recorded for "+selected+". Download the records before closing or refreshing.")
    st.subheader("Viva completion")
    st.table([{"Candidate":code,"Paper":FORMS[n],"Viva": "Recorded" if code in sessions else "Awaiting viva", "Score /12":sessions[code]["Total /12"] if code in sessions else ""} for n,code in enumerate(codes)])
    if sessions:
        st.download_button("Download viva marks and feedback (CSV)", csv_text(list(sessions.values())), file_name="oxygen_viva_marks.csv", mime="text/csv")
        st.download_button("Download viva record backup (JSON)", json.dumps(list(sessions.values()),ensure_ascii=False,indent=2), file_name="oxygen_viva_records.json", mime="application/json")

def tutor_access():
    """Unlock this browser session using a code kept in Streamlit Secrets."""
    try:
        configured = st.secrets.get("tutor_access_code", "")
    except (FileNotFoundError, KeyError):
        configured = ""
    configured = configured if isinstance(configured, str) else ""
    fingerprint = hashlib.sha256(configured.encode()).hexdigest() if configured else ""
    unlocked = bool(fingerprint) and hmac.compare_digest(
        st.session_state.get("tutor_access_fingerprint", ""), fingerprint
    )
    with st.expander("Tutor / teaching access"):
        if unlocked:
            st.success("Teaching access unlocked in this browser session.")
            if st.button("Lock teaching access", key="lock_teaching"):
                st.session_state.pop("tutor_access_fingerprint", None)
                st.rerun()
        elif not configured:
            st.caption("Assessments are ready. To enable teaching access, the app owner must set tutor_access_code in Streamlit App settings → Secrets.")
        else:
            with st.form("tutor_access_form", clear_on_submit=True):
                entered = st.text_input("Tutor access code", type="password", key="tutor_code_entry", max_chars=256)
                submitted = st.form_submit_button("Unlock teaching access")
            if submitted:
                if hmac.compare_digest(entered.encode(), configured.encode()):
                    st.session_state["tutor_access_fingerprint"] = fingerprint
                    st.rerun()
                else:
                    st.error("Access code not recognised.")
    return unlocked


def main():
    st.set_page_config(page_title="Oxygen Lab · MMed / DM / FRCA",page_icon="🫁",layout="wide")
    st.markdown("<style>.block-container{max-width:1180px;padding-top:2rem}h1,h2,h3{color:#003B71}div[data-testid='stMetric']{background:#edf5fb;padding:16px;border-radius:9px}section[data-testid='stSidebar']{background:#f1f6fa}</style>",unsafe_allow_html=True)
    st.title("Oxygen Lab")
    st.caption("Advanced applied physiology · kPa only · MMed / DM / Primary FRCA")
    with st.sidebar:
        st.header("Assessment workspace")
        learner = st.text_input("Learner code",max_chars=80,key="learner_code",placeholder="S01–S10",help="Use the code allocated by your tutor. Each of the ten codes has its own question paper; it is captured when the assessment starts.")
        unit = "kPa"
        st.caption("All oxygen and carbon dioxide pressures use kPa.")
        unlocked = tutor_access()
        pages = ["Oxygen cascade", "Oxygen transport", "My results"]
        if unlocked:
            pages += ["Viva station", "Teaching room", "Tutor review", "Sources & assumptions"]
        if st.session_state.get("page") not in pages:
            st.session_state["page"] = pages[0]
        page = st.radio("Open", pages, key="page")
        st.caption("Each browser connection has its own session. Download results before refreshing. No login or central database is included.")
        if st.button("Clear session for next learner",key="clear_session"):
            for key in list(st.session_state):
                del st.session_state[key]
            st.rerun()
    if page in ("Oxygen cascade","Oxygen transport"):
        topic = "cascade" if page=="Oxygen cascade" else "transport"
        st.header(TOPICS[topic])
        assessment(topic,unit,learner)
    elif page == "Teaching room" and unlocked:
        selected = st.selectbox("Teaching topic", list(TOPICS.values()), key="teaching_topic")
        topic = "cascade" if selected == "Oxygen cascade" else "transport"
        lesson(topic)
        mode = st.radio("Activity",["Explore & calculate","Sketch & viva"],horizontal=True,key=f"mode_{topic}")
        if mode=="Explore & calculate":
            cascade_lab(unit) if topic=="cascade" else transport_lab(unit)
        elif mode=="Sketch & viva":
            sketch_viva(topic)
    elif page=="My results":
        reports()
    elif page=="Viva station" and unlocked:
        viva_station()
    elif page=="Tutor review" and unlocked:
        tutor_dashboard()
    elif page=="Sources & assumptions" and unlocked:
        references()


if __name__ == "__main__":
    main()
