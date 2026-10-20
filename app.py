"""Oxygen Lab: run with `streamlit run app.py`. Python 3.11+.

Formative teaching and assessment; values are models, not patient advice.
All learner state is session-local. No persistent learner database is used.
"""
from __future__ import annotations

import csv
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


def make_questions(topic, seed=0, unit="kPa"):
    """One shared, ordered learning set. Seed is retained for report compatibility."""
    if topic not in TOPICS:
        raise ValueError("Unknown topic")
    if topic == "cascade":
        q = [
            mcq("c_order", 1, "Which route does oxygen follow from the air to where cells use it?",
                ["Air → airways → alveoli → arterial blood → tissues → mitochondria", "Air → arterial blood → alveoli → tissues → mitochondria", "Air → alveoli → tissues → arterial blood → mitochondria", "Air → airways → mitochondria → arterial blood → tissues"],
                "Air → airways → alveoli → arterial blood → tissues → mitochondria",
                "Oxygen enters the airways, reaches the alveoli and diffuses into pulmonary blood. Arterial blood carries it to tissues, where it diffuses to mitochondria for use in aerobic metabolism. Think: lungs load, blood carries, cells use."),
            numeric("c_air", 1, "Room air contains 21% oxygen. At sea level PB = 101.3 kPa. Calculate PO₂ in dry air: PO₂ = 0.21 × 101.3.", 21.273, .2, "kPa",
                "Dry-air PO₂ = oxygen fraction × total pressure = 0.21 × 101.3 = 21.273, approximately 21.3 kPa. A fraction of 0.21 means 21%, not 21 kPa at every altitude."),
            mcq("c_water", 2, "Why is PO₂ slightly lower after inspired air is humidified?",
                ["Water vapour takes up part of the total gas pressure", "Water reacts with and destroys oxygen", "Haemoglobin removes oxygen inside the airways", "The airways produce nitrogen"],
                "Water vapour takes up part of the total gas pressure",
                "At 37°C, water vapour contributes about 6.3 kPa. The remaining pressure is shared by the dry gases. PIO₂ = 0.21 × (101.3 − 6.3) = 19.95 kPa. Humidification reduces oxygen partial pressure without consuming oxygen."),
            numeric("c_inspired", 1, "Calculate humidified inspired PO₂ using PIO₂ = FiO₂ × (PB − PH₂O). Use FiO₂ = 0.21, PB = 101.3 kPa and PH₂O = 6.3 kPa.", 19.95, .2, "kPa",
                "First subtract water vapour pressure: 101.3 − 6.3 = 95.0 kPa. Then take the oxygen fraction: 0.21 × 95.0 = 19.95, approximately 20.0 kPa. Subtract water vapour before multiplying by FiO₂."),
            mcq("c_alveoli", 2, "Why is alveolar PO₂ lower than humidified inspired PO₂?",
                ["Blood takes up oxygen and adds carbon dioxide to alveolar gas", "Alveoli contain no water vapour", "Haemoglobin produces oxygen in the alveoli", "All inspired oxygen is used by the airways"],
                "Blood takes up oxygen and adds carbon dioxide to alveolar gas",
                "Fresh gas replenishes alveolar oxygen, while pulmonary blood continuously removes oxygen and delivers CO₂. This balance explains why alveolar PO₂ is about 13 kPa rather than the approximately 20 kPa in humidified inspired gas under these reference conditions."),
            mcq("c_tissue", 2, "What helps oxygen diffuse from tissue capillaries towards mitochondria?",
                ["Mitochondria use oxygen, maintaining a lower local PO₂", "Mitochondria produce oxygen, increasing local PO₂", "Oxygen moves only towards a higher PO₂", "All tissue compartments have identical PO₂"],
                "Mitochondria use oxygen, maintaining a lower local PO₂",
                "Oxygen diffuses down its partial-pressure gradient. Mitochondrial consumption helps maintain that gradient. Tissue and mitochondrial PO₂ vary with organ, blood flow and metabolic activity; they are not one fixed value in every patient."),
            mcq("c_gradient", 2, "Why is arterial PO₂ normally slightly lower than alveolar PO₂?",
                ["Small V/Q differences and physiological venous admixture", "Systemic tissues extract oxygen before it reaches an artery", "Water vapour is added directly to arterial blood", "Normal lungs cannot transfer oxygen into blood"],
                "Small V/Q differences and physiological venous admixture",
                "Ventilation and perfusion are not perfectly matched throughout healthy lungs, and a small amount of venous blood mixes with oxygenated blood. These create the normal alveolar–arterial difference. A small normal difference does not mean diffusion has failed."),
            mcq("c_baseline", 3, "Which simplified equation estimates alveolar PO₂? All pressures use kPa.",
                ["PAO₂ = FiO₂ × (PB − PH₂O) − PaCO₂/R", "PAO₂ = FiO₂ × (PB + PH₂O) + PaCO₂/R", "PAO₂ = PaCO₂ × R", "PAO₂ = FiO₂ × PB × PaCO₂"],
                "PAO₂ = FiO₂ × (PB − PH₂O) − PaCO₂/R",
                "Start with humidified inspired oxygen: FiO₂ × (PB − PH₂O). Then subtract PaCO₂/R. This simplified steady-state equation assumes alveolar PCO₂ is approximately arterial PCO₂. R is usually taken as 0.8 for a mixed diet."),
            numeric("c_vent", 3, "Use PAO₂ = PIO₂ − PaCO₂/R. PIO₂ = 20.0 kPa, PaCO₂ = 5.3 kPa and R = 0.8. Calculate PAO₂.", 13.375, .2, "kPa",
                "First calculate 5.3 ÷ 0.8 = 6.625 kPa. Then PAO₂ = 20.0 − 6.625 = 13.375, approximately 13.4 kPa. This is alveolar PO₂, not arterial PO₂; the A–a difference must be considered separately."),
            mcq("c_high", 3, "A person hypoventilates: PaCO₂ rises while FiO₂ and atmospheric pressure stay unchanged. What happens to alveolar PO₂?",
                ["It falls", "It rises", "It must remain unchanged", "It becomes equal to atmospheric pressure"],
                "It falls",
                "In PAO₂ = PIO₂ − PaCO₂/R, a larger PaCO₂ means a larger subtraction. Alveolar PO₂ falls. With R = 0.8, each 1 kPa rise in PaCO₂ lowers PAO₂ by about 1.25 kPa if other inputs stay fixed."),
            mcq("c_altitude", 3, "At altitude, room air still contains about 21% oxygen. Why does inspired PO₂ fall?",
                ["Atmospheric pressure is lower", "The oxygen fraction always falls below 21%", "Water vapour pressure becomes equal to atmospheric pressure", "Haemoglobin stops binding oxygen"],
                "Atmospheric pressure is lower",
                "PIO₂ depends on both oxygen fraction and atmospheric pressure. At altitude, the same fraction is multiplied by a smaller pressure. Hyperventilation can partly compensate by lowering PaCO₂, but it does not restore atmospheric pressure."),
            mcq("c_oxygen", 3, "FiO₂ is increased while PB, PH₂O, PaCO₂ and R stay unchanged. What happens to calculated PAO₂?",
                ["It rises", "It falls", "It cannot change", "It becomes the same as PaCO₂"],
                "It rises",
                "Increasing FiO₂ increases the inspired oxygen term, so calculated alveolar PO₂ rises. Arterial improvement also depends on gas exchange; alveolar and arterial PO₂ are not interchangeable. The simplified equation is an approximation, especially at high FiO₂."),
        ]
    else:
        q = [
            mcq("t_pressure", 1, "What does PaO₂ tell you?",
                ["The partial pressure of dissolved oxygen in arterial blood", "The total oxygen carried in a volume of blood", "The percentage of haemoglobin occupied by oxygen", "The volume of oxygen delivered each minute"],
                "The partial pressure of dissolved oxygen in arterial blood",
                "PaO₂ describes oxygen tension, measured here in kPa. It does not measure how much haemoglobin is present. Therefore normal PaO₂ can coexist with low oxygen content in anaemia."),
            mcq("t_dissolved", 1, "What does an SaO₂ of 98% mean?",
                ["About 98% of haemoglobin oxygen-binding sites are occupied", "98% of blood volume consists of oxygen", "PaO₂ is 98 kPa", "98% of oxygen is dissolved in plasma"],
                "About 98% of haemoglobin oxygen-binding sites are occupied",
                "Saturation describes the proportion of haemoglobin binding sites carrying oxygen. It does not tell you the amount of haemoglobin. A nearly full small carrier can still carry less than a nearly full large carrier."),
            mcq("t_anaemia", 1, "Two patients have the same PaO₂ and SaO₂. One has Hb 15 g/dL and the other Hb 7.5 g/dL. Who has lower arterial oxygen content?",
                ["The patient with Hb 7.5 g/dL", "The patient with Hb 15 g/dL", "Both must have identical content", "Haemoglobin has no effect on content"],
                "The patient with Hb 7.5 g/dL",
                "Most oxygen is carried on haemoglobin. Halving Hb approximately halves total content when PaO₂ and saturation are unchanged. The dissolved component remains the same, so the reduction is approximately, rather than exactly, one half."),
            mcq("t_cooperative", 2, "Why is the oxyhaemoglobin dissociation curve S-shaped?",
                ["Binding one oxygen molecule makes further binding easier", "Haemoglobin has only one oxygen-binding site", "Dissolved oxygen increases in an S-shaped pattern", "Haemoglobin concentration rises as PO₂ rises"],
                "Binding one oxygen molecule makes further binding easier",
                "This is positive cooperative binding: oxygen binding changes haemoglobin towards a higher-affinity state. On a sketch, place PO₂ in kPa on the x-axis and saturation (%) on the y-axis."),
            mcq("t_plateau", 2, "What is useful about the plateau of the dissociation curve?",
                ["Saturation stays relatively high despite a moderate fall in PO₂", "It guarantees normal oxygen content in severe anaemia", "It means dissolved oxygen cannot increase", "It prevents any oxygen unloading in tissues"],
                "Saturation stays relatively high despite a moderate fall in PO₂",
                "The upper flat part helps preserve loading in the lungs. The steeper portion allows more oxygen to be unloaded as tissue PO₂ falls. The plateau does not protect against low haemoglobin or low cardiac output."),
            mcq("t_p50", 2, "P₅₀ is approximately 3.5 kPa in a normal adult. What does P₅₀ mean?",
                ["The PO₂ at which haemoglobin is 50% saturated", "The PO₂ at which all haemoglobin is saturated", "The amount of oxygen carried by 50 g of haemoglobin", "The saturation at a PO₂ of 50 kPa"],
                "The PO₂ at which haemoglobin is 50% saturated",
                "P₅₀ is a measure of oxygen affinity under stated conditions. Higher P₅₀ means more PO₂ is needed for 50% saturation, so affinity is lower. The normal adult value is approximately 3.5–3.6 kPa at 37°C and pH 7.4."),
            mcq("t_shift", 2, "At the same tissue PO₂, what does a right shift favour?",
                ["Oxygen unloading because haemoglobin affinity is lower", "Oxygen retention because haemoglobin affinity is higher", "An increase in the number of haemoglobin molecules", "A change in oxygen into carbon dioxide inside haemoglobin"],
                "Oxygen unloading because haemoglobin affinity is lower",
                "A right shift increases P₅₀ and lowers affinity. At a given PO₂, haemoglobin holds less oxygen, favouring unloading. This describes affinity, not an automatic increase in total oxygen delivery."),
            mcq("t_factors", 2, "Which change shifts the adult dissociation curve to the right?",
                ["A rise in temperature", "A fall in temperature", "A rise in pH", "A fall in PaCO₂"],
                "A rise in temperature",
                "Higher temperature, higher CO₂, lower pH and higher 2,3-DPG favour a right shift. Warm, metabolically active tissues can therefore receive oxygen more readily. The opposite changes favour a left shift."),
            mcq("t_anchor", 3, "Which set of factors directly determines oxygen delivery to the body?",
                ["Haemoglobin concentration, saturation, dissolved oxygen and cardiac output", "PaO₂ alone", "Saturation alone", "Respiratory rate alone"],
                "Haemoglobin concentration, saturation, dissolved oxygen and cardiac output",
                "Delivery = arterial oxygen content × cardiac output. Content is mainly haemoglobin-bound oxygen plus a small dissolved component. Think: amount carried in each volume of blood × volume pumped each minute."),
            numeric("t_content", 3, "Calculate CaO₂ in mL O₂/dL. Use CaO₂ = 1.34 × Hb × SaO₂ + 0.0225 × PaO₂, with Hb 15 g/dL, SaO₂ 0.98 and PaO₂ 13.3 kPa. Enter 0.98, not 98, in the formula.", 19.99725, .2, "mL O₂/dL",
                "Bound oxygen = 1.34 × 15 × 0.98 = 19.698 mL/dL. Dissolved oxygen = 0.0225 × 13.3 = 0.299 mL/dL. Total = 19.997, approximately 20.0 mL O₂/dL. The coefficient 0.0225 is for PaO₂ in kPa."),
            numeric("t_delivery", 3, "CaO₂ = 20 mL O₂/dL and cardiac output = 5 L/min. Calculate DO₂ using DO₂ = CaO₂ × cardiac output × 10. The factor 10 converts litres to decilitres.", 1000., 10., "mL O₂/min",
                "DO₂ = 20 × 5 × 10 = 1000 mL O₂/min. Each litre contains ten decilitres, so 5 L/min is 50 dL/min. Multiply 20 mL O₂/dL by 50 dL/min. Delivery is not the same as oxygen consumption."),
            mcq("t_units", 3, "Cardiac output falls from 5 to 2.5 L/min, but oxygen content is unchanged. What happens to oxygen delivery?",
                ["It halves", "It doubles", "It stays unchanged because saturation is normal", "It becomes zero"],
                "It halves",
                "DO₂ is directly proportional to cardiac output when CaO₂ is fixed. Halving blood flow halves delivery. This is why a normal saturation alone cannot demonstrate adequate tissue oxygen delivery."),
        ]
    # Distribute answer positions once, identically for every candidate and retry.
    rng = random.Random(20261020)
    for item in q:
        if item["type"] == "mcq":
            rng.shuffle(item["options"])
    return q


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
    return {k:record[k] for k in ("learner_code","topic","attempt_id","submitted_utc","score","total","percentage","duration_seconds","pressure_unit")}


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
        expected = {q["id"]:q["objective"] for q in make_questions(r["topic"],0)}
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
    st.caption("*Tissue 20 mmHg and mitochondrial 2 mmHg are illustrative resting teaching points. At low arterial estimates these nodes are reduced only to preserve the schematic gradient; they are NOT a calculated tissue oxygenation model. Tissue/mitochondrial PO₂ varies widely.")
    st.info("Mixed venous PO₂ ≈ 40 mmHg (5.3 kPa) is a resting reference after extraction. Venous blood returns from tissues; it is not a serial diffusion step between arterial blood and mitochondria.")
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
    for label,p,c in [("Left: P₅₀ 20 mmHg",20.,TEAL),("Normal: P₅₀ 26.6 mmHg",26.6,BLUE),("Right: P₅₀ 33 mmHg",33.,ORANGE)]:
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
    st.latex(r"C_{a O_2}=1.34\times Hb\times\frac{S_{a O_2}(\%)}{100}+0.003\times P_{a O_2}\;(mmHg)")
    st.caption("CaO₂ in mL O₂/dL, Hb in g/dL. For PaO₂ in kPa use dissolved coefficient 0.022502. Hb in g/L must be divided by 10 first. Hb capacity 1.34 is the chosen clinical teaching constant.")
    st.write(f"CaO₂ = 1.34 × {hb:g} × {sa/100:.3f} + 0.003 × {pa:g} = **{content:.2f} mL O₂/dL**")
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
        rubric = ["State room air, sea level, rest and 37°C","Label axes and units; order air → humidified → alveoli → arterial → tissue → mitochondria","Dry air ≈21.3, humidified inspired ≈20.0, alveolar ≈13.3, arterial ≈12–13 kPa under stated conditions","Mixed venous ≈5.3 kPa is a returning-blood reference; tissue/mitochondrial values vary","Explain water vapour, alveolar exchange, A–a gradient and tissue extraction","Write the gas equation and predict hypoventilation, altitude and FiO₂ effects"]
        viva = ["What are the main stages of the oxygen cascade?", "Why does humidification lower inspired PO₂?", "Why is alveolar PO₂ lower than humidified inspired PO₂?", "Use PIO₂ 20.0 kPa, PaCO₂ 5.3 kPa and R 0.8 to calculate PAO₂.", "If PaCO₂ rises, what happens to PAO₂?", "Why does inspired PO₂ fall at altitude?"]
    else:
        st.write("Sketch the normal adult dissociation curve, mark P₅₀ and approximate 5.3/8.0/13.3 kPa anchor points, then overlay a left and a right shift. Explain loading and unloading. Use the second calculation to explain why normal saturation can be misleading in anaemia.")
        rubric = ["PO₂ on x-axis, saturation (%) on y-axis; sigmoid curve","P₅₀ ≈3.5–3.6 kPa at 50% saturation","Approximate anchors: 5.3 kPa≈75%, 8.0 kPa≈90%, 13.3 kPa≈97–98%; explain steep and plateau regions","Explain cooperative binding","Right shift: increased P₅₀, lower affinity; list heat/CO₂/H⁺/2,3-DPG","Distinguish pressure/saturation/content and calculate CaO₂ and DO₂ with correct units"]
        viva = ["What is the difference between PaO₂, SaO₂ and CaO₂?", "Why is the dissociation curve S-shaped?", "What does P₅₀ mean?", "Name two factors causing a right shift and explain unloading.", "Why can oxygen content be low despite a normal saturation in anaemia?", "Calculate DO₂ for CaO₂ 20 mL/dL and cardiac output 5 L/min."]
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
    st.caption("12 items · 1 mark each · no negative marking. Numerical tolerances are shown. Everyone receives the same questions, numbers and answer choices in the same order. Questions build from understanding to straightforward application. All pressures are in kPa. This is an open-resource teaching test, not a secure examination.")
    key = f"attempt_{topic}"
    attempt = st.session_state.get(key)
    if not attempt:
        if st.button("Start assessment",type="primary",key=f"start_{topic}"):
            if not learner.strip():
                st.error("Enter a learner code in the sidebar first.")
                return
            attempt = {"id":str(uuid.uuid4()),"learner":learner.strip(),"unit":unit,"started":time.time(),"questions":make_questions(topic,0,"kPa"),"result":None}
            st.session_state[key] = attempt
            st.rerun()
        return
    st.write(f"Learner: **{attempt['learner']}** · pressure unit: **{attempt['unit']}**")
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
                items.append(dict(id=q["id"],objective=q["objective"],prompt=q["prompt"],response=response,expected=q["answer"],earned=grade(q,response),explanation=q["explanation"]))
            score = sum(x["earned"] for x in items)
            result = dict(learner_code=attempt["learner"],topic=topic,attempt_id=attempt["id"],submitted_utc=datetime.now(timezone.utc).isoformat(),score=score,total=len(items),percentage=round(100*score/len(items),1),duration_seconds=round(time.time()-attempt["started"],1),pressure_unit=attempt["unit"],items=items)
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
            st.write(f"Your answer: {item['response']}")
            st.write(f"Correct answer: {item['expected']}")
            st.write(item["explanation"])
    st.download_button("Download this attempt (JSON)",json.dumps({"schema_version":1,"attempts":[result]},ensure_ascii=False,indent=2),file_name=f"oxygen_{topic}_{attempt['id'][:8]}.json",mime="application/json")
    if st.button("Start a new attempt",key=f"again_{topic}"):
        del st.session_state[key]
        st.rerun()


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
    st.caption("Download before refreshing or closing the browser. Results remain only in this active session; there is no automatic central collection.")


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
    st.markdown("""- All displayed pressure values use **kPa**.
- Simplified alveolar gas equation: **PAO₂ ≈ FiO₂ × (PB − PH₂O) − PaCO₂/R**. At 37°C use PH₂O ≈6.3 kPa. Assume steady state and PACO₂ ≈ PaCO₂; R=0.8 for these exercises.
- Reference exercise: PB 101.3 kPa, FiO₂ 0.21, PH₂O 6.3 kPa gives PIO₂ ≈20.0 kPa. With PaCO₂ 5.3 kPa and R 0.8, PAO₂ ≈13.3–13.4 kPa depending on rounding.
- CaO₂ = **1.34 × Hb (g/dL) × SaO₂ (fraction) + 0.0225 × PaO₂ (kPa)**, yielding mL O₂/dL.
- DO₂ = **CaO₂ (mL/dL) × cardiac output (L/min) × 10**, yielding mL O₂/min.
- Normal adult P₅₀ is approximately **3.5–3.6 kPa**. The Hill curve is a teaching approximation, not a measured saturation.
- Internal calculation helpers retain their original pressure representation for compatibility. Learner questions, calculations and feedback use kPa.
- Arterial PO₂ and tissue points in the simulator are illustrative. Alveolar and arterial PO₂ are not interchangeable. The simplified gas equation is an approximation, particularly at high FiO₂.

Sources for further reading:

1. [Collins et al. — Relating oxygen partial pressure, saturation and content (Breathe, 2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4666443/).
2. [Popel — Theory of oxygen transport to tissue](https://pmc.ncbi.nlm.nih.gov/articles/PMC5445261/).
3. [Oxygen transport: basic principles (BMJ)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1114207/).
4. [Neethling — The Oxygen Cascade, anaesthesia refresher course](https://anaesthetics.ukzn.ac.za/wp-content/uploads/2024/09/Oxygen-cascade-Ref-2021.pdf).
5. [Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

Built for Prof Madapura Shashidhara's MMed/DM and Primary FRCA teaching. AI-assisted educational resource: tutor review before use. Formative feedback, not a certified examination or clinical decision tool.
""")


def main():
    st.set_page_config(page_title="Oxygen Lab · MMed / DM / FRCA",page_icon="🫁",layout="wide")
    st.markdown("<style>.block-container{max-width:1180px;padding-top:2rem}h1,h2,h3{color:#003B71}div[data-testid='stMetric']{background:#edf5fb;padding:16px;border-radius:9px}section[data-testid='stSidebar']{background:#f1f6fa}</style>",unsafe_allow_html=True)
    st.title("Oxygen Lab")
    st.caption("Explore → calculate → sketch → test · MMed / DM / Primary FRCA")
    with st.sidebar:
        st.header("Learning workspace")
        learner = st.text_input("Learner code",max_chars=80,key="learner_code",placeholder="e.g. DM-01",help="Use a code rather than a full name. It is captured when an assessment starts.")
        unit = "kPa"
        st.caption("Pressure units: kPa throughout.")
        page = st.radio("Open",["Oxygen cascade","Oxygen transport","My results","Tutor review","Sources & assumptions"],key="page")
        st.caption("Each browser connection has its own session. Download results before refreshing. No login or central database is included.")
        if st.button("Clear session for next learner",key="clear_session"):
            for key in list(st.session_state):
                del st.session_state[key]
            st.rerun()
    if page in ("Oxygen cascade","Oxygen transport"):
        topic = "cascade" if page=="Oxygen cascade" else "transport"
        lesson(topic)
        mode = st.radio("Activity",["Explore & calculate","Sketch & viva","Assessment"],horizontal=True,key=f"mode_{topic}")
        if mode=="Explore & calculate":
            cascade_lab(unit) if topic=="cascade" else transport_lab(unit)
        elif mode=="Sketch & viva":
            sketch_viva(topic)
        else:
            assessment(topic,unit,learner)
    elif page=="My results":
        reports()
    elif page=="Tutor review":
        tutor_dashboard()
    else:
        references()


if __name__ == "__main__":
    main()
