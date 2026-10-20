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


def make_questions(topic, seed, unit="kPa"):
    """12 equally weighted items/topic; unique parameters and shuffled choices."""
    rng = random.Random(seed)
    u = lambda x: from_mmhg(x, unit)
    tol = 0.20 if unit == "kPa" else 1.5
    if topic == "cascade":
        pco2 = rng.choice([32, 40, 48])
        pb = rng.choice([600, 650, 700])
        q = [
            mcq("c_order", 1, "Which sequence follows oxygen from inspired air to its site of use?",
                ["Dry air → humidified air → alveoli → arterial blood → tissues → mitochondria",
                 "Dry air → arterial blood → alveoli → tissues → mitochondria",
                 "Dry air → humidified air → tissues → alveoli → mitochondria",
                 "Dry air → alveoli → mitochondria → arterial blood → tissues"],
                "Dry air → humidified air → alveoli → arterial blood → tissues → mitochondria",
                "Oxygen passes through the airways and alveoli, enters arterial blood, then diffuses into tissues and mitochondria. Mixed venous blood returns after extraction; it is not an intermediate on the tissue diffusion path."),
            mcq("c_water", 2, "Why does inspired PO₂ fall during humidification at 37°C?",
                ["Water vapour contributes to total pressure, reducing the pressure available to dry gases",
                 "Water vapour consumes oxygen in a chemical reaction", "Humidification adds carbon dioxide from blood",
                 "Water vapour increases the oxygen fraction"],
                "Water vapour contributes to total pressure, reducing the pressure available to dry gases",
                "At 37°C, saturated water vapour pressure is 47 mmHg (6.3 kPa). PIO₂ = FiO₂ × (PB − PH₂O)."),
            mcq("c_alveoli", 2, "Why is alveolar PO₂ below humidified inspired PO₂ at steady state?",
                ["Oxygen uptake into blood and carbon dioxide entry into alveolar gas",
                 "Water vapour pressure becomes zero in the alveoli", "Haemoglobin releases oxygen into alveoli",
                 "Nitrogen is completely absorbed"],
                "Oxygen uptake into blood and carbon dioxide entry into alveolar gas",
                "Alveolar ventilation replenishes oxygen while blood takes up oxygen and delivers CO₂. Their steady-state balance gives the gas equation."),
            mcq("c_gradient", 2, "What chiefly explains the small normal A–a oxygen gradient in a healthy adult at rest?",
                ["V/Q inequality and physiological venous admixture", "Failure of all normal capillaries to equilibrate",
                 "Humidification after blood leaves the lungs", "Systemic tissue extraction before arterial sampling"],
                "V/Q inequality and physiological venous admixture",
                "Normal end-capillary blood usually equilibrates with its alveolus. V/Q inequality and physiological shunt lower the resulting arterial PO₂; diffusion limitation may add to the gradient in disease."),
            mcq("c_tissue", 2, "What maintains the tissue-to-mitochondrial oxygen pressure gradient?",
                ["Oxygen consumption during oxidative phosphorylation", "Oxygen production by mitochondria",
                 "Equal PO₂ in every tissue compartment", "A rise in haemoglobin saturation in tissues"],
                "Oxygen consumption during oxidative phosphorylation",
                "Consumption at mitochondria maintains low local PO₂ and diffusion towards the cells. Tissue values vary by organ and metabolic demand."),
            mcq("c_altitude", 3, "At altitude, with FiO₂ and PaCO₂ held fixed, what happens to PAO₂?",
                ["It falls because barometric pressure falls", "It rises because FiO₂ rises automatically",
                 "It is unchanged because FiO₂ stays 21%", "It falls only if water vapour pressure rises"],
                "It falls because barometric pressure falls",
                "FiO₂ stays approximately 0.21 in ambient air. Lower PB reduces PIO₂. Real altitude adaptation often lowers PaCO₂ through hyperventilation."),
        ]
        for qid, label, mm, explanation in [
            ("c_air", "dry atmospheric air", 0.21*760, "PO₂ = 0.21 × 760 = 159.6 mmHg, approximately 21.3 kPa."),
            ("c_inspired", "fully humidified inspired gas", 0.21*713, "PIO₂ = 0.21 × (760 − 47) = 149.7 mmHg, approximately 20.0 kPa."),
            ("c_baseline", "alveolar gas", 0.21*713-50, "PAO₂ = 0.21 × 713 − 40/0.8 = 99.7 mmHg, approximately 13.3 kPa."),
        ]:
            q.append(numeric(qid, 1, f"Healthy adult at rest, room air, sea level: FiO₂ 0.21, PB 760 mmHg, PH₂O 47 mmHg, PaCO₂ 40 mmHg, R 0.8. Estimate PO₂ in {label} in {unit}.", u(mm), tol, unit, explanation))
        q.append(numeric("c_vent", 3, f"Use the simplified equation. FiO₂ 0.21, PB {u(760):.3f} {unit}, PH₂O {u(47):.3f} {unit}, PaCO₂ {u(pco2):.3f} {unit}, R 0.8. Calculate PAO₂ in {unit}.", u(alveolar_po2(.21, 760, pco2)), tol, unit,
                         f"PAO₂ = 0.21 × (760 − 47) − {pco2}/0.8 = {alveolar_po2(.21,760,pco2):.2f} mmHg. Convert by dividing by 7.50062 for kPa."))
        q.append(numeric("c_high", 3, f"Use the simplified equation. PB {u(pb):.3f} {unit}, FiO₂ 0.21, PH₂O {u(47):.3f} {unit}, PaCO₂ {u(32):.3f} {unit}, R 0.8. Calculate PAO₂ in {unit}.", u(alveolar_po2(.21,pb,32)), tol, unit,
                         f"PAO₂ = 0.21 × ({pb} − 47) − 32/0.8 = {alveolar_po2(.21,pb,32):.2f} mmHg."))
        q.append(numeric("c_oxygen", 3, f"Use the simplified equation. Sea level PB 760 mmHg, PH₂O 47 mmHg, FiO₂ 0.40, PaCO₂ 40 mmHg, R 0.8. Calculate PAO₂ in {unit}.", u(alveolar_po2(.4)), tol, unit,
                         "PAO₂ ≈ 0.40 × 713 − 50 = 235.2 mmHg (31.4 kPa). This is the requested simplified estimate, not the full equation."))
    else:
        hb, sao, co = rng.choice([9.,12.,15.]), rng.choice([.90,.95,.98]), rng.choice([4.,5.,6.])
        pa = rng.choice([75.,90.,100.])
        content = oxygen_content(hb,sao,pa)
        q = [
            mcq("t_pressure", 1, "What does PaO₂ describe?",
                ["Oxygen tension determined by dissolved oxygen", "Total oxygen carried per litre of blood",
                 "Percentage of haemoglobin binding sites occupied", "Oxygen delivery per minute"],
                "Oxygen tension determined by dissolved oxygen",
                "PaO₂ is a partial pressure. SaO₂ is binding-site occupancy; CaO₂ is the amount of oxygen per blood volume."),
            mcq("t_anaemia", 1, "Hb falls from 15 to 7.5 g/dL. PaO₂, SaO₂ and cardiac output are unchanged. What happens?",
                ["CaO₂ and DO₂ fall to approximately half", "PaO₂ must fall to half", "SaO₂ must fall to half", "CaO₂ remains unchanged"],
                "CaO₂ and DO₂ fall to approximately half",
                "The Hb-bound component halves. Dissolved oxygen stays the same, so total content and delivery are approximately, not exactly, halved."),
            mcq("t_cooperative", 2, "Why is the adult oxyhaemoglobin dissociation curve sigmoid?",
                ["Binding one oxygen molecule increases affinity for subsequent molecules", "Haemoglobin has only one binding site",
                 "Dissolved oxygen binds cooperatively to water", "Oxygen content is independent of saturation"],
                "Binding one oxygen molecule increases affinity for subsequent molecules",
                "Cooperative binding produces a steep middle region and a plateau at high PO₂."),
            mcq("t_shift", 2, "At a fixed tissue PO₂, what does a right shift mean?",
                ["Higher P₅₀, lower affinity and greater unloading", "Lower P₅₀, higher affinity and greater unloading",
                 "Higher P₅₀, higher affinity and less unloading", "No change in saturation or unloading"],
                "Higher P₅₀, lower affinity and greater unloading",
                "At the same PO₂, a right-shifted curve has lower saturation. Increased P₅₀ indicates reduced affinity; loading can also be impaired at low alveolar PO₂."),
            mcq("t_factors", 2, "Which group shifts the adult curve to the right?",
                ["Increased temperature, increased CO₂, reduced pH, increased 2,3-DPG",
                 "Reduced temperature, reduced CO₂, increased pH, reduced 2,3-DPG",
                 "Increased temperature, reduced CO₂, increased pH, reduced 2,3-DPG",
                 "Reduced temperature, increased CO₂, increased pH, reduced 2,3-DPG"],
                "Increased temperature, increased CO₂, reduced pH, increased 2,3-DPG",
                "Heat, CO₂, H⁺ and 2,3-DPG favour lower oxygen affinity. The converse factors and fetal haemoglobin favour a left shift."),
            mcq("t_plateau", 2, "What does the plateau at higher PO₂ help preserve?",
                ["High saturation despite moderate falls in alveolar/arterial PO₂", "Constant dissolved oxygen at any PO₂",
                 "Complete tissue unloading at high PO₂", "Normal content regardless of haemoglobin concentration"],
                "High saturation despite moderate falls in alveolar/arterial PO₂",
                "The plateau supports pulmonary loading; the steep middle part facilitates tissue unloading for a small fall in PO₂."),
            mcq("t_units", 3, "CaO₂ is in mL O₂/dL and cardiac output is in L/min. What is DO₂ in mL O₂/min?",
                ["CaO₂ × cardiac output × 10", "CaO₂ × cardiac output", "CaO₂ × cardiac output ÷ 10", "CaO₂ ÷ cardiac output"],
                "CaO₂ × cardiac output × 10",
                "One litre contains ten decilitres. Omitting ×10 gives a tenfold underestimation."),
            numeric("t_p50", 2, f"At 37°C, pH 7.4 and PaCO₂ 40 mmHg, give an approximate normal adult P₅₀ in {unit}.", u(26.6), .25 if unit=="kPa" else 2., unit,
                    "P₅₀ is the PO₂ at 50% saturation, approximately 26–27 mmHg or 3.5–3.6 kPa."),
            numeric("t_anchor", 2, f"On a normal adult curve, PO₂ of approximately {u(60):.1f} {unit} corresponds to what approximate saturation (%)?", 90., 3., "%",
                    "The teaching anchor is 60 mmHg (8 kPa) ≈ 90% saturation; below this point saturation falls more rapidly."),
            numeric("t_content", 3, f"Hb {hb:g} g/dL, SaO₂ {sao*100:g}%, PaO₂ {u(pa):.3f} {unit}. Using 1.34 and dissolved coefficient 0.003 per mmHg (0.022502 per kPa), calculate CaO₂ (mL O₂/dL).", content, .15, "mL O₂/dL",
                    f"CaO₂ = 1.34 × {hb:g} × {sao:.2f} + 0.003 × {pa:g} = {content:.3f} mL O₂/dL."),
            numeric("t_delivery", 3, f"In an independent case, CaO₂ is {content:.3f} mL O₂/dL and cardiac output is {co:g} L/min. Calculate DO₂ (mL O₂/min).", oxygen_delivery(content,co), 10., "mL O₂/min",
                    f"DO₂ = {content:.3f} × {co:g} × 10 ≈ {oxygen_delivery(content,co):.1f} mL O₂/min."),
            numeric("t_dissolved", 1, f"PaO₂ is {u(100):.3f} {unit}. Using 0.003 mL O₂/dL/mmHg (0.022502 per kPa), calculate dissolved oxygen content (mL O₂/dL).", .3, .02, "mL O₂/dL",
                    "Dissolved content = 0.003 × 100 = 0.30 mL O₂/dL, a small fraction of normal total arterial content."),
        ]
    for item in q:
        if item["type"] == "mcq":
            rng.shuffle(item["options"])
    rng.shuffle(q)
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
        rubric = ["State room air, sea level, rest and 37°C","Label axes and units; order air → humidified → alveoli → arterial → tissue → mitochondria","Air ≈159, humidified ≈150, alveolar ≈100, arterial ≈95 mmHg","Recognise mixed venous ≈40 mmHg; tissue/mitochondrial values vary","Explain water vapour, alveolar exchange, A–a gradient and tissue extraction","Write the gas equation and predict hypoventilation, altitude and FiO₂ effects"]
        viva = ["Why does 47 mmHg water vapour reduce inspired PO₂ by only about 10 mmHg?","Why does hypoventilation lower PAO₂ without necessarily widening the A–a gradient?","Which assumptions must be stated before quoting a mitochondrial PO₂?"]
    else:
        st.write("Sketch the normal adult dissociation curve, mark P₅₀ and the 40/60/100 mmHg anchor points, then overlay a left and a right shift. Explain loading and unloading. Use the second calculation to explain why normal saturation can be misleading in anaemia.")
        rubric = ["PO₂ on x-axis, saturation (%) on y-axis; sigmoid curve","P₅₀ ≈26–27 mmHg (3.5–3.6 kPa) at 50%","Anchors: 40≈75%, 60≈90%, 100≈97–98%; steep and plateau regions","Explain cooperative binding","Right shift: increased P₅₀, lower affinity; list heat/CO₂/H⁺/2,3-DPG","Distinguish pressure/saturation/content and calculate CaO₂ and DO₂ with correct units"]
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
    st.caption("12 items · 1 mark each · no negative marking. Numerical tolerances are shown. Each new attempt has shuffled choices and varied calculation parameters. This is an open-resource teaching test, not a secure examination.")
    key = f"attempt_{topic}"
    attempt = st.session_state.get(key)
    if not attempt:
        if st.button("Start assessment",type="primary",key=f"start_{topic}"):
            if not learner.strip():
                st.error("Enter a learner code in the sidebar first.")
                return
            attempt = {"id":str(uuid.uuid4()),"learner":learner.strip(),"unit":unit,"started":time.time(),"questions":make_questions(topic,uuid.uuid4().int,unit),"result":None}
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
    st.markdown("""- Pressure conversion: **1 kPa = 7.50062 mmHg**.
- Simplified alveolar gas equation: **PAO₂ ≈ FiO₂ × (PB − 47) − PaCO₂/R**, pressures in mmHg, 37°C, steady state, PACO₂ ≈ PaCO₂. In kPa, PH₂O ≈6.266. At high FiO₂ the fuller equation differs: **PAO₂ = FiO₂(PB−PH₂O) − PaCO₂[FiO₂+(1−FiO₂)/R]**.
- CaO₂ = **1.34 × Hb (g/dL) × SaO₂ (fraction) + 0.003 × PaO₂ (mmHg)**, yielding mL O₂/dL.
- DO₂ = **CaO₂ (mL/dL) × cardiac output (L/min) × 10**, yielding mL O₂/min.
- Hill curve uses n=2.7 and normal P₅₀=26.6 mmHg. Do not interpret model saturation as a measured blood-gas value. Fetal Hb and dyshemoglobins need additional models.
- Arterial PO₂ is estimated using an explicitly assumed A–a gradient. Tissue points are illustrative; venous blood is shown separately. No patient-specific gas exchange or tissue perfusion is predicted.

Sources for further reading:

1. [Collins et al. — Relating oxygen partial pressure, saturation and content (Breathe, 2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4666443/).
2. [Popel — Theory of oxygen transport to tissue](https://pmc.ncbi.nlm.nih.gov/articles/PMC5445261/).
3. [Oxygen transport: basic principles (BMJ)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1114207/).
4. [Neethling — The Oxygen Cascade, anaesthesia refresher course](https://anaesthetics.ukzn.ac.za/wp-content/uploads/2024/09/Oxygen-cascade-Ref-2021.pdf).
5. [Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

Built for Prof Madapura Shashidhara's MMed/DM and Primary FRCA teaching. AI-assisted educational resource: tutor review before use. Formative feedback, not a certified examination or clinical decision tool.
""")


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
    st.caption("Individual assessment · MMed / DM / Primary FRCA")
    with st.sidebar:
        st.header("Assessment workspace")
        learner = st.text_input("Learner code",max_chars=80,key="learner_code",placeholder="e.g. DM-01",help="Use a code rather than a full name. It is captured when an assessment starts.")
        unit = st.radio("Pressure units",["kPa","mmHg"],key="units",horizontal=True)
        unlocked = tutor_access()
        pages = ["Oxygen cascade", "Oxygen transport", "My results"]
        if unlocked:
            pages += ["Teaching room", "Tutor review", "Sources & assumptions"]
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
    elif page=="Tutor review" and unlocked:
        tutor_dashboard()
    elif page=="Sources & assumptions" and unlocked:
        references()


if __name__ == "__main__":
    main()
