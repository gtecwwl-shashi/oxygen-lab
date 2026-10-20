"""Oxygen Lab: run with `streamlit run app.py`. Python 3.11+.

Formative teaching and assessment; values are models, not patient advice.
All learner state is session-local. No persistent learner database is used.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import hmac
import hashlib
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
    "cascade": {1: "Draw and label the cascade", 2: "Explain each fall in PO₂", 3: "Use the alveolar gas equation", 4: "Apply altitude and gas-exchange changes"},
    "transport": {1: "Curve shape and cooperative binding", 2: "P₅₀ and right/left shifts", 3: "Loading, unloading and Bohr/Haldane effects", 4: "Pressure, saturation, content and delivery"},
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


def tutor_password():
    value = os.environ.get("TUTOR_PASSWORD", "")
    if not value:
        try:
            value = st.secrets.get("TUTOR_PASSWORD", "")
        except (FileNotFoundError, KeyError):
            value = ""
    return value if isinstance(value, str) and len(value) >= 12 else ""


def is_tutor():
    password = tutor_password()
    if not password:
        return False
    fingerprint = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return hmac.compare_digest(str(st.session_state.get("tutor_token", "")), fingerprint)


def require_tutor():
    if not is_tutor():
        st.error("Tutor sign-in required.")
        st.stop()


def tutor_sign_in():
    if is_tutor():
        st.success("Tutor signed in")
        if st.button("Lock tutor access", key="lock_tutor"):
            st.session_state.pop("tutor_token", None)
            for key in ("page", "mode_cascade", "mode_transport"):
                st.session_state.pop(key, None)
            st.rerun()
        return
    with st.expander("Tutor sign-in"):
        with st.form("tutor_login", clear_on_submit=True):
            entered = st.text_input("Tutor password", type="password", key="tutor_password_input", max_chars=256)
            submitted = st.form_submit_button("Unlock tutor access")
        if submitted:
            if time.time() < st.session_state.get("tutor_retry_after", 0):
                st.error("Please wait briefly before trying again.")
                return
            password = tutor_password()
            if password and hmac.compare_digest(entered.encode("utf-8"), password.encode("utf-8")):
                st.session_state["tutor_token"] = hashlib.sha256(password.encode("utf-8")).hexdigest()
                st.session_state["tutor_failures"] = 0
                for key in ("page", "mode_cascade", "mode_transport"):
                    st.session_state.pop(key, None)
                st.rerun()
            failures = st.session_state.get("tutor_failures", 0) + 1
            st.session_state["tutor_failures"] = failures
            st.session_state["tutor_retry_after"] = time.time() + min(60, 2 ** min(failures, 6))
            st.error("Tutor access could not be unlocked. Check your password and configuration.")


def student_submission(records):
    # No key, explanations, per-item marks or scores enter student downloads.
    fields = ("learner_code", "topic", "attempt_id", "submitted_utc", "duration_seconds", "pressure_unit")
    attempts = []
    for record in records:
        clean = {key: record[key] for key in fields}
        clean["items"] = [{"id": item["id"], "response": item["response"]} for item in record["items"]]
        attempts.append(clean)
    return {"schema_version": 1, "report_type": "responses_only", "attempts": attempts}


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
    if topic == "transport":
        q.extend([
            mcq('t_axes', 2, 'Which axes should you label on an oxyhaemoglobin dissociation curve?', ['PO₂ (kPa) horizontally; haemoglobin saturation (%) vertically', 'Haemoglobin concentration horizontally; PO₂ vertically', 'Cardiac output horizontally; saturation vertically', 'Saturation horizontally; oxygen delivery vertically'], 'PO₂ (kPa) horizontally; haemoglobin saturation (%) vertically', 'The curve relates oxygen partial pressure to haemoglobin saturation. It does not directly plot oxygen content or delivery. Always label units and state reference conditions.'),
            mcq('t_steep', 2, 'What does the steep part of the dissociation curve help the tissues do?', ['Extract a substantial amount of oxygen for a relatively small fall in PO₂', 'Maintain identical saturation at every tissue PO₂', 'Increase the amount of haemoglobin in each red cell', 'Prevent oxygen leaving haemoglobin'], 'Extract a substantial amount of oxygen for a relatively small fall in PO₂', 'Along the steep region, a modest fall in PO₂ produces a relatively large fall in saturation. Oxygen is released from haemoglobin and can diffuse into the tissues. This is movement along a curve, not necessarily a shift of the curve.'),
            mcq('t_left', 2, 'At the same PO₂, what does a left-shifted curve mean?', ['Higher oxygen affinity, lower P₅₀ and higher saturation', 'Lower oxygen affinity, higher P₅₀ and lower saturation', 'Higher haemoglobin concentration with unchanged affinity', 'Higher cardiac output with unchanged affinity'], 'Higher oxygen affinity, lower P₅₀ and higher saturation', 'A left shift means haemoglobin binds oxygen more readily: less PO₂ is needed for 50% saturation. At a fixed tissue PO₂ it holds onto more oxygen, which can make unloading more difficult. Affinity and total oxygen delivery are different concepts.'),
            mcq('t_cold', 2, 'A patient cools from 37°C to 32°C. With pH, CO₂ and 2,3-DPG unchanged, which change is expected?', ['A left shift with increased oxygen affinity', 'A right shift with decreased oxygen affinity', 'A rise in haemoglobin concentration without any affinity change', 'A right shift with decreased P₅₀'], 'A left shift with increased oxygen affinity', 'Lower temperature favours a left shift and lower P₅₀. Higher temperature has the opposite effect. The statement concerns haemoglobin affinity with the other variables held constant.'),
            mcq('t_alkalosis', 2, 'Blood pH rises from 7.40 to 7.55 while temperature and 2,3-DPG remain unchanged. What happens to the oxygen dissociation curve?', ['It shifts left, increasing oxygen affinity', 'It shifts right, decreasing oxygen affinity', 'Its shift is determined only by haemoglobin concentration', 'Its P₅₀ rises because oxygen affinity increases'], 'It shifts left, increasing oxygen affinity', 'Higher pH means fewer hydrogen ions. This favours higher oxygen affinity and a left shift. Lower pH favours a right shift and oxygen unloading: this pH dependence is part of the Bohr effect.'),
            mcq('t_dpg', 2, 'Why does an increase in red-cell 2,3-DPG shift the curve to the right?', ['It stabilises deoxygenated haemoglobin, reducing oxygen affinity', 'It increases the dissolved oxygen pressure directly', 'It converts adult haemoglobin into fetal haemoglobin', 'It adds extra oxygen-binding sites to haemoglobin'], 'It stabilises deoxygenated haemoglobin, reducing oxygen affinity', '2,3-DPG, also called 2,3-BPG, binds preferentially to deoxygenated adult haemoglobin and stabilises its lower-affinity state. More PO₂ is then needed for 50% saturation, so P₅₀ rises. This can aid unloading.'),
            mcq('t_along', 2, 'PO₂ falls, but pH, temperature, CO₂, 2,3-DPG and haemoglobin type stay the same. How should you describe the resulting fall in saturation?', ['Movement down the same dissociation curve', 'A right shift of the entire curve', 'A left shift of the entire curve', 'A necessary change in P₅₀'], 'Movement down the same dissociation curve', 'Changing PO₂ alone moves the operating point along the existing curve. A shift means saturation at the same PO₂ has changed because affinity has changed. Do not call every fall in saturation a right shift.'),
            mcq('t_hbf', 2, 'Why does fetal haemoglobin have a higher oxygen affinity than adult haemoglobin in red cells?', ['Its gamma chains interact less strongly with 2,3-DPG', 'It has more than four oxygen-binding sites', 'It contains no haem groups', 'It binds 2,3-DPG more strongly than adult haemoglobin'], 'Its gamma chains interact less strongly with 2,3-DPG', 'Fetal haemoglobin has gamma rather than beta chains and interacts less strongly with 2,3-DPG. Its curve is therefore relatively left-shifted, helping oxygen uptake at the lower PO₂ found in the placenta. This intrinsic difference is separate from the dynamic double Bohr effect.'),
            mcq('t_stored', 2, 'During storage, red-cell 2,3-DPG falls. What is the expected effect on oxygen affinity?', ['Affinity increases and the curve shifts left', 'Affinity decreases and the curve shifts right', 'The number of oxygen-binding sites increases', 'Haemoglobin loses all ability to carry oxygen'], 'Affinity increases and the curve shifts left', 'Reduced 2,3-DPG means less stabilisation of deoxygenated haemoglobin, so affinity increases and P₅₀ falls. The effect on an actual transfused patient also depends on other physiological factors; stored red cells can still carry oxygen.'),
            mcq('t_co', 1, 'How does carbon monoxide impair oxygen carriage and unloading?', ['It occupies haemoglobin sites and increases oxygen affinity at remaining sites', 'It only lowers dissolved oxygen without affecting haemoglobin', 'It increases available oxygen-binding capacity', 'It shifts the remaining oxygen-binding sites to lower affinity'], 'It occupies haemoglobin sites and increases oxygen affinity at remaining sites', 'Carbon monoxide reduces the haemoglobin capacity available for oxygen and favours a left shift at the remaining oxygen-binding sites, impairing unloading. PaO₂ can remain normal because it reflects dissolved oxygen. A normal PaO₂ therefore does not exclude impaired oxygen carriage.'),
            mcq('t_bohr', 2, 'What is the Bohr effect?', ['Increased CO₂ and/or reduced pH lowers haemoglobin oxygen affinity', 'Oxygenation lowers the amount of CO₂ blood can carry', 'A fall in haemoglobin concentration lowers oxygen content', 'Increased cardiac output increases oxygen delivery'], 'Increased CO₂ and/or reduced pH lowers haemoglobin oxygen affinity', 'The Bohr effect describes how CO₂ and hydrogen ions affect oxygen binding. In active tissues, added CO₂ and increased acidity favour a right shift and oxygen release. Remember the direction: CO₂/pH changes influence O₂ affinity.'),
            mcq('t_haldane', 2, 'At the same PCO₂, deoxygenated blood can carry more CO₂ than oxygenated blood. What is this called?', ['The Haldane effect', 'The Bohr effect', 'Positive cooperative oxygen binding', 'The oxygen cascade'], 'The Haldane effect', 'The Haldane effect describes how haemoglobin oxygenation affects CO₂ carriage. Deoxygenated haemoglobin buffers hydrogen ions more effectively and forms carbamino compounds more readily, helping blood carry CO₂ from tissues. CO₂ carriage includes bicarbonate as well as directly bound CO₂.'),
            mcq('t_haldane_lung', 2, 'How does oxygen loading in the lungs help carbon dioxide removal?', ['Oxygenation reduces blood CO₂-carrying capacity and promotes CO₂ release', 'Oxygenation makes haemoglobin retain more CO₂ at the same PCO₂', 'Oxygenation converts CO₂ into O₂', 'Oxygenation prevents bicarbonate conversion back to CO₂'], 'Oxygenation reduces blood CO₂-carrying capacity and promotes CO₂ release', 'This is the Haldane effect in the lungs. Oxygenated haemoglobin binds fewer hydrogen ions and less carbamino CO₂. Released hydrogen ions combine with bicarbonate, helping regenerate CO₂ for exhalation. This complements the diffusion gradient towards alveolar gas.'),
            mcq('t_compare_effects', 2, 'Which statement correctly distinguishes the Bohr and Haldane effects?', ['Bohr: CO₂/pH affects O₂ binding; Haldane: oxygenation affects CO₂ carriage', 'Bohr: oxygenation affects CO₂ carriage; Haldane: pH affects O₂ binding', 'Both effects describe only changes in haemoglobin concentration', 'Both effects describe only diffusion through the alveolar membrane'], 'Bohr: CO₂/pH affects O₂ binding; Haldane: oxygenation affects CO₂ carriage', 'Keep the causal direction clear. Bohr asks how CO₂ and acidity alter haemoglobin affinity for oxygen. Haldane asks how oxygenation alters blood CO₂-carrying capacity. They work together during gas exchange but are not the same effect.'),
            mcq('t_double_mother', 2, 'In the placenta, maternal blood takes up CO₂ from the fetus. What does the local rise in CO₂ and acidity favour?', ['A maternal right shift and oxygen unloading', 'A maternal left shift and stronger oxygen retention', 'A conversion of maternal haemoglobin into fetal haemoglobin', 'An increase in maternal oxygen affinity caused by CO₂'], 'A maternal right shift and oxygen unloading', 'CO₂ moves from fetal towards maternal blood. Maternal blood locally becomes more acidic, reducing haemoglobin oxygen affinity and favouring release of oxygen. This is the maternal half of the double Bohr effect; maternal and fetal blood remain separate circulations.'),
            mcq('t_double_fetus', 2, 'Fetal blood loses CO₂ across the placenta. What does the resulting local rise in fetal pH favour?', ['A fetal left shift and oxygen loading', 'A fetal right shift and oxygen unloading', 'A rise in fetal P₅₀ and reduced affinity', 'A loss of all haemoglobin oxygen-binding sites'], 'A fetal left shift and oxygen loading', 'As fetal blood loses CO₂, its pH rises locally, increasing haemoglobin oxygen affinity. This favours oxygen uptake. This is the fetal half of the double Bohr effect and adds to the intrinsically higher affinity of fetal haemoglobin.'),
            mcq('t_double_bohr', 2, 'Why is the placental effect called the double Bohr effect?', ['CO₂ transfer promotes maternal O₂ unloading and fetal O₂ loading at the same time', 'Both maternal and fetal oxygen curves shift right', 'Both maternal and fetal oxygen curves shift left', 'Maternal blood directly mixes with fetal blood to exchange haemoglobin'], 'CO₂ transfer promotes maternal O₂ unloading and fetal O₂ loading at the same time', 'One CO₂ transfer produces two complementary Bohr responses: maternal CO₂ rises and pH falls, favouring a right shift; fetal CO₂ falls and pH rises, favouring a left shift. Both favour oxygen transfer from mother to fetus. The effect supplements, rather than replaces, the oxygen partial-pressure gradient.'),
            mcq('t_double_haldane', 2, 'What does the double Haldane effect help achieve in the placenta?', ['Fetal oxygenation promotes CO₂ release while maternal deoxygenation promotes CO₂ uptake', 'Both fetal and maternal oxygenation prevent CO₂ transfer', 'CO₂ transfer changes maternal haemoglobin into fetal haemoglobin', 'It explains only the higher oxygen affinity of fetal haemoglobin'], 'Fetal oxygenation promotes CO₂ release while maternal deoxygenation promotes CO₂ uptake', 'Fetal blood gains oxygen, reducing its CO₂-carrying capacity and favouring CO₂ release. Maternal blood loses oxygen, increasing its CO₂-carrying capacity and favouring uptake. Together these Haldane responses assist CO₂ transfer from fetus to mother. Double Bohr concerns oxygen affinity; double Haldane concerns CO₂ carriage.'),
        ])
        order = ['t_pressure', 't_dissolved', 't_anaemia', 't_axes', 't_cooperative', 't_plateau', 't_steep', 't_p50', 't_along', 't_shift', 't_left', 't_factors', 't_cold', 't_alkalosis', 't_dpg', 't_stored', 't_hbf', 't_co', 't_bohr', 't_haldane', 't_haldane_lung', 't_compare_effects', 't_double_mother', 't_double_fetus', 't_double_bohr', 't_double_haldane', 't_anchor', 't_content', 't_delivery', 't_units']
        by_id = {item["id"]: item for item in q}
        q = [by_id[qid] for qid in order]
    if topic == "transport":
        q.extend([
            mcq('t_hemes', 2, 'How many oxygen molecules can one fully oxygenated normal adult haemoglobin molecule carry?', ['Four: one O₂ molecule at each of four haem groups', 'One: all four chains share one binding site', 'Two: only the beta chains bind oxygen', 'Eight: each haem binds two O₂ molecules'], 'Four: one O₂ molecule at each of four haem groups', 'Adult HbA is a tetramer with two alpha and two beta chains. Each chain contains a haem group with ferrous iron, and each haem can bind one oxygen molecule. One haemoglobin molecule can therefore bind four O₂ molecules, meaning eight oxygen atoms.'),
            mcq('t_occupancy', 2, 'What happens to haemoglobin as oxygen binds progressively to its haem groups?', ['Binding favours the higher-affinity R state, making further oxygen binding easier', 'Each oxygen molecule destroys a neighbouring haem group', 'The molecule becomes a single chain like myoglobin', 'The remaining sites necessarily become less able to bind oxygen'], 'Binding favours the higher-affinity R state, making further oxygen binding easier', 'Oxygen binding changes the local iron–haem environment and the interactions between globin subunits, favouring the relaxed, higher-affinity R state over the tense, lower-affinity T state. This allosteric communication explains positive cooperativity and the sigmoid curve; the haem groups do not pass oxygen directly between one another.'),
            mcq('t_p75', 2, 'On a normal adult curve at 37°C and pH 7.4, which PO₂ is closest to that giving 75% saturation (P₇₅)?', ['5.3 kPa', '3.5 kPa', '8.0 kPa', '13.3 kPa'], '5.3 kPa', 'The usual teaching point is approximately 5.3 kPa at 75% saturation, near normal mixed venous values at rest. This is an approximate reference, not a fixed value in every organ. Lower tissue PO₂ encourages oxygen release on the steep part of the curve.'),
            mcq('t_p90', 2, 'On a normal adult curve, which PO₂ is closest to that giving 90% saturation (P₉₀)?', ['8.0 kPa', '3.5 kPa', '5.3 kPa', '20.0 kPa'], '8.0 kPa', 'Approximately 8.0 kPa corresponds to 90% saturation under standard teaching conditions. Below this region, further reductions in PO₂ can cause a relatively rapid fall in saturation. Temperature, pH and other affinity changes alter these relationships.'),
            mcq('t_p100', 2, 'Is there a single universal PO₂ that should be quoted as P₁₀₀, meaning exactly 100% haemoglobin saturation?', ['No; the curve approaches full saturation and the value depends on conditions and measurement', 'Yes; 13.3 kPa always means exactly 100% saturation', 'Yes; 8.0 kPa always means exactly 100% saturation', 'Yes; 3.5 kPa always means exactly 100% saturation'], 'No; the curve approaches full saturation and the value depends on conditions and measurement', 'For standard teaching, PaO₂ around 13.3 kPa corresponds to roughly 97–98% saturation, not a guaranteed exact 100%. The plateau approaches full saturation, and measured displays can round to 100%. Do not invent a universal finite P₁₀₀; state the saturation, PO₂ and reference conditions.'),
            mcq('t_lung_tissue', 2, 'How do local gas exchange conditions help loading in lungs and unloading in metabolically active tissues?', ['Tissue CO₂ gain and lower pH favour unloading; pulmonary CO₂ loss and higher pH favour loading', 'The curve is permanently right-shifted in lungs and left-shifted in active tissues', 'Oxygen can load only if haemoglobin concentration increases each breath', 'CO₂ loss in the lungs lowers pH and reduces oxygen affinity'], 'Tissue CO₂ gain and lower pH favour unloading; pulmonary CO₂ loss and higher pH favour loading', 'Active tissues add CO₂ and hydrogen ions, favouring reduced oxygen affinity through the Bohr effect; local warming may further favour unloading. In the lungs, CO₂ removal raises pH and favours affinity and loading. PO₂ also changes along the curve; both changes in gradient and local affinity contribute. HbF is prominent in newborns and is progressively replaced by HbA after birth, so infant affinity depends partly on age and the mixture of haemoglobins.'),
        ])
        by_id = {item["id"]: item for item in q}
        q = [by_id[qid] for qid in ['t_pressure', 't_dissolved', 't_anaemia', 't_hemes', 't_occupancy', 't_axes', 't_cooperative', 't_plateau', 't_steep', 't_p50', 't_p75', 't_p90', 't_p100', 't_along', 't_shift', 't_left', 't_factors', 't_cold', 't_alkalosis', 't_dpg', 't_stored', 't_hbf', 't_co', 't_bohr', 't_lung_tissue', 't_haldane', 't_haldane_lung', 't_compare_effects', 't_double_mother', 't_double_fetus', 't_double_bohr', 't_double_haldane', 't_anchor', 't_content', 't_delivery', 't_units']]
    if topic == "cascade":
        q.extend([
            mcq('c_aadiff', 3, 'PAO₂ is 13.4 kPa and PaO₂ is 10.0 kPa. What is the alveolar–arterial oxygen difference?', ['3.4 kPa', '23.4 kPa', '0.75 kPa', '13.4 kPa'], '3.4 kPa', 'Subtract arterial from alveolar PO₂: 13.4 − 10.0 = 3.4 kPa. A is alveolar; a is arterial. Interpretation depends on age, FiO₂ and the clinical context.'),
            mcq('c_purehypov', 2, 'In isolated hypoventilation with otherwise normal lungs, what happens to the A–a oxygen difference?', ['It may remain normal despite reduced arterial PO₂', 'It must become very large', 'It becomes equal to atmospheric pressure', 'It measures cardiac output rather than oxygen exchange'], 'It may remain normal despite reduced arterial PO₂', 'Hypoventilation lowers alveolar PO₂, and arterial PO₂ falls with it. A wide A–a difference suggests an additional gas-exchange abnormality such as V/Q mismatch, shunt or diffusion limitation.'),
            mcq('c_lowvq', 2, 'What happens to gas in an alveolus receiving too little ventilation for its blood flow?', ['Its PO₂ falls and its PCO₂ rises towards venous values', 'Its PO₂ rises towards dry-air values', 'Its gas becomes identical to inspired oxygen', 'Its PO₂ is determined only by haemoglobin concentration'], 'Its PO₂ falls and its PCO₂ rises towards venous values', 'In a low-V/Q unit, relatively little fresh oxygen arrives compared with uptake by blood; CO₂ removal is also reduced. Blood leaving such units can lower arterial oxygenation when it mixes with blood from better ventilated regions.'),
            mcq('c_shunt', 2, 'Why may a substantial right-to-left shunt respond poorly to increasing inspired oxygen?', ['Some blood bypasses ventilated alveoli and cannot load oxygen there', 'High FiO₂ lowers alveolar PO₂', 'Shunted blood contains no haemoglobin', 'Oxygen cannot dissolve in blood at any pressure'], 'Some blood bypasses ventilated alveoli and cannot load oxygen there', 'Oxygen raises PO₂ in ventilated units, but the shunted fraction does not contact that alveolar gas. Its admixture limits the arterial response. This is a physiological explanation, not a claim that supplemental oxygen never helps.'),
            mcq('c_anaemia_pa', 2, 'In isolated anaemia with normal ventilation and lung gas exchange, what may happen to PaO₂?', ['It can remain normal despite reduced oxygen content', 'It must fall in proportion to haemoglobin concentration', 'It necessarily rises to inspired PO₂', 'It becomes equal to P₅₀'], 'It can remain normal despite reduced oxygen content', 'PaO₂ measures dissolved oxygen tension, not the amount carried on haemoglobin. Anaemia can lower oxygen content and delivery without lowering PaO₂. Distinguish hypoxaemia from low oxygen content and tissue hypoxia.'),
            mcq('c_diffusion', 2, 'Which change directly reduces the rate of oxygen diffusion across the alveolar–capillary membrane, other factors unchanged?', ['An increase in membrane thickness', 'An increase in exchange surface area', 'An increase in the oxygen partial-pressure gradient', 'A decrease in membrane thickness'], 'An increase in membrane thickness', 'Fick’s diffusion law relates transfer to surface area and the pressure gradient, and inversely to barrier thickness. A thicker barrier impedes diffusion. This is distinct from the Fick principle used to relate cardiac output to oxygen consumption.'),
            mcq('c_exercise', 2, 'If local oxygen use rises without a matching rise in delivery, what happens to tissue oxygen extraction?', ['It initially rises and venous oxygen content falls', 'It stops because haemoglobin cannot unload oxygen', 'Venous oxygen content must rise', 'Arterial PO₂ must become zero immediately'], 'It initially rises and venous oxygen content falls', 'Tissues initially meet greater demand by extracting a larger fraction of delivered oxygen. Venous oxygen content and often venous PO₂ fall. Extraction has limits; if delivery cannot meet demand, tissue hypoxia can develop.'),
            mcq('c_altadapt', 3, 'Why does hyperventilation partly improve alveolar PO₂ after ascent to altitude?', ['It lowers PaCO₂, reducing the subtraction term in the alveolar gas equation', 'It restores sea-level atmospheric pressure', 'It raises ambient oxygen fraction above 21%', 'It makes water vapour pressure zero'], 'It lowers PaCO₂, reducing the subtraction term in the alveolar gas equation', 'At a fixed altitude and FiO₂, lower PaCO₂ increases calculated PAO₂ because PaCO₂/R is smaller. This is compensation: inspired PO₂ is still reduced because barometric pressure remains lower.'),
            numeric('c_alt1', 1, 'At approximately 1000 m, use the supplied PB = 89.9 kPa. Room air FiO₂ = 0.21 and PH₂O = 6.3 kPa. Calculate humidified PIO₂ = FiO₂ × (PB − PH₂O).', 17.556, 0.2, 'kPa', 'PIO₂ = 0.21 × (89.9 − 6.3) = 0.21 × 83.6 = 17.556 kPa, approximately 17.6 kPa. The oxygen fraction stays about 21%, but its partial pressure is lower than at sea level.'),
            numeric('c_alt2', 3, 'At approximately 2000 m, PB = 79.5 kPa. Use FiO₂ = 0.21, PH₂O = 6.3 kPa, PaCO₂ = 5.3 kPa and R = 0.8. Calculate PAO₂ = FiO₂ × (PB − PH₂O) − PaCO₂/R. Assume no ventilatory adaptation.', 8.747, 0.2, 'kPa', 'Inspired term = 0.21 × 73.2 = 15.372 kPa. CO₂ term = 5.3/0.8 = 6.625 kPa. PAO₂ = 8.747, approximately 8.7 kPa. This stipulated fixed-CO₂ example is not a prediction of every person’s response to altitude.'),
            numeric('c_alt3', 3, 'At approximately 3000 m, PB = 70.1 kPa. A hyperventilating person has PaCO₂ = 3.5 kPa. Use FiO₂ = 0.21, PH₂O = 6.3 kPa and R = 0.8 to calculate PAO₂.', 9.023, 0.2, 'kPa', 'PIO₂ = 0.21 × (70.1 − 6.3) = 13.398 kPa. PAO₂ = 13.398 − 3.5/0.8 = 9.023 kPa. Hyperventilation partly offsets the altitude-related fall. Compare only after checking both PB and PaCO₂.'),
            numeric('c_alt4', 3, 'At the same altitude, PaCO₂ falls from 5.3 to 3.3 kPa. PIO₂ and R = 0.8 stay unchanged. By how many kPa does calculated PAO₂ increase? Use increase = fall in PaCO₂/R.', 2.5, 0.2, 'kPa', 'PaCO₂ falls by 2.0 kPa. The PAO₂ increase is 2.0/0.8 = 2.5 kPa. There is no need to know the altitude pressure for this difference calculation because PIO₂ is held fixed.'),
            numeric('c_hypovnum', 3, 'PIO₂ = 20.0 kPa, PaCO₂ = 8.0 kPa and R = 0.8. Calculate PAO₂ = PIO₂ − PaCO₂/R during hypoventilation.', 10.0, 0.2, 'kPa', 'PAO₂ = 20.0 − 8.0/0.8 = 20.0 − 10.0 = 10.0 kPa. Increased alveolar CO₂ reduces the alveolar oxygen estimate at the same inspired oxygen pressure.'),
            numeric('c_hyperventnum', 3, 'During hyperventilation at sea level, PaCO₂ is 3.2 kPa. Use PIO₂ = 20.0 kPa and R = 0.8. Calculate PAO₂.', 16.0, 0.2, 'kPa', 'PAO₂ = 20.0 − 3.2/0.8 = 20.0 − 4.0 = 16.0 kPa. Lower CO₂ raises alveolar PO₂ when the inspired conditions are unchanged.'),
            numeric('c_supplementnum', 3, 'At sea level, FiO₂ = 0.40, PB = 101.3 kPa, PH₂O = 6.3 kPa, PaCO₂ = 5.3 kPa and R = 0.8. Estimate PAO₂ using the simplified alveolar gas equation.', 31.375, 0.2, 'kPa', 'Inspired term = 0.40 × 95.0 = 38.0 kPa. Subtract 5.3/0.8 = 6.625 kPa, giving 31.375, approximately 31.4 kPa. This is the simplified estimate; the fuller equation differs, particularly at higher FiO₂.'),
            numeric('c_rqnum', 3, 'To isolate the effect of R, hold PIO₂ = 20.0 kPa and PaCO₂ = 5.3 kPa fixed, but use R = 1.0. Calculate PAO₂ = PIO₂ − PaCO₂/R.', 14.7, 0.2, 'kPa', 'PAO₂ = 20.0 − 5.3/1.0 = 14.7 kPa. At these fixed inputs, a higher R reduces the subtraction term. Real metabolic changes can also alter other variables; this question isolates the equation.'),
            numeric('c_dryalt', 1, 'At approximately 4000 m, use PB = 61.6 kPa. For dry room air with FiO₂ = 0.21, calculate PO₂ = FiO₂ × PB. Do not subtract water vapour for dry air.', 12.936, 0.2, 'kPa', 'Dry-air PO₂ = 0.21 × 61.6 = 12.936, approximately 12.9 kPa. This is dry atmospheric PO₂, not humidified inspired, alveolar or arterial PO₂. These compartments must not be interchanged.'),
            mcq("c_venous", 1, "Which value is closest to normal mixed venous PO₂ at rest, and how should it appear on a cascade diagram?", ["5.3 kPa, as a separate returning-blood reference after tissue extraction", "13.3 kPa, as the value in every mitochondrion", "20.0 kPa, between alveoli and arterial blood", "21.3 kPa, as a mandatory tissue diffusion intermediate"], "5.3 kPa, as a separate returning-blood reference after tissue extraction", "Mixed venous PO₂ is commonly about 5.3 kPa in a healthy adult at rest, varying with delivery and extraction. It represents blood returning after tissue extraction, not an extra compartment between tissue capillaries and mitochondria. Tissue and mitochondrial PO₂ are lower and variable."),
        ])
    if topic == "transport":
        q.extend([
            mcq("t_methaemoglobin", 4, "Why can methaemoglobinaemia impair oxygen delivery even when PaO₂ is normal?", ["Ferric iron cannot bind oxygen, and remaining ferrous sites may hold oxygen more tightly", "Methaemoglobin increases the number of useful oxygen-binding sites", "Methaemoglobin lowers oxygen content only by reducing atmospheric pressure", "Methaemoglobin converts dissolved oxygen directly into carbon dioxide"], "Ferric iron cannot bind oxygen, and remaining ferrous sites may hold oxygen more tightly", "In methaemoglobin, haem iron is oxidised from ferrous Fe²⁺ to ferric Fe³⁺ and cannot bind oxygen. Oxygen affinity at remaining functional sites can increase, impairing unloading. PaO₂ still measures dissolved oxygen and may be normal. Co-oximetry measures dyshemoglobin fractions; conventional saturation readings do not reliably describe this problem."),
            mcq("t_hypoxaemia", 4, "How does hypoxaemia differ from isolated anaemia?", ["Hypoxaemia means low arterial PO₂; isolated anaemia can reduce content despite normal PaO₂", "Both always mean PaO₂ below 5.3 kPa", "Anaemia means low oxygen affinity with normal haemoglobin concentration", "Hypoxaemia is defined only by low cardiac output"], "Hypoxaemia means low arterial PO₂; isolated anaemia can reduce content despite normal PaO₂", "Hypoxaemia refers to reduced arterial oxygen partial pressure. Anaemia reduces haemoglobin available to carry oxygen; oxygen tension and saturation can remain normal. Both can reduce content and delivery, but through different mechanisms. Tissue hypoxia is inadequate oxygen availability or use at tissue level, not simply a synonym for low PaO₂."),
        ])
        curve = {"t_hemes", "t_occupancy", "t_axes", "t_cooperative", "t_plateau", "t_steep", "t_along"}
        shifts = {"t_p50", "t_p75", "t_p90", "t_p100", "t_shift", "t_left", "t_factors", "t_cold", "t_alkalosis", "t_dpg", "t_stored", "t_hbf"}
        effects = {"t_bohr", "t_lung_tissue", "t_haldane", "t_haldane_lung", "t_compare_effects", "t_double_mother", "t_double_fetus", "t_double_bohr", "t_double_haldane"}
        for item in q:
            item["objective"] = 1 if item["id"] in curve else 2 if item["id"] in shifts else 3 if item["id"] in effects else 4
    else:
        clinical = {"c_altitude", "c_high", "c_altadapt", "c_aadiff", "c_purehypov", "c_lowvq", "c_shunt", "c_anaemia_pa", "c_diffusion", "c_exercise", "c_alt1", "c_alt2", "c_alt3", "c_alt4", "c_dryalt"}
        for item in q:
            if item["id"] in clinical:
                item["objective"] = 4
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
    if payload.get("report_type") == "responses_only":
        marked = []
        for record in records:
            if not isinstance(record, dict) or record.get("topic") not in TOPICS:
                raise ValueError("Unknown topic")
            bank = make_questions(record["topic"])
            items = record.get("items", [])
            if not isinstance(items, list) or len(items) != len(bank) or any(not isinstance(x, dict) for x in items):
                raise ValueError("Incomplete response file")
            responses = {item.get("id"): item.get("response") for item in items}
            if set(responses) != {q["id"] for q in bank}:
                raise ValueError("Question IDs do not match this test")
            updated = dict(record)
            updated["items"] = [dict(id=q["id"], objective=q["objective"], prompt=q["prompt"], response=responses[q["id"]], expected=q["answer"], earned=grade(q,responses[q["id"]]), explanation=q["explanation"]) for q in bank]
            updated["score"] = sum(x["earned"] for x in updated["items"])
            updated["total"] = len(bank)
            updated["percentage"] = round(100 * updated["score"] / len(bank), 1)
            marked.append(updated)
        records = marked
    for r in records:
        if not isinstance(r,dict) or r.get("topic") not in TOPICS:
            raise ValueError("Unknown topic")
        uuid.UUID(r["attempt_id"])
        if not isinstance(r.get("learner_code"),str) or not 1 <= len(r["learner_code"]) <= 80:
            raise ValueError("Invalid learner code")
        datetime.fromisoformat(r["submitted_utc"])
        if r.get("pressure_unit") not in ("kPa","mmHg"):
            raise ValueError("Invalid pressure unit")
        expected = {q["id"]:q["objective"] for q in make_questions(r["topic"],0)}
        count = len(expected)
        if not isinstance(r.get("items"),list) or len(r["items"]) != count:
            raise ValueError(f"Expected {count} marked items for this topic and question set")
        if {x.get("id") for x in r["items"]} != set(expected):
            raise ValueError("Question IDs are missing or duplicated")
        if any(type(x.get("earned")) is not int or x["earned"] not in (0,1) or x.get("objective") != expected[x["id"]] for x in r["items"]):
            raise ValueError("Invalid item marks or objectives")
        score = sum(x["earned"] for x in r["items"])
        if r.get("score") != score or r.get("total") != count or r.get("percentage") != round(100*score/count,1):
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
    require_tutor()
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
    require_tutor()
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
    require_tutor()
    st.subheader("Tutor interaction · sketch and viva")
    st.caption("Use the same core prompts with each candidate. Allow 15 minutes: sketch 4 minutes, core viva 8 minutes, feedback 3 minutes. Extra prompts are for guided teaching after the core viva.")
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
    st.markdown("**Core viva — use with every candidate**")
    for i, question in enumerate(viva, 1):
        st.write(f"{i}. {question}")
    if topic == "transport":
        with st.expander("Extra teaching prompts: Bohr, Haldane and placental exchange"):
            st.write("Explain the Bohr effect. How is it different from the Haldane effect?")
            st.write("What happens to maternal and fetal oxygen affinity when CO₂ crosses the placenta?")
            st.write("Explain how the double Bohr and double Haldane effects assist placental gas exchange.")
    with st.expander("Tutor/self-review checklist and viva prompts"):
        for i,item in enumerate(rubric):
            st.checkbox(item,key=f"rubric_{topic}_{i}")
    st.text_area("Your explanation or tutor feedback (optional, temporary)",key=f"viva_{topic}")
    notes = st.session_state.get(f"viva_{topic}","")
    st.download_button("Download viva notes",notes,file_name=f"{topic}_viva_notes.txt",mime="text/plain")


def assessment(topic, unit, learner):
    st.subheader("Individual formative assessment")
    st.caption(f"{len(make_questions(topic))} questions · pressures in kPa · numerical tolerances are shown. Everyone receives the same questions. Answers, explanations and marks are reserved for tutor review.")
    key = f"attempt_{topic}"
    attempt = st.session_state.get(key)
    if attempt and attempt.get("bank_version") != "2026-10-20-test-mode":
        del st.session_state[key]
        attempt = None
    if not attempt:
        if st.button("Start assessment",type="primary",key=f"start_{topic}"):
            if not learner.strip():
                st.error("Enter a learner code in the sidebar first.")
                return
            attempt = {"bank_version":"2026-10-20-test-mode","id":str(uuid.uuid4()),"learner":learner.strip(),"unit":unit,"started":time.time(),"questions":make_questions(topic,0,"kPa"),"result":None}
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
            submitted = st.form_submit_button("Submit assessment",type="primary")
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
    if not is_tutor():
        st.success("Assessment submitted. Your tutor will review your responses.")
        st.download_button("Download submission for tutor", json.dumps(student_submission([result]), ensure_ascii=False, indent=2), file_name=f"oxygen_{topic}_{attempt['id'][:8]}_submission.json", mime="application/json")
        return
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
    if not is_tutor():
        st.header("My submissions")
        records = st.session_state.get("results", [])
        if not records:
            st.info("No submitted assessments in this session.")
            return
        st.table([{"Learner": r["learner_code"], "Topic": TOPICS[r["topic"]], "Status": "Submitted"} for r in records])
        st.download_button("Download my submissions for tutor", json.dumps(student_submission(records), ensure_ascii=False, indent=2), file_name="oxygen_submissions.json", mime="application/json")
        return
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
    require_tutor()
    st.header("Group results")
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
    require_tutor()
    st.header(TOPICS[topic]+" · 15 minutes")
    rows = [("0–3 min","Draw and label typical values"),("3–6 min","Explain the fall at each stage"),("6–10 min","Gas equation and worked example"),("10–15 min","Individual assessment; sketch/viva discussion")] if topic=="cascade" else [("0–2 min","Pressure, saturation and content"),("2–6 min","Dissociation curve, P₅₀ and shifts"),("6–10 min","Content and delivery worked example"),("10–15 min","Individual assessment; sketch/viva discussion")]
    with st.expander("Learning objectives and teaching sequence"):
        for i,label in OBJECTIVES[topic].items():
            st.write(f"{i}. {label}")
        st.table([{"Time":a,"Focus":b} for a,b in rows])
        st.caption("Use selected questions during teaching. The full assessment is for independent practice: 30 cascade questions or 38 transport questions. Allow roughly 25–35 minutes for either full set, adjusted to the learner.")


def references():
    require_tutor()
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
3. [Placental gas exchange: Bohr and Haldane effects (2023)](https://pubmed.ncbi.nlm.nih.gov/36527666/).
4. [Experimental Haldane effect](https://pubmed.ncbi.nlm.nih.gov/1757157/).
5. [Fetal/adult haemoglobin and 2,3-DPG](https://pubmed.ncbi.nlm.nih.gov/5009828/).
6. [Carbon monoxide and oxygen binding](https://pubmed.ncbi.nlm.nih.gov/3976/).
7. [Oxygen transport: basic principles (BMJ)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1114207/).
4. [Neethling — The Oxygen Cascade, anaesthesia refresher course](https://anaesthetics.ukzn.ac.za/wp-content/uploads/2024/09/Oxygen-cascade-Ref-2021.pdf).
5. [Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

Built for Prof Madapura Shashidhara's MMed/DM and Primary FRCA teaching. AI-assisted educational resource: tutor review before use. Formative feedback, not a certified examination or clinical decision tool.
""")


def marking_answers(topic):
    require_tutor()
    st.header(TOPICS[topic] + " · tutor answers")
    for i, question in enumerate(make_questions(topic), 1):
        with st.expander(f"{i}. {question['prompt']}"):
            st.write(f"Correct answer: {question['answer']}")
            st.write(question["explanation"])


def main():
    st.set_page_config(page_title="Oxygen Lab · Primary FRCA", page_icon="🫁", layout="wide")
    st.title("Oxygen Lab")
    st.caption("Primary FRCA · oxygen cascade and transport")
    with st.sidebar:
        st.header("Assessment workspace")
        tutor_sign_in()
        tutor = is_tutor()
        learner = st.text_input("Learner code", max_chars=80, key="learner_code", placeholder="e.g. DM-01")
        unit = "kPa"
        st.caption("All pressures use kPa.")
        pages = ["Oxygen cascade", "Oxygen transport", "My submissions"]
        if tutor:
            pages += ["Group results", "Marking & answers", "Sources & assumptions"]
        if st.session_state.get("page") not in pages:
            st.session_state["page"] = pages[0]
        page = st.radio("Open", pages, key="page")
        st.caption("Download your submission before closing this browser. Responses are session-local.")
        if tutor and st.button("Clear session for next learner", key="clear_session"):
            for key in list(st.session_state):
                del st.session_state[key]
            st.rerun()
    if page in ("Oxygen cascade", "Oxygen transport"):
        topic = "cascade" if page == "Oxygen cascade" else "transport"
        if not tutor:
            st.header(TOPICS[topic] + " · assessment")
            assessment(topic, unit, learner)
            return
        lesson(topic)
        mode = st.radio("Activity", ["Assessment", "Interaction", "Tutor interaction"], horizontal=True, key=f"mode_{topic}")
        if mode == "Interaction":
            cascade_lab(unit) if topic == "cascade" else transport_lab(unit)
        elif mode == "Tutor interaction":
            sketch_viva(topic)
        else:
            assessment(topic, unit, learner)
    elif page == "My submissions":
        reports()
    elif page == "Group results":
        tutor_dashboard()
    elif page == "Marking & answers":
        require_tutor()
        topic = st.selectbox("Topic", list(TOPICS), format_func=lambda value: TOPICS[value])
        marking_answers(topic)
    else:
        references()


if __name__ == "__main__":
    main()
