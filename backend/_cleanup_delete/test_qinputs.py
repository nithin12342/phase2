import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append(r'c:\Users\thela\OneDrive\Desktop\phase 2\backend')
import models

def run_sample_test(name, text, survey_values):
    print(f"\n--- TESTING {name} ---")
    models.set_survey_context(survey_values)
    results = models.get_fusion_prediction({"text": text})
    print(f"RESULT: {results}")

# PID 303: Not Depressed
survey_303 = {
    "phq8_total": 0.0,
    "growing_stress": "No",
    "changes_habits": "No",
    "mental_health_history": "No",
    "family_history": "No"
}
text_303 = "I feel fine honestly. Everything is going well and I'm happy."

# PID 346: Depression
survey_346 = {
    "phq8_total": 23.0,
    "growing_stress": "Yes",
    "changes_habits": "Yes",
    "mental_health_history": "Yes",
    "family_history": "Yes"
}
text_346 = "I've been feeling very low lately. I can't sleep, I've lost my appetite, and I feel hopeless."

run_sample_test("PID 303 (Expected: Not Depressed)", text_303, survey_303)
run_sample_test("PID 346 (Expected: Depression)", text_346, survey_346)
