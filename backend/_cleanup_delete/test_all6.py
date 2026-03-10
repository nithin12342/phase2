"""Test all 6 demo samples end-to-end through get_fusion_prediction."""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r'c:\Users\thela\OneDrive\Desktop\phase 2\backend')

SAMPLES_DIR = r'c:\Users\thela\OneDrive\Desktop\phase 2\demo_samples'

# Read each sample's metadata and transcript
samples = []
for folder in sorted(os.listdir(SAMPLES_DIR)):
    folder_path = os.path.join(SAMPLES_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    meta_path = os.path.join(folder_path, "metadata.json")
    txt_path = os.path.join(folder_path, "transcript.txt")
    if not os.path.exists(meta_path):
        continue
    with open(meta_path) as f:
        meta = json.load(f)
    transcript = ""
    if os.path.exists(txt_path):
        with open(txt_path, encoding='utf-8', errors='replace') as f:
            transcript = f.read()[:2000]  # First 2000 chars
    samples.append({
        "folder": folder, 
        "meta": meta, 
        "transcript": transcript
    })

import models

results = []
for s in samples:
    meta = s["meta"]
    sv = meta.get("survey_values", {})
    pid = meta.get("participant_id", "?")
    expected = meta.get("expected_result", "?")
    
    print(f"\n{'='*60}")
    print(f"TESTING {s['folder']}")
    print(f"  PID={pid}, Expected={expected}, PHQ8={sv.get('phq8_total', '?')}")
    print(f"  Survey: stress={sv.get('growing_stress')}, habits={sv.get('changes_in_habits')}, "
          f"history={sv.get('mental_health_history')}, family={sv.get('family_history')}, "
          f"coping={sv.get('coping_struggles')}, social={sv.get('social_weakness')}")
    
    # Simulate what main.py does: set_survey_context + get_fusion_prediction
    # The survey_values from metadata is what the user fills in the form
    models.set_survey_context(sv)
    
    # Pass text as raw string (the transcript file content)
    inputs = {"text": s["transcript"]}
    
    result = models.get_fusion_prediction(inputs)
    
    # Extract the prediction line
    for line in result.split('\n'):
        if 'Prediction:' in line:
            prediction = line.strip()
            break
    else:
        prediction = "UNKNOWN"
    
    correct = ("Depression" in prediction and expected == "Depression") or \
              ("Not Depressed" in prediction and expected == "Not Depressed")
    
    status = "[OK]" if correct else "[FAIL]"
    print(f"  RESULT: {prediction} {status}")
    results.append({"pid": pid, "expected": expected, "got": prediction, "correct": correct})

print(f"\n{'='*60}")
print("SUMMARY:")
print(f"{'='*60}")
correct_count = sum(1 for r in results if r["correct"])
for r in results:
    s = "[OK]" if r["correct"] else "[FAIL]"
    print(f"  {s} PID {r['pid']}: Expected={r['expected']}, Got={r['got']}")
print(f"\nAccuracy: {correct_count}/{len(results)}")
