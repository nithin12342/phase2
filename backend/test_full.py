import requests
import json
import sys

url = 'http://localhost:8000/api/v1/submit-survey'
TIMEOUT = 300  # 5 minutes for HF cold starts

def test_sample(pid, dir_name, is_depressed):
    sample_dir = fr'C:\Users\thela\OneDrive\Desktop\phase 2\demo_samples\{dir_name}'
    label = "Depressed" if is_depressed else "Not Depressed"
    
    survey_data = {
        'gender': 'Unknown', 'country': 'Unknown', 'occupation': 'Unknown',
        'days_indoors': '1-14 days', 'is_self_employed': 'No', 'self_employed_date': '',
        'growing_stress': 'Yes' if is_depressed else 'No',
        'changes_habits': 'Yes' if is_depressed else 'No',
        'mental_health_history': 'Yes' if is_depressed else 'No',
        'family_history': 'Yes' if is_depressed else 'No',
        'treatment_sought': 'No',
        'mood_swings': 'High' if is_depressed else 'Low',
        'work_interest': 'No' if is_depressed else 'Yes',
        'social_weakness': 'Yes' if is_depressed else 'No',
        'coping_struggles': 'Yes' if is_depressed else 'No',
        'interview_attended': 'No', 'care_options_awareness': 'No'
    }

    files = {
        'audio': ('audio.wav', open(sample_dir + r'\audio.wav', 'rb'), 'audio/wav'),
        'video': ('video.mp4', open(sample_dir + r'\video_clip.mp4', 'rb'), 'video/mp4'),
        'photo': ('photo.jpg', open(sample_dir + r'\face_frame.jpg', 'rb'), 'image/jpeg'),
        'doc': ('doc.txt', open(sample_dir + r'\transcript.txt', 'rb'), 'text/plain')
    }
    
    print(f"Testing PID {pid} ({label})...", flush=True)
    resp = requests.post(url, data={'survey_data': json.dumps(survey_data)}, files=files, timeout=TIMEOUT)
    res = resp.json().get('depression_risk', '')
    
    # Extract the Prediction line
    for line in res.split('\n'):
        if line.startswith('Prediction:'):
            prediction = line.replace('Prediction: ', '')
            correct = "CORRECT" if (("Depression" in prediction and is_depressed) or ("Not Depressed" in prediction and not is_depressed)) else "WRONG"
            print(f"PID {pid} | Ground Truth: {label:15} | Model: {prediction:15} | {correct}")
            return
        elif 'Depression Risk:' in line:
            print(f"PID {pid} | FALLBACK: {line.strip()}")
            return

    print(f"PID {pid} | Full output:\n{res}")

# Test just one sample at a time based on arg
if len(sys.argv) > 1:
    pid = int(sys.argv[1])
    if pid == 303:
        test_sample(303, 'sample_4_Not_Depressed_PID303', False)
    elif pid == 346:
        test_sample(346, 'sample_1_Depression_PID346', True)
    elif pid == 308:
        test_sample(308, 'sample_2_Depression_PID308', True)
else:
    test_sample(303, 'sample_4_Not_Depressed_PID303', False)
    test_sample(346, 'sample_1_Depression_PID346', True)
    test_sample(308, 'sample_2_Depression_PID308', True)
