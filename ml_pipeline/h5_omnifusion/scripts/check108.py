import h5py
import numpy as np

f = h5py.File('features_account4.h5', 'r')
pids = list(f.keys())
daic = [p for p in pids if not p.startswith('t_')]
eatd = [p for p in pids if p.startswith('t_')]

print("="*70)
print("108-FEATURE COMPLIANCE LIST - features_account4.h5")
print("="*70)
print(f"Participants: {len(pids)} ({len(daic)} DAIC-WOZ, {len(eatd)} EATD)")
print()

sample = daic[0] if daic else pids[0]
print(f"SAMPLE: {sample}")
print("-"*70)
datasets = list(f[sample].keys())
attrs = list(f[sample].attrs.keys())
print(f"Datasets ({len(datasets)}): {datasets}")
print(f"Attrs ({len(attrs)}): {attrs}")
print()

ALL_108_FEATURES = [
    'audio_embedding', 'text_embedding', 'video_embedding', 'face_embedding',
    'tabular_embedding', 'fusion_embedding', 'audio', 'text', 'video', 'face',
    'tabular', 'au_embedding', 'audio_egemaps_embedding',
    'quality_scores', 'phq8_score', 'prosodic_features', 'linguistic_features',
    'sentiment_scores', 'gaze_features', 'optical_flow', 'supplementary_features',
    'au_mean', 'au_std',
    'audio_pause_ratio', 'audio_pause_mean', 'audio_speaking_rate', 'audio_phonation_ratio',
    'audio_f0_mean', 'audio_f0_std', 'audio_f0_range', 'audio_jitter', 'audio_shimmer',
    'audio_breath_count', 'audio_sigh_count', 'audio_loudness_mean', 'audio_snr',
    'audio_voice_activity_ratio', 'audio_f1_mean', 'audio_f2_mean', 'audio_f1_slope',
    'audio_latency_mean', 'audio_latency_max', 'audio_slow_response_ratio',
    'text_sentiment_compound', 'text_sentiment_neg', 'text_sentiment_pos',
    'text_first_person_ratio', 'text_negative_ratio', 'text_positive_ratio',
    'text_absolutist_ratio', 'text_cognitive_ratio', 'text_word_count',
    'text_lexical_diversity', 'text_flesch_reading_ease', 'text_avg_sentence_length',
    'text_talk_ratio', 'text_turn_count', 'text_engagement_slope',
    'text_emotion_sadness', 'text_emotion_anger', 'text_emotion_fear', 'text_emotion_joy',
    'text_disfluency_rate',
    'video_flow_mean', 'video_flow_std', 'video_motion_score', 'video_quality_score',
    'face_blink_rate', 'face_gaze_direct_ratio', 'face_gaze_averted_ratio',
    'face_head_yaw_mean', 'face_head_pitch_mean', 'face_au_mean',
    'face_micro_expression_rate', 'face_expression_variability',
    'video_discrete_blur_ratio', 'video_discrete_dark_ratio', 'face_detection_rate',
    'congruence_score', 'congruence_audio_text_match', 'congruence_masking_detected',
    'phq8_somatic_score', 'phq8_cognitive_score', 'phq8_anhedonia', 'phq8_fatigue',
    'phq8_anxiety', 'phq8_total_estimate',
    'clnf_features_found'
]

print(f"CHECKING {len(ALL_108_FEATURES)} FEATURES:")
print("-"*70)

present = []
missing = []
for feat in ALL_108_FEATURES:
    count = sum(1 for p in pids if feat in f[p] or feat in f[p].attrs)
    if count > 0:
        present.append(f"{feat}: {count}/{len(pids)}")
    else:
        missing.append(feat)

print(f"\nPRESENT ({len(present)}):")
for p in present:
    print(f"  [OK] {p}")

print(f"\nMISSING ({len(missing)}):")
for m in missing:
    print(f"  [X] {m}")

print()
print("EATD SAMPLES:")
for p in eatd[:3]:
    n = len(list(f[p].keys()))
    print(f"  {p}: {n} datasets {'(EMPTY!)' if n==0 else ''}")

f.close()
