"""
108-Feature Compliance List Generator for H5 files
Generates a complete checklist of all features per XML specification
"""
import h5py
import numpy as np

FEATURE_SCHEMA = {
    'CORE_EMBEDDINGS_768D': [
        'audio_embedding', 'text_embedding', 'video_embedding',
        'face_embedding', 'tabular_embedding', 'fusion_embedding',
        'audio', 'text', 'video', 'face', 'tabular',
        'au_embedding', 'audio_egemaps_embedding'
    ],
    'XML_COMPLIANCE_FIELDS': [
        'quality_scores', 'phq8_score', 'prosodic_features',
        'linguistic_features', 'sentiment_scores', 'gaze_features',
        'optical_flow', 'supplementary_features'
    ],
    'AU_FEATURES': ['au_mean', 'au_std'],
    'AUDIO_SCALARS': [
        'audio_pause_ratio', 'audio_pause_mean', 'audio_speaking_rate', 'audio_phonation_ratio',
        'audio_f0_mean', 'audio_f0_std', 'audio_f0_range', 'audio_jitter', 'audio_shimmer',
        'audio_breath_count', 'audio_sigh_count', 'audio_loudness_mean', 'audio_snr',
        'audio_voice_activity_ratio', 'audio_f1_mean', 'audio_f2_mean', 'audio_f1_slope',
        'audio_latency_mean', 'audio_latency_max', 'audio_slow_response_ratio'
    ],
    'TEXT_SCALARS': [
        'text_sentiment_compound', 'text_sentiment_neg', 'text_sentiment_pos',
        'text_first_person_ratio', 'text_negative_ratio', 'text_positive_ratio',
        'text_absolutist_ratio', 'text_cognitive_ratio', 'text_word_count',
        'text_lexical_diversity', 'text_flesch_reading_ease', 'text_avg_sentence_length',
        'text_talk_ratio', 'text_turn_count', 'text_engagement_slope',
        'text_emotion_sadness', 'text_emotion_anger', 'text_emotion_fear', 'text_emotion_joy',
        'text_disfluency_rate'
    ],
    'VIDEO_FACE_SCALARS': [
        'video_flow_mean', 'video_flow_std', 'video_motion_score', 'video_quality_score',
        'face_blink_rate', 'face_gaze_direct_ratio', 'face_gaze_averted_ratio',
        'face_head_yaw_mean', 'face_head_pitch_mean', 'face_au_mean',
        'face_micro_expression_rate', 'face_expression_variability',
        'video_discrete_blur_ratio', 'video_discrete_dark_ratio', 'face_detection_rate'
    ],
    'FUSION_SCALARS': [
        'congruence_score', 'congruence_audio_text_match', 'congruence_masking_detected',
        'phq8_somatic_score', 'phq8_cognitive_score', 'phq8_anhedonia', 'phq8_fatigue',
        'phq8_anxiety', 'phq8_total_estimate'
    ]
}

def generate_compliance_list(h5_path='features_account4.h5'):
    f = h5py.File(h5_path, 'r')
    pids = list(f.keys())
    daic = [p for p in pids if not p.startswith('t_')]
    eatd = [p for p in pids if p.startswith('t_')]

    print('=' * 80)
    print('108-FEATURE COMPLIANCE LIST FOR', h5_path)
    print('=' * 80)
    print(f'Total: {len(pids)} participants ({len(daic)} DAIC-WOZ, {len(eatd)} EATD-Corpus)')
    print()

    total_features = 0
    present_features = 0
    feature_status = {}

    for cat, features in FEATURE_SCHEMA.items():
        for feat in features:
            total_features += 1
            count = sum(1 for p in pids if feat in f[p] or (hasattr(f[p], 'attrs') and feat in f[p].attrs))
            pct = count / len(pids) * 100 if pids else 0
            feature_status[feat] = {'count': count, 'total': len(pids), 'pct': pct, 'cat': cat}
            if count > 0:
                present_features += 1

    idx = 1
    for cat, features in FEATURE_SCHEMA.items():
        print(f'\n=== {cat} ({len(features)} features) ===')
        for feat in features:
            s = feature_status[feat]
            if s['count'] == len(pids):
                icon = 'OK  '
            elif s['count'] > 0:
                icon = 'PART'
            else:
                icon = 'MISS'
            print(f'  {idx:3d}. [{icon}] {feat:40s}  {s["count"]:2d}/{s["total"]:2d}  ({s["pct"]:5.1f}%)')
            idx += 1

    print()
    print('=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'Total features in schema:   {total_features}')
    print(f'Features with any data:     {present_features}')
    print(f'Features completely missing: {total_features - present_features}')
    
    missing = [f for f, s in feature_status.items() if s['count'] == 0]
    if missing:
        print(f'\nMISSING FEATURES ({len(missing)}):')
        for m in missing:
            print(f'  - {m}')

    print()
    print('EATD-CORPUS SAMPLES STATUS:')
    for p in eatd[:5]:
        fields = list(f[p].keys())
        status = 'EMPTY' if len(fields) == 0 else f'{len(fields)} fields'
        print(f'  {p}: {status}')

    print()
    print('L2 NORMALIZATION CHECK (DAIC-WOZ samples):')
    for p in daic[:3]:
        for emb in ['audio_embedding', 'text_embedding', 'video_embedding', 'face_embedding']:
            if emb in f[p]:
                norm = np.linalg.norm(f[p][emb][:])
                status = 'OK' if 0.99 < norm < 1.01 else f'NOT NORMALIZED (norm={norm:.2f})'
                print(f'  {p}/{emb}: {status}')

    f.close()

if __name__ == '__main__':
    generate_compliance_list()
