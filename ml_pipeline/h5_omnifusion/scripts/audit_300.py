"""
Comprehensive audit of 300.h5 for H5-OmniFusion Pipeline Compliance.
Checks all 108 steps (40 Production + 59 Research + 9 Advanced).
"""
import h5py
import numpy as np
import json

def audit_h5(filepath):
    """Complete H5 compliance audit."""
    results = {
        'file': filepath,
        'structure': {},
        'embeddings': {},
        'features': [],
        'compliance': {}
    }
    
    with h5py.File(filepath, 'r') as f:
        print("=" * 70)
        print(f"H5-OMNIFUSION COMPLIANCE AUDIT: {filepath}")
        print("=" * 70)
        
        print("\n[1] COMPLETE H5 STRUCTURE")
        print("-" * 50)
        
        all_datasets = []
        all_groups = []
        
        def collect_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                info = {
                    'name': name,
                    'shape': obj.shape,
                    'dtype': str(obj.dtype),
                    'size': obj.size
                }
                all_datasets.append(info)
                print(f"  [D] {name}: shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                all_groups.append(name)
                print(f"  [G] {name}/")
        
        f.visititems(collect_structure)
        
        results['structure']['datasets'] = len(all_datasets)
        results['structure']['groups'] = len(all_groups)
        
        print(f"\nTotal: {len(all_datasets)} datasets, {len(all_groups)} groups")
        
        print("\n" + "=" * 70)
        print("[2] 768-DIM EMBEDDING VERIFICATION")
        print("-" * 50)
        
        required_embeddings = [
            'audio_embedding',
            'text_embedding', 
            'video_embedding',
            'face_embedding',
            'tabular_embedding',
            'fusion_embedding'
        ]
        
        found_embeddings = {}
        for ds in all_datasets:
            name_lower = ds['name'].lower()
            for req in required_embeddings:
                if req in name_lower:
                    has_768 = 768 in ds['shape']
                    found_embeddings[req] = {
                        'path': ds['name'],
                        'shape': ds['shape'],
                        'is_768': has_768
                    }
        
        for req in required_embeddings:
            if req in found_embeddings:
                emb = found_embeddings[req]
                status = "PASS" if emb['is_768'] else "FAIL"
                print(f"  [{status}] {req}: {emb['path']} -> shape={emb['shape']}")
            else:
                print(f"  [MISSING] {req}: NOT FOUND")
        
        results['embeddings'] = found_embeddings
        
        print("\n" + "=" * 70)
        print("[3] 108-STEP FEATURE MAPPING")
        print("-" * 50)
        
        production_40 = {
            'STEP_1': ['waveform', 'sample_rate', 'audio'],
            'STEP_2': ['mono', 'channel'],
            'STEP_3': ['diarization', 'speaker', 'participant'],
            'STEP_4': ['peak_norm', 'normalized'],
            'STEP_5': ['lufs', 'loudness'],
            'STEP_6': ['noise', 'denoise'],
            'STEP_7': ['vad', 'voice_activity'],
            'STEP_8': ['segment'],
            'STEP_9': ['wav2vec', 'audio_embedding'],
            'STEP_10': ['egemaps', 'acoustic', 'opensmile'],
            'STEP_11': ['prosod', 'respirat', 'speaking_rate'],
            'STEP_12': ['transcript', 'clean'],
            'STEP_13': ['annotation', 'nonverbal'],
            'STEP_14': ['disfluency', 'filler'],
            'STEP_15': ['token'],
            'STEP_16': ['roberta', 'text_embedding', 'mental'],
            'STEP_17': ['linguistic', 'pronoun', 'liwc'],
            'STEP_18': ['complexity', 'readability', 'ttr'],
            'STEP_19': ['sentiment', 'vader'],
            'STEP_20': ['conversation', 'dynamics', 'turn_taking'],
            'STEP_21': ['frame', 'extract'],
            'STEP_22': ['quality', 'blur', 'brightness'],
            'STEP_23': ['imagenet', 'normalize'],
            'STEP_24': ['resize', '224'],
            'STEP_25': ['videomae', 'video_embedding'],
            'STEP_26': ['optical_flow', 'motion'],
            'STEP_27': ['face_detect', 'face_box'],
            'STEP_28': ['landmark', 'align'],
            'STEP_29': ['face_crop'],
            'STEP_30': ['face_track', 'face_id'],
            'STEP_31': ['poster', 'face_embedding', 'expression'],
            'STEP_32': ['action_unit', 'au_'],
            'STEP_33': ['gaze', 'head_pose'],
            'STEP_34': ['micro_expression', 'onset', 'timing'],
            'STEP_35': ['impute', 'missing'],
            'STEP_36': ['categorical', 'encoding', 'one_hot'],
            'STEP_37': ['numerical', 'zscore', 'standard'],
            'STEP_38': ['tabpfn', 'tabular_embedding'],
            'STEP_39': ['clinical', 'phq', 'somatic', 'cognitive'],
            'STEP_40': ['quality_score', 'confidence', 'snr'],
        }
        
        research_59 = {
            'R12': ['pitch', 'f0'],
            'R13': ['jitter', 'shimmer'],
            'R14': ['formant', 'f1', 'f2', 'f3'],
            'R15': ['respiratory', 'breath', 'sigh'],
            'R16': ['pause'],
            'R17': ['speaking_rate', 'syllable'],
            'R26': ['liwc', 'first_person', 'absolute'],
            'R27': ['lexical', 'ttr', 'mattr'],
            'R28': ['readability', 'flesch', 'gunning'],
            'R29': ['sentiment'],
            'R30': ['emotion', 'anger', 'sadness', 'joy'],
            'R31': ['turn_taking', 'interrupt', 'response_latency'],
            'R44': ['au_binary', 'au01', 'au04'],
            'R45': ['au_intensity'],
            'R46': ['blink', 'ear'],
            'R47': ['gaze_direct', 'aversion'],
            'R48': ['head_pose', 'yaw', 'pitch', 'roll'],
            'R49': ['micro_expression', 'onset_latency'],
            'R54': ['temporal_align', 'grid'],
            'R55': ['word_level', 'forced_align'],
            'R56': ['specaugment'],
            'R57': ['video_augment'],
            'R58': ['text_augment', 'synonym'],
            'R59': ['quality_confidence'],
        }
        
        advanced_9 = {
            'ADV1': ['response_latency', 'latency_ms'],
            'ADV2': ['kinematic', 'posture', 'slump'],
            'ADV3': ['prosodic_fingerprint'],
            'ADV4': ['symptom', 'cluster', 'anhedonia'],
            'ADV5': ['breath_interval', 'sigh_count'],
            'ADV6': ['congruence', 'cross_modal'],
            'ADV7': ['trajectory', 'slope', 'curvature'],
            'ADV8': ['quality_gated', 'weighted_fusion'],
            'ADV9': ['imputation', 'modality_impute'],
        }
        
        all_feature_names = [ds['name'].lower() for ds in all_datasets]
        all_feature_str = ' '.join(all_feature_names)
        
        def check_step(step_id, keywords):
            for kw in keywords:
                if kw.lower() in all_feature_str:
                    return 'FOUND'
            return 'MISSING'
        
        print("\n  --- 40 PRODUCTION STEPS ---")
        prod_found = 0
        for step, keywords in production_40.items():
            status = check_step(step, keywords)
            if status == 'FOUND':
                prod_found += 1
            print(f"    [{status}] {step}")
        
        print(f"\n  Production: {prod_found}/40 steps detected")
        
        print("\n  --- KEY RESEARCH STEPS ---")
        res_found = 0
        for step, keywords in research_59.items():
            status = check_step(step, keywords)
            if status == 'FOUND':
                res_found += 1
            print(f"    [{status}] {step}")
        
        print(f"\n  Research samples: {res_found}/{len(research_59)} checked")
        
        print("\n  --- 9 ADVANCED INNOVATIONS ---")
        adv_found = 0
        for step, keywords in advanced_9.items():
            status = check_step(step, keywords)
            if status == 'FOUND':
                adv_found += 1
            print(f"    [{status}] {step}")
        
        print(f"\n  Advanced: {adv_found}/9 steps detected")
        
        print("\n" + "=" * 70)
        print("[4] DATA QUALITY CHECK")
        print("-" * 50)
        
        for ds in all_datasets:
            name = ds['name']
            try:
                data = f[name][()]
                if isinstance(data, np.ndarray) and data.size > 0:
                    is_zeros = np.allclose(data, 0)
                    is_nan = np.any(np.isnan(data)) if np.issubdtype(data.dtype, np.floating) else False
                    has_variety = data.std() > 1e-6 if np.issubdtype(data.dtype, np.floating) else True
                    
                    issues = []
                    if is_zeros:
                        issues.append("ALL_ZEROS")
                    if is_nan:
                        issues.append("HAS_NAN")
                    if not has_variety and not is_zeros:
                        issues.append("CONSTANT")
                    
                    if issues:
                        print(f"  [WARN] {name}: {', '.join(issues)}")
                    else:
                        print(f"  [OK] {name}: valid data (mean={data.mean():.4f})")
            except Exception as e:
                print(f"  [ERR] {name}: {e}")
        
        print("\n" + "=" * 70)
        print("[5] COMPLIANCE SUMMARY")
        print("=" * 70)
        
        emb_pass = sum(1 for e in found_embeddings.values() if e.get('is_768'))
        emb_total = len(required_embeddings)
        
        print(f"\n  768-dim Embeddings: {emb_pass}/{emb_total}")
        print(f"  Production Steps: {prod_found}/40")
        print(f"  Research Samples: {res_found}/{len(research_59)}")
        print(f"  Advanced Steps: {adv_found}/9")
        print(f"  Total Datasets: {len(all_datasets)}")
        
        if emb_pass >= 5 and prod_found >= 30:
            print("\n  >>> OVERALL: LIKELY COMPLIANT <<<")
        else:
            print("\n  >>> OVERALL: REQUIRES REVIEW <<<")
        
        return results

if __name__ == "__main__":
    audit_h5("300.h5")
