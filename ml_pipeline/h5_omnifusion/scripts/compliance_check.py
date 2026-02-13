"""
H5 file compliance checker for XML specification
Updated to check all required features per H5_OMNIFUSION_PREPROCESSING_PROMPT.xml
"""
import h5py
import numpy as np

def check_compliance(h5_path='features_account4.h5'):
    f = h5py.File(h5_path, 'r')
    keys = list(f.keys())
    
    daic_woz = [k for k in keys if not k.startswith('t_')]
    eatd = [k for k in keys if k.startswith('t_')]
    
    print("=" * 70)
    print("H5 COMPLIANCE CHECK REPORT (vs XML Spec)")
    print("=" * 70)
    print()
    print(f"DATASET DISTRIBUTION: {len(daic_woz)} DAIC-WOZ, {len(eatd)} EATD-Corpus")
    print()
    
    all_pass = True
    
    print("=" * 70)
    print("CORE 768-DIM EMBEDDINGS (Spec Lines 1416-1420)")
    print("=" * 70)
    required_emb = ['audio_embedding', 'text_embedding', 'video_embedding', 
                    'face_embedding', 'tabular_embedding', 'fusion_embedding']
    for emb in required_emb:
        count = sum(1 for k in keys if emb in f[k])
        correct_dim = sum(1 for k in keys if emb in f[k] and f[k][emb].shape == (768,))
        status = "PASS" if count == len(keys) else "PARTIAL"
        if count < len(keys):
            all_pass = False
        print(f"  {emb}: {count}/{len(keys)} present, {correct_dim} correct dim -> {status}")
    
    print()
    print("=" * 70)
    print("L2 NORMALIZATION CHECK (Spec Line 1430)")
    print("=" * 70)
    sample = keys[0]
    for emb in ['audio_embedding', 'text_embedding', 'video_embedding', 'face_embedding']:
        if emb in f[sample]:
            data = f[sample][emb][:]
            norm = np.linalg.norm(data)
            is_normalized = 0.99 < norm < 1.01
            status = "PASS" if is_normalized else "FAIL"
            if not is_normalized:
                all_pass = False
            print(f"  {emb}: L2 norm={norm:.4f} -> {status}")
    
    print()
    print("=" * 70)
    print("REQUIRED FIELDS CHECK")
    print("=" * 70)
    
    required_fields = {
        'quality_scores': 'Line 1421 - Per-modality confidence',
        'phq8_score': 'Line 1422 - Ground truth label',
        'prosodic_features': 'Step P11 - Speaking rate, pause ratio',
        'linguistic_features': 'Step P17 - First person ratio, etc',
        'sentiment_scores': 'Step P19 - VADER/sentiment analysis',
        'gaze_features': 'Step P33 - Head pose, gaze aversion',
        'optical_flow': 'Step P26 - Motion magnitude',
        'supplementary_features': 'Line 1423 - All scalar features group',
    }
    
    for field, desc in required_fields.items():
        if field == 'supplementary_features':
            count = sum(1 for k in keys if field in f[k] and isinstance(f[k][field], h5py.Group))
        else:
            count = sum(1 for k in keys if field in f[k])
        status = "PASS" if count == len(keys) else f"MISSING ({count}/{len(keys)})"
        if count < len(keys):
            all_pass = False
        print(f"  {field}: {status} - {desc}")
    
    print()
    print("=" * 70)
    print("ACTION UNIT FEATURES (Step P32, R44-R45)")
    print("=" * 70)
    au_count = sum(1 for k in keys if 'au_mean' in f[k])
    print(f"  au_mean (17-dim): {au_count}/{len(keys)} -> {'PASS' if au_count >= len(daic_woz) else 'PARTIAL'}")
    au_emb = sum(1 for k in keys if 'au_embedding' in f[k])
    print(f"  au_embedding (768-dim): {au_emb}/{len(keys)} -> {'PASS' if au_emb >= len(daic_woz) else 'PARTIAL'}")
    
    print()
    print("=" * 70)
    print("EATD-CORPUS SAMPLES CHECK (No Video)")
    print("=" * 70)
    if eatd:
        eatd_sample = eatd[0]
        has_data = len(list(f[eatd_sample].keys())) > 0
        status = "PASS (has data)" if has_data else "FAIL (empty group)"
        if not has_data:
            all_pass = False
        print(f"  Sample {eatd_sample}: {status}")
        if has_data:
            print(f"    Fields: {list(f[eatd_sample].keys())[:5]}...")
    else:
        print("  No EATD samples found")
    
    print()
    print("=" * 70)
    print("DATA QUALITY CHECK")
    print("=" * 70)
    nan_count = 0
    for k in keys:
        for ds_name in f[k].keys():
            if isinstance(f[k][ds_name], h5py.Dataset):
                if f[k][ds_name].dtype in [np.float32, np.float16, np.float64]:
                    data = f[k][ds_name][:]
                    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                        nan_count += 1
    status = "PASS" if nan_count == 0 else f"FAIL ({nan_count} datasets with NaN/Inf)"
    if nan_count > 0:
        all_pass = False
    print(f"  NaN/Inf check: {status}")
    
    print()
    print("=" * 70)
    print("OVERALL COMPLIANCE VERDICT")
    print("=" * 70)
    if all_pass:
        print("  ✅ FULLY COMPLIANT with XML specification")
    else:
        print("  ⚠️ PARTIAL COMPLIANCE - Some fields missing or incorrect")
        print("  Note: Re-run the pipeline to generate compliant output")
    
    f.close()
    return all_pass

if __name__ == '__main__':
    check_compliance()
