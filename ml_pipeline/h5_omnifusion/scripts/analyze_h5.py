"""
H5 Compliance Analyzer
Checks if H5 files comply with H5_OMNIFUSION_PREPROCESSING_PROMPT.xml specification.
"""

import h5py
import numpy as np
import json
import sys

def analyze_h5_file(filepath):
    """Analyze a single H5 file and return its structure."""
    result = {
        'filepath': filepath,
        'participants': [],
        'all_features': set(),
        'embedding_dims': {},
        'issues': []
    }
    
    try:
        with h5py.File(filepath, 'r') as f:
            for pid in f.keys():
                grp = f[pid]
                participant_info = {
                    'pid': pid,
                    'dataset': grp.attrs.get('dataset', 'unknown'),
                    'features': {},
                    'embedding_status': {}
                }
                
                for key in grp.keys():
                    if isinstance(grp[key], h5py.Dataset):
                        shape = grp[key].shape
                        dtype = str(grp[key].dtype)
                        participant_info['features'][key] = {'shape': shape, 'dtype': dtype}
                        result['all_features'].add(key)
                        
                        if 'embedding' in key.lower():
                            result['embedding_dims'][key] = shape
                            if len(shape) == 1 and shape[0] == 768:
                                participant_info['embedding_status'][key] = 'OK (768-dim)'
                            elif len(shape) == 1:
                                participant_info['embedding_status'][key] = f'WRONG DIM ({shape[0]})'
                                result['issues'].append(f"{pid}/{key}: Expected 768-dim, got {shape[0]}")
                            else:
                                participant_info['embedding_status'][key] = f'Shape: {shape}'
                    else:
                        participant_info['features'][key] = {'type': 'Group'}
                
                result['participants'].append(participant_info)
                
    except Exception as e:
        result['error'] = str(e)
    
    result['all_features'] = list(result['all_features'])
    return result

REQUIRED_EMBEDDINGS = [
    'audio_embedding',
    'text_embedding', 
    'video_embedding',
    'face_embedding',
    'tabular_embedding',
    'fusion_embedding'
]

REQUIRED_SCALARS = [
    'phq8_score',
    'gaze_features',
    'optical_flow'
]

RESEARCH_FEATURES = [
    'PHQ8_Anhedonia', 'PHQ8_Fatigue', 'PHQ8_Cognitive',  # ADV4
    'overall_congruence', 'masking_detected',  # ADV6
    'audio_snr', 'text_word_count',  # Quality
]

def check_compliance(analysis):
    """Check compliance against specification."""
    compliance = {
        'total_participants': len(analysis['participants']),
        'embeddings': {},
        'scalars': {},
        'research': {},
        'overall_score': 0,
        'missing': [],
        'present': []
    }
    
    all_features = set(analysis['all_features'])
    
    for emb in REQUIRED_EMBEDDINGS:
        if emb in all_features:
            compliance['embeddings'][emb] = 'PRESENT'
            compliance['present'].append(emb)
            if emb in analysis['embedding_dims']:
                shape = analysis['embedding_dims'][emb]
                if len(shape) == 1 and shape[0] == 768:
                    compliance['embeddings'][emb] = 'OK (768-dim)'
                else:
                    compliance['embeddings'][emb] = f'WRONG ({shape})'
        else:
            compliance['embeddings'][emb] = 'MISSING'
            compliance['missing'].append(emb)
    
    for scalar in REQUIRED_SCALARS:
        if scalar in all_features:
            compliance['scalars'][scalar] = 'PRESENT'
            compliance['present'].append(scalar)
        else:
            compliance['scalars'][scalar] = 'MISSING'
            compliance['missing'].append(scalar)
    
    for feat in RESEARCH_FEATURES:
        if feat in all_features:
            compliance['research'][feat] = 'PRESENT'
            compliance['present'].append(feat)
        else:
            compliance['research'][feat] = 'MISSING'
            compliance['missing'].append(feat)
    
    total_checks = len(REQUIRED_EMBEDDINGS) + len(REQUIRED_SCALARS) + len(RESEARCH_FEATURES)
    passed = len(compliance['present'])
    compliance['overall_score'] = round(passed / total_checks * 100, 1)
    
    return compliance

def main():
    files = [
        r"c:\Users\thela\OneDrive\Desktop\phase 2\300.h5",
        r"c:\Users\thela\OneDrive\Desktop\phase 2\301.h5",
        r"c:\Users\thela\OneDrive\Desktop\phase 2\features_ALL_MERGED.h5"
    ]
    
    print("=" * 70)
    print("H5 COMPLIANCE ANALYSIS REPORT")
    print("=" * 70)
    
    for filepath in files:
        print(f"\n{'─' * 70}")
        print(f"📁 FILE: {filepath.split(chr(92))[-1]}")
        print(f"{'─' * 70}")
        
        analysis = analyze_h5_file(filepath)
        
        if 'error' in analysis:
            print(f"❌ ERROR: {analysis['error']}")
            continue
        
        print(f"Participants: {len(analysis['participants'])}")
        print(f"Total Features: {len(analysis['all_features'])}")
        
        if analysis['participants']:
            p = analysis['participants'][0]
            print(f"\n📊 Sample Participant: {p['pid']} (Dataset: {p['dataset']})")
            print(f"   Features: {len(p['features'])}")
            
            print("\n   🔹 Embedding Dimensions:")
            for emb, status in p['embedding_status'].items():
                print(f"      {emb}: {status}")
        
        compliance = check_compliance(analysis)
        print(f"\n✅ COMPLIANCE SCORE: {compliance['overall_score']}%")
        
        print("\n   Required Embeddings:")
        for k, v in compliance['embeddings'].items():
            icon = '✓' if 'MISSING' not in v else '✗'
            print(f"      {icon} {k}: {v}")
        
        print("\n   Required Scalars:")
        for k, v in compliance['scalars'].items():
            icon = '✓' if v == 'PRESENT' else '✗'
            print(f"      {icon} {k}: {v}")
        
        print("\n   Research/Advanced Features:")
        for k, v in compliance['research'].items():
            icon = '✓' if v == 'PRESENT' else '✗'
            print(f"      {icon} {k}: {v}")
        
        if compliance['missing']:
            print(f"\n   ⚠️ Missing ({len(compliance['missing'])}): {', '.join(compliance['missing'][:10])}")
        
        if analysis['issues']:
            print(f"\n   ❌ Issues ({len(analysis['issues'])}):")
            for issue in analysis['issues'][:5]:
                print(f"      - {issue}")
        
        print(f"\n   📋 All Features in File ({len(analysis['all_features'])}):")
        for feat in sorted(analysis['all_features'])[:30]:
            print(f"      - {feat}")
        if len(analysis['all_features']) > 30:
            print(f"      ... and {len(analysis['all_features']) - 30} more")

if __name__ == "__main__":
    main()
