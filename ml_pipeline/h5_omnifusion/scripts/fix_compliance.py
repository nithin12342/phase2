import h5py
import numpy as np
import os
import sys
import argparse
from unittest.mock import MagicMock

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSING_DIR = os.path.join(CURRENT_DIR, "preprocessing_and_feature_extraction")
if PREPROCESSING_DIR not in sys.path:
    sys.path.append(PREPROCESSING_DIR)

sys.modules["librosa"] = MagicMock()
sys.modules["librosa.feature"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["parselmouth"] = MagicMock()
sys.modules["mediapipe"] = MagicMock()

try:
    from research_layer_extensions import AdvancedFeatures
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import AdvancedFeatures. Check paths. {e}")
    AdvancedFeatures = None

def fix_h5_file(h5_path):
    if not os.path.exists(h5_path):
        print(f"Error: File not found at {h5_path}")
        return

    print(f"Opening {h5_path} for patching...")
    
    with h5py.File(h5_path, 'r+') as f: # r+ is read/write
        participants = [key for key in f.keys()]
        print(f"Found participants: {participants}")
        
        if not participants:
            print("No groups found in H5 file.")
            return

        if AdvancedFeatures:
            try:
                adv_feats = AdvancedFeatures()
                print("Loaded AdvancedFeatures module.")
            except Exception as e:
                print(f"Failed to initialize AdvancedFeatures: {e}")
                adv_feats = None
        else:
            adv_feats = None

        for pid in participants:
            print(f"\nProcessing Participant: {pid}")
            grp = f[pid]
            
            if 'advanced' not in grp:
                adv_grp = grp.create_group('advanced')
            else:
                adv_grp = grp['advanced']

            if 'turn_taking' in adv_grp or 'conversation_dynamics' in adv_grp:
                print("  - P20 (Turn Taking) already exists.")
            else:
                supp = grp.get('supplementary_features')
                turn_taking_vec = np.zeros(4, dtype=np.float32)
                
                if supp:
                    t_ratio = float(supp.attrs.get('text_talk_ratio', 0.0))
                    t_count = float(supp.attrs.get('text_turn_count', 0.0))
                    w_per_t = float(supp.attrs.get('text_word_count', 0.0)) / max(1.0, t_count) 
                    e_slope = float(supp.attrs.get('text_engagement_slope', 0.0))
                    
                    turn_taking_vec = np.array([t_ratio, t_count, w_per_t, e_slope], dtype=np.float32)
                
                adv_grp.create_dataset('turn_taking', data=turn_taking_vec)
                print(f"  ✅ Fixed P20 (Turn Taking): {turn_taking_vec}")

            if 'fusion_embedding' in grp:
                fusion_emb = grp['fusion_embedding'][:]
                
                if adv_feats:
                    if 'trajectory' not in adv_grp:
                        try:
                            traj = adv_feats.trajectory_encoder.encode_trajectory(fusion_emb)
                            adv_grp.create_dataset('trajectory', data=traj)
                            print("  ✅ Fixed ADV7 (Temporal Trajectory)")
                        except Exception as e:
                            print(f"  ❌ Failed to generate ADV7: {e}")
                    else:
                        print("  - ADV7 (Trajectory) already exists.")

                    if 'symptom_clusters' not in adv_grp:
                        try:
                            clusters = adv_feats.cluster_projector.project(fusion_emb)
                            adv_grp.create_dataset('symptom_clusters', data=clusters)
                            print("  ✅ Fixed ADV4 (Symptom Clusters)")
                        except Exception as e:
                            print(f"  ❌ Failed to generate ADV4: {e}")
                    else:
                        print("  - ADV4 (Symptom Clusters) already exists.")
                else:
                    print("  ❌ AdvancedFeatures module not available, cannot fix ADV4/ADV7.")
            else:
                print("  ❌ 'fusion_embedding' missing. Cannot derive ADV4/ADV7.")

        print("\nPatching complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix H5 Compliance Issues")
    parser.add_argument("file", nargs='?', default="301.h5", help="Path to H5 file")
    args = parser.parse_args()
    
    target_path = args.file
    if not os.path.exists(target_path):
        candidates = [
            target_path,
             os.path.join(CURRENT_DIR, "output file folder", target_path)
        ]
        for c in candidates:
            if os.path.exists(c):
                target_path = c
                break
    
    fix_h5_file(target_path)
