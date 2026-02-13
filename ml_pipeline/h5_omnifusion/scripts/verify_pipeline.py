"""
H5-OmniFusion Pipeline Verification Script
Verifies all modular pipeline imports and ADV feature wiring.
Run this script before Colab execution to catch import errors early.
"""

import sys
import os

PIPELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "ml_pipeline", "h5_omnifusion", "preprocessing_and_feature_extraction")
sys.path.insert(0, PIPELINE_DIR)

print("=" * 60)
print("H5-OMNIFUSION PIPELINE VERIFICATION")
print("=" * 60)
print(f"Pipeline directory: {PIPELINE_DIR}")
print()

def check_import(module_name, class_names, description):
    """Check if classes can be imported from a module."""
    try:
        module = __import__(module_name)
        found = []
        missing = []
        for cls in class_names:
            if hasattr(module, cls):
                found.append(cls)
            else:
                missing.append(cls)
        if missing:
            print(f"⚠️  {description}: Found {len(found)}/{len(class_names)} classes")
            print(f"   Missing: {', '.join(missing)}")
            return False
        print(f"✅ {description}: All {len(found)} classes found")
        return True
    except ImportError as e:
        print(f"❌ {description}: Import failed - {e}")
        return False
    except Exception as e:
        print(f"❌ {description}: Unexpected error - {e}")
        return False

all_passed = True

print("\n--- Enhancement Modules ---")

all_passed &= check_import(
    "audio_enhancements",
    ["ProsodicFingerprint", "LoudnessNormalizer", "PauseAnalyzer", "SighDetector", 
     "BreathIntervalAnalyzer", "AudioQualityChecker"],
    "audio_enhancements.py"
)

all_passed &= check_import(
    "video_face_enhancements", 
    ["KinematicsPostureAnalyzer", "VideoQualityFilter", "OpticalFlowAnalyzer",
     "SimpleFaceTracker", "GazeCategorizer", "MicroExpressionAnalyzer"],
    "video_face_enhancements.py"
)

all_passed &= check_import(
    "text_enhancements",
    ["TranscriptCleaner", "PsycholinguisticExtractor", "ComplexityAnalyzer",
     "MultilingualSentimentAnalyzer", "ConversationDynamicsAnalyzer", "LanguageDetector"],
    "text_enhancements.py"
)

all_passed &= check_import(
    "fusion_enhancements",
    ["QualityGatedFusion", "ModalityImputer", "ClinicalClusterer", "WordLevelAligner",
     "CrossModalCongruenceScorer", "TemporalTrajectoryEncoder", "ResponseLatencyExtractor"],
    "fusion_enhancements.py"
)

print("\n--- Modular Pipeline Files ---")

all_passed &= check_import(
    "pipeline_audio",
    ["AudioPreprocessor", "TranscriptDiarizer", "EGeMAPSExtractor", "PitchAnalyzer"],
    "pipeline_audio.py"
)

all_passed &= check_import(
    "pipeline_text",
    ["TextPreprocessor", "TranscriptCleaner", "PsycholinguisticExtractor"],
    "pipeline_text.py"
)

all_passed &= check_import(
    "pipeline_video_face",
    ["VideoPreprocessor", "FacePreprocessor", "FrameExtractor", "FaceDetector"],
    "pipeline_video_face.py"
)

all_passed &= check_import(
    "pipeline_fusion_main",
    ["H5OmniFusionPipeline", "QualityGatedFusion", "ModalityImputer", "ZipExtractor"],
    "pipeline_fusion_main.py"
)

print("\n--- ADV Feature Import Flags ---")

try:
    from pipeline_audio import ADV3_OK
    print(f"   ADV3 (ProsodicFingerprint) available: {ADV3_OK}")
except:
    print("   ADV3_OK flag not found (fallback mode)")

try:
    from pipeline_video_face import ADV2_OK
    print(f"   ADV2 (KinematicsPostureAnalyzer) available: {ADV2_OK}")
except:
    print("   ADV2_OK flag not found (fallback mode)")

try:
    from pipeline_fusion_main import ALIGNER_OK
    print(f"   WordLevelAligner available: {ALIGNER_OK}")
except:
    print("   ALIGNER_OK flag not found (fallback mode)")

try:
    from pipeline_text import TEXT_ENHANCEMENTS_OK
    print(f"   Text Enhancements available: {TEXT_ENHANCEMENTS_OK}")
except:
    print("   TEXT_ENHANCEMENTS_OK flag not found (fallback mode)")

print("\n" + "=" * 60)
if all_passed:
    print("✅ VERIFICATION COMPLETE: ALL IMPORTS SUCCESSFUL")
    print("Pipeline is ready for Colab execution.")
else:
    print("⚠️  VERIFICATION COMPLETE: SOME IMPORTS FAILED")
    print("Pipeline will use fallback implementations for missing components.")
print("=" * 60)
