# Complete 108-Step Specification for H5-OmniFusion Pipeline

> **Version**: 3.0 | **Source**: `H5_OMNIFUSION_PREPROCESSING_PROMPT.xml`  
> **Structure**: 40 Production + 59 Research + 9 Advanced Innovations = 108 Steps

---

## GROUP I: 40-STEP PRODUCTION PIPELINE (P1-P40)

### Audio Modality (P1-P11)

| # | Step ID | Name | Description |
|---|---------|------|-------------|
| 1 | P1 | Loading_Resampling | Raw audio loading with automatic resampling to 16 kHz for Wav2Vec2 |
| 2 | P2 | Stereo_to_Mono | Average channels to ensure consistent single-channel signal |
| 3 | P3 | Speaker_Diarization | Separate interviewer (Ellie) from participant using pyannote.audio |
| 4 | P4 | Peak_Normalization | Scale amplitude to [-1, 1] range |
| 5 | P5 | Loudness_Normalization | Adjust energy to -23 LUFS (EBU R128 standard) |
| 6 | P6 | Noise_Reduction | Spectral subtraction/gating to remove background noise |
| 7 | P7 | Voice_Activity_Detection | Isolate speech regions from silence using Silero VAD |
| 8 | P8 | Segmentation | Divide audio into 10-second sliding windows with 50% overlap |
| 9 | P9 | Wav2Vec2_Embeddings | Extract 768-dim contextual representations via Wav2Vec2-Large-XLSR-53 |
| 10 | P10 | eGeMAPSv02_Features | Extract 88 acoustic markers using OpenSMILE → project to 768-dim |
| 11 | P11 | Prosodic_Respiratory_Analysis | Extract speaking rates, pause ratios, and sigh detection |

#### P1-P11 Detailed Technical Specifications
*   **P1 Loading_Resampling**: Uses `torchaudio.load()` with backend dispatch. Automatically detects original sample rate and applies `Resample` transform with Kaiser window resampling if SR != 16000Hz. Handles FLAC, WAV, and MP3 formats.
*   **P3 Speaker_Diarization**: Implements `pyannote/speaker-diarization` pipeline. Applies strict thresholding to identify the "Participant" cluster. Segments attributed to "Interviewer" are masked or discarded to prevent model leakage.
*   **P6 Noise_Reduction**: utilizes `noisereduce` library with stationary noise assumption. Estimates noise profile from the first 0.5s of audio (assumed silence/ambient). Applies spectral gating with a sensitivity of 1.0 and frequency smoothing.
*   **P9 Wav2Vec2_Embeddings**: Loads `facebook/wav2vec2-large-xlsr-53`. Input waveform is normalized. Forward pass extracts the last hidden state. Output tensor shape `(Batch, Time, 1024)` is projected via linear layer to `(Batch, 768)` and mean-pooled over the time dimension for segment-level representation.
*   **P10 eGeMAPSv02_Features**: Runs `opensmile` standard config `eGeMAPSv02.conf`. Extracts 88 functionals (mean, std, percentiles) covering frequency, energy, spectral, and cepstral domains. Features are Z-score normalized before projection.


### Text Modality (P12-P20)

| # | Step ID | Name | Description |
|---|---------|------|-------------|
| 12 | P12 | Transcript_Cleaning | Remove timestamps and speaker identification tags |
| 13 | P13 | Annotation_Removal | Strip non-verbal cues like [laughter], [sigh], [pause] |
| 14 | P14 | Disfluency_Handling | Preserve fillers ("um", "uh") for diagnostic counts |
| 15 | P15 | Tokenization | Apply RoBERTa Byte-Pair Encoding (BPE), max 512 tokens |
| 16 | P16 | MentalRoBERTa_Embeddings | Generate 768-dim [CLS] token embeddings from domain-adapted model |
| 17 | P17 | Linguistic_Features | Count first-person pronouns, absolutist words, negative emotion terms |
| 18 | P18 | Complexity_Metrics | Calculate Type-Token Ratio and readability scores |
| 19 | P19 | Sentiment_Scoring | Extract valence/polarity using VADER or DistilRoBERTa |
| 20 | P20 | Conversation_Dynamics | Measure talk ratios and engagement change over time |

#### P12-P20 Detailed Technical Specifications
*   **P12 Transcript_Cleaning**: Regex-based pipeline to remove `scrubbed` tags, timestamps `\d+\.\d+`, and speaker labels `Participant:`. Handles inconsistencies in DAIC-WOZ transcript formatting.
*   **P16 MentalRoBERTa_Embeddings**: Utilizes `mental/mental-roberta-base`. Inputs are truncated to 512 tokens. The model outputs the last hidden state of the `[CLS]` token, which serves as the aggregate sequence representation. This embedding captures depression-specific semantic nuances missed by standard BERT.
*   **P17 Linguistic_Features**: Implements a dictionary-based counter for LIWC-like categories. Specifically tracks 'Absolutist' words (always, never, completely) which correlate with suicidality, and First-Person Singular pronouns (I, me, my) indicating self-focus.
*   **P19 Sentiment_Scoring**: Hybrid approach. Uses VADER for rule-based sentence-level polarity (handling negations and intensifiers) and DistilRoBERTa-base-finetuned-sst-2 for deep contextual sentiment. Scores are averaged to produce a robust valence metric.


### Video Modality (P21-P26)

| # | Step ID | Name | Description |
|---|---------|------|-------------|
| 21 | P21 | Frame_Extraction | Sample video uniformly at 5-8 FPS |
| 22 | P22 | Quality_Filtering | Filter frames based on Laplacian variance and brightness |
| 23 | P23 | ImageNet_Normalization | Standardize pixels using ImageNet mean/std |
| 24 | P24 | Resizing | Scale images to uniform 224x224 resolution |
| 25 | P25 | VideoMAE_Embeddings | Extract 768-dim spatiotemporal features via VideoMAE-base |
| 26 | P26 | Optical_Flow_Analysis | Measure pixel motion magnitudes between frames |

#### P21-P26 Detailed Technical Specifications
*   **P21 Frame_Extraction**: Uses `ffmpeg` or `opencv` to decode video streams. Enforces a constant frame rate sampling (default 8 FPS) to ensure temporal consistency for the VideoMAE transformer.
*   **P22 Quality_Filtering**: Computes the variance of the Laplacian of each frame. Frames with variance < 100 are flagged as blurry and discarded. Brightness histogram analysis rejects frames with mean intensity < 30 (too dark) or > 225 (overexposed).
*   **P25 VideoMAE_Embeddings**: Input is a tensor of shape `(Batch, 16, 3, 224, 224)` representing a 16-frame clip. The VideoMAE encoder (masked autoencoder pre-trained on Kinetics-400) processes the clip. We extract the mean pooled output of the encoder's last hidden layer to obtain a 768-dim vector representing the spatiotemporal context.


### Face Modality (P27-P34)

| # | Step ID | Name | Description |
|---|---------|------|-------------|
| 27 | P27 | Face_Detection | Locate faces using RetinaFace or MediaPipe (confidence >0.8) |
| 28 | P28 | Landmark_Alignment | Warp faces to canonical pose based on 5-point landmarks |
| 29 | P29 | Face_Cropping | Isolate face region with 20% margin expansion to 224x224 |
| 30 | P30 | Face_Tracking | Associate detections across frames using SORT/DeepSORT |
| 31 | P31 | POSTER_v2_Embeddings | Extract 768-dim facial expression/emotion embeddings |
| 32 | P32 | Action_Unit_Detection | Identify presence/intensity of 17+ Action Units (OpenFace 2.0) |
| 33 | P33 | Gaze_Head_Pose_Analysis | Track eye contact ratio, gaze aversion, yaw/pitch/roll |
| 34 | P34 | Micro_Expression_Timing | Analyze onset and duration of facial movements |

### Tabular & Cross-Modal (P35-P40)

| # | Step ID | Name | Description |
|---|---------|------|-------------|
| 35 | P35 | Missing_Value_Imputation | Fill gaps using Median/Mode strategies |
| 36 | P36 | Categorical_Encoding | Convert categories to one-hot/embeddings |
| 37 | P37 | Numerical_Normalization | Scale clinical inputs using StandardScaler (z-score) |
| 38 | P38 | TabPFN_Projection | Generate 768-dim tabular embeddings for fusion |
| 39 | P39 | Clinical_Engineering | Derive PHQ-8 sub-scores (Somatic vs. Cognitive clusters) |
| 40 | P40 | Quality_Confidence_Scoring | Calculate SNR and detection confidence for fusion weighting |

---

## GROUP II: 59-STEP EXHAUSTIVE RESEARCH PIPELINE (R1-R59)

### Audio Research (R1-R17)

| # | Step ID | Name | Description |
|---|---------|------|-------------|
| 1 | R1 | Audio_File_Loading | Read raw audio files from participant folders |
| 2 | R2 | Sample_Rate_Conversion | Convert sample rate to 16 kHz for Wav2Vec2 |
| 3 | R3 | Stereo_to_Mono | Average channels to create single-channel signal |
| 4 | R4 | Speaker_Diarization | Isolate participant speech from interviewer |
| 5 | R5 | Peak_Normalization | Scale amplitude to [-1, 1] range |
| 6 | R6 | LUFS_Normalization | Adjust loudness to -23 LUFS (EBU R128) |
| 7 | R7 | Spectral_Noise_Gating | Remove background noise via spectral subtraction |
| 8 | R8 | Voice_Activity_Detection | Detect and extract voiced speech regions |
| 9 | R9 | Temporal_Segmentation | Divide audio into overlapping windows or utterance-level segments |
| 10 | R10 | Wav2Vec2_Deep_Inference | Extract 768-dim contextual audio embeddings |
| 11 | R11 | eGeMAPSv02_Acoustic_Features | Extract 88 acoustic markers using OpenSMILE |
| 12 | R12 | Pitch_F0_Tracking | Extract fundamental frequency contour (f0_mean, f0_std, f0_range) |
| 13 | R13 | Jitter_Shimmer_Analysis | Measure micro-fluctuations in pitch and amplitude |
| 14 | R14 | Formant_Analysis | Extract vocal tract resonance frequencies (F1-F4) |
| 15 | R15 | Respiratory_Pattern_Detection | Detect breath groups and breathing patterns, sigh detection |
| 16 | R16 | Pause_Analysis | Analyze silence duration and frequency patterns |
| 17 | R17 | Speaking_Rate_Analysis | Calculate syllables per second and articulation rate |

### Text Research (R18-R31)

| # | Step ID | Name | Description |
|---|---------|------|-------------|
| 18 | R18 | Transcript_Loading | Read transcript from .tsv, .txt, or ASR output |
| 19 | R19 | Timestamp_Tag_Cleaning | Remove timestamps and speaker identification tags |
| 20 | R20 | Contraction_Handling | Expand or standardize English contractions |
| 21 | R21 | NonVerbal_Removal | Strip non-verbal cue annotations ([laughter], [sigh]) |
| 22 | R22 | Whitespace_Normalization | Trim excess whitespace and normalize spacing |
| 23 | R23 | Disfluency_Processing | Count fillers for diagnostics (um, uh, er, ah, like) |
| 24 | R24 | Transformer_Tokenization | Apply BPE tokenization for transformer models |
| 25 | R25 | Text_Embedding_Inference | Generate 768-dim [CLS] embeddings (MentalRoBERTa) |
| 26 | R26 | LIWC_Analysis | Count psycholinguistic categories (first-person, negative emotion) |
| 27 | R27 | Lexical_Diversity | Calculate type-token ratio, hapax legomena ratio |
| 28 | R28 | Readability_Scoring | Calculate Flesch-Kincaid grade, Gunning Fog index |
| 29 | R29 | Sentiment_Analysis | Extract emotional valence and polarity (VADER/DistilBERT) |
| 30 | R30 | Emotion_Labeling | Classify text into categorical emotions (anger, sadness, joy, etc.) |
| 31 | R31 | Turn_Taking_Dynamics | Analyze response latency, talk ratio, turn count |

### Video Research (R32-R38)

| # | Step ID | Name | Description |
|---|---------|------|-------------|
| 32 | R32 | Frame_Sampling | Extract frames at uniform rate (5-8 FPS, 16 frames for VideoMAE) |
| 33 | R33 | Blur_Filtering | Remove low-quality blurred frames (Laplacian variance <50) |
| 34 | R34 | Exposure_Filtering | Remove underexposed/overexposed frames (brightness 80-180) |
| 35 | R35 | ImageNet_Normalization | Standardize pixel values using ImageNet statistics |
| 36 | R36 | Resolution_Scaling | Resize frames to uniform 224x224 resolution |
| 37 | R37 | VideoMAE_Inference | Extract 768-dim spatiotemporal embeddings |
| 38 | R38 | Optical_Flow | Calculate motion magnitude between frames |

### Face Research (R39-R49)

| # | Step ID | Name | Description |
|---|---------|------|-------------|
| 39 | R39 | Face_Detection | Locate faces with RetinaFace/MediaPipe (confidence >0.8) |
| 40 | R40 | Landmark_Alignment | Warp face to canonical pose (5-point landmarks) |
| 41 | R41 | Face_Cropping | Extract face region with 20% margin to 224x224 |
| 42 | R42 | Face_Tracking | Associate face detections across frames (DeepSORT/SORT) |
| 43 | R43 | POSTER_v2_Embeddings | Extract 768-dim facial expression embeddings |
| 44 | R44 | AU_Binary_Detection | Detect presence of 17+ Action Units (OpenFace/py-feat) |
| 45 | R45 | AU_Intensity_Estimation | Estimate continuous AU intensity (0-5 scale) |
| 46 | R46 | Blink_Rate_Analysis | Track eye closure frequency and duration (EAR) |
| 47 | R47 | Gaze_Direction_Tracking | Track gaze vector, eye contact (direct/indirect/averted) |
| 48 | R48 | Head_Pose_Estimation | Estimate head orientation (yaw, pitch, roll) |
| 49 | R49 | Micro_Expression_Timing | Analyze onset/offset of expressions, emotional lag |

### Tabular Research (R50-R53)

| # | Step ID | Name | Description |
|---|---------|------|-------------|
| 50 | R50 | Missing_Value_Imputation | Handle missing data (median, mode, KNN imputation) |
| 51 | R51 | Categorical_Encoding | Convert categorical variables (one-hot, target encoding) |
| 52 | R52 | Numerical_Normalization | Scale numeric features (z-score, min-max) |
| 53 | R53 | TabPFN_Embedding | Generate 768-dim tabular embeddings for fusion |

### Cross-Modal & Augmentation (R54-R59)

| # | Step ID | Name | Description |
|---|---------|------|-------------|
| 54 | R54 | Temporal_Grid_Alignment | Synchronize all modalities to common time grid |
| 55 | R55 | Word_Level_Alignment | Map audio/video features to specific words (Montreal Forced Aligner) |
| 56 | R56 | SpecAugment | Audio data augmentation (freq masking, time masking) |
| 57 | R57 | Video_Augmentation | Video data augmentation (random crop, flip, color jitter) |
| 58 | R58 | Text_Augmentation | Text data augmentation (synonym replacement, back-translation) |
| 59 | R59 | Quality_Confidence_Scoring | Calculate quality metrics for fusion weighting |

---

## GROUP III: 9 ADVANCED INNOVATIONS (ADV1-ADV9)

### Biomarker Extractors (ADV1-ADV5)

| # | Step ID | Name | Description |
|---|---------|------|-------------|
| 1 | ADV1 | Response_Latency_Extraction | Measure precise ms gap between interviewer offset and participant onset |
| 2 | ADV2 | Kinematics_Posture_Analysis | Track body slumping trends and head movement velocity |
| 3 | ADV3 | Prosodic_Fingerprint | Generate 32-dim learned embedding of speech rhythm and pause distributions |
| 4 | ADV4 | Symptom_Specific_Clustering | Map features to PHQ-8 sub-scales (Anhedonia, Sleep, Fatigue, etc.) |
| 5 | ADV5 | Breath_Interval_Variability | Calculate std dev of intervals between breath groups |

### Advanced Fusion (ADV6-ADV9)

| # | Step ID | Name | Description |
|---|---------|------|-------------|
| 6 | ADV6 | Cross_Modal_Congruence_Scoring | Calculate alignment between modalities (text sentiment vs audio valence) |
| 7 | ADV7 | Temporal_Trajectory_Encoding | Model slope and curvature of features over session (fatigue progression) |
| 8 | ADV8 | Adaptive_Quality_Gated_Fusion | Dynamically weight modalities based on real-time quality metrics |
| 9 | ADV9 | Modality_Imputation | Hallucinate missing modality features using cross-modal mappings |

---

## Summary Statistics

| Category | Count | Type |
|----------|-------|------|
| Production Pipeline | 40 | Core Implementation |
| Research Pipeline | 59 | Deep Analysis |
| Advanced Innovations | 9 | Novel Features |
| **TOTAL** | **108** | **Complete Specification** |

---

## Output Embedding Requirements

All core modality embeddings must be **768-dimensional** for fusion compatibility:

| Embedding | Source | Required Dim |
|-----------|--------|--------------|
| `audio_embedding` | Wav2Vec2-Large-XLSR-53 | 768 |
| `audio_egemaps_embedding` | eGeMAPSv02 → Linear Projection | 768 |
| `text_embedding` | MentalRoBERTa / Chinese-BERT | 768 |
| `video_embedding` | VideoMAE-base / ViT-DINO | 768 |
| `face_embedding` | POSTER_v2 / ViT-base | 768 |
| `tabular_embedding` | TabPFN / MLP Projection | 768 |
| `fusion_embedding` | Multimodal Fusion Layer | 768 |

---

*Document generated: 2026-01-20*
