# Deep Manual Feature Extraction Catalog — H5-OmniFusion (Depression / PHQ-8)

Target datasets: **DAIC-WOZ**, **Extended-DAIC**, **EATD-Corpus**. Context: a person
sitting in a face-to-face clinical interview (Ellie the interviewer asks, the
participant answers), captured as audio, transcript, video, face, and session
metadata. Every feature below is **hand-designed and interpretable** — a
clinically-motivated signal, not a learned embedding.

## Design principles

- **Interpretable over dense.** Each feature has a name, an explicit formula, a
  plausible range, and a stated depression rationale, so a model's decision can
  be explained back to a clinician.
- **Beyond the current 64, and collision-checked against the 108-step spec.**
  The repo's authoritative `docs/ml_pipeline/108_step_specification.md`
  (P1–P40 production, R1–R59 research, ADV1–ADV9) already defines many of these
  signals. This catalog enumerates **254 candidate features** but tags each so
  there is **no silent collision**: **[K]** already emitted by the pipeline,
  **[N]** genuinely new, and **[S→step]** *already produced by a spec step —
  read it from that step's output by name, do NOT re-derive it* (re-deriving
  creates a duplicate/colliding column). The full collision map is §10.
- **Robust to missing modalities.** EATD has no video/face; E-DAIC audio can be
  silent. Each family declares a fallback so a missing channel yields NaN +
  a per-feature `present` flag, never a fabricated zero (the bug that made the
  last run's features useless).
- **Within-interview normalization.** Acoustic/speed scales differ per mic and
  per person; each raw feature is paired with a speaker-normalized (z-score or
  robust-median) variant where it matters.

Feature ID convention: `<MOD>-<nnn>` — AUD (audio), TXT (text), VID (video),
FAC (face), TAB (tabular/session/context), XM (cross-modal congruence),
Q (quality/reliability).

---

## 1. AUDIO — prosody, voice quality, spectrum, respiratory, timing

Depression signature: **monopitch, reduced loudness, slow psychomotor speech,
longer/paused-fragmented utterances, breathy/tense voice quality, downward
spectral tilt.**

### 1.1 Pitch / F0 (contour dynamics beyond mean+std)

| ID | Feature | Computation | Rationale / range |
|----|---------|-------------|-------------------|
| AUD-001 | **[K]** f0_mean | median of voiced F0 (librosa/pyworld) | flat affect → lower variability, mean stable |
| AUD-002 | **[K]** f0_std | std of voiced F0 | core monotony marker |
| AUD-003 | **[K]** f0_range | P95−P05 of F0 | reduced expressive range |
| AUD-004 | **[N]** f0_iqr | P75−P25 | outlier-robust range |
| AUD-005 | **[N]** f0_cv | f0_std / f0_mean | scale-free monotony |
| AUD-006 | **[N]** f0_delta_mean | mean abs frame-to-frame semitone change | articulatory prosodic liveness |
| AUD-007 | **[N]** f0_delta_max | max pitch excursion per utterance | peak expressivity |
| AUD-008 | **[N]** f0_pitch_reset | mean semitone drop at phrase end | flat declination → blunted affect |
| AUD-009 | **[N]** f0_question_rise | rise slope on "...?"-type turns | reduced intonation interest |
| AUD-010 | **[N]** f0_trend_slope | linear slope of F0 vs time over session | psychomotor slowing / fatigue buildup |
| AUD-011 | **[N]** f0_autocorr_period | dominant periodicity of F0 contour | pitch stereotypy / repetition |
| AUD-012 | **[N]** f0_entropy | entropy of F0 histogram (20 bins, semitones) | spread of pitch usage |
| AUD-013 | **[N]** f0_skewness | skew of voiced F0 | downward vs upward bias |
| AUD-014 | **[N]** f0_kurtosis | kurtosis of voiced F0 | monotone spikes |
| AUD-015 | **[N]** f0_pct_below_person_median | fraction of voiced frames < speaker median | time spent "low" |
| AUD-016 | **[N]** f0_contour_slope_std | std of local regression slope of F0 | erratic vs smooth prosody |
| AUD-017 | **[N]** f0_syllable_peak_count | pitch peaks per phonation run | intonation events per utterance |
| AUD-018 | **[S→R11]** semitone_mean | mean in semitone scale — *in eGeMAPS `F0semitoneFrom27.5Hz`* | perceptually linear pitch |

### 1.2 Loudness / energy / dynamics

| ID | Feature | Computation | Rationale / range |
|----|---------|-------------|-------------------|
| AUD-020 | **[S→R11]** loudness_mean | RMS (dB) mean — *in eGeMAPS `loudness`* | hypophonia |
| AUD-021 | **[S→R11]** loudness_std | RMS dB std — *in eGeMAPS `loudness` functional* | flat dynamics |
| AUD-022 | **[S→R11]** loudness_range | P95−P05 dB — *eGeMAPS `loudness` max−min* | expressive amplitude |
| AUD-023 | **[N]** loudness_cv | std/mean linear RMS | scale-free |
| AUD-024 | **[N]** loudness_slope_session | dB trend over interview | fading voice / fatigue |
| AUD-025 | **[N]** loudness_delta_mean | mean abs frame energy change | dynamics liveness |
| AUD-026 | **[N]** loudness_peak_freq | # of stressed-syllable energy peaks / min | reduced stress marking |
| AUD-027 | **[N]** whisper_ratio | frames in low-freq-dominant/noise-like band | breathy/whispery voice |

### 1.3 Voice quality — jitter, shimmer, harmonic noise, spectral

| ID | Feature | Computation | Rationale / range |
|----|---------|-------------|-------------------|
| AUD-030 | **[K]** jitter | local period variation (Praat-style) | vocal tension / psychomotor |
| AUD-031 | **[N]** jitter_local_pct | jitter ppq5 variant | robust tension index |
| AUD-032 | **[N]** jitter_rap | 3-point relative average | |
| AUD-033 | **[K]** shimmer | cycle-to-cycle amplitude variation | breathiness/hoarseness |
| AUD-034 | **[N]** shimmer_apq3 | 3-pt amplitude perturbation | |
| AUD-035 | **[S→R11]** hnr | harmonic-to-noise ratio — *in eGeMAPS `hnrc`* | hoarseness/weak voice |
| AUD-036 | **[S→R11]** cpp | cepstral peak prominence — *in eGeMAPS `cppes`* | overall voice regularity |
| AUD-037 | **[S→R11]** spectral_centroid_mean | mean centroid — *in eGeMAPS `spectralCentroid`* | brightness / tension |
| AUD-038 | **[S→R11]** spectral_tilt | slope — *in eGeMAPS `slopeVov`/`slopeVoHi`* | downward tilt → depressed |
| AUD-039 | **[N]** spectral_rolloff_85 | 85% energy freq — *not in eGeMAPS* | dull/flat voice |
| AUD-040 | **[N]** spectral_bandwidth_mean | spread of energy across bands — *novel* | narrowband → monopitch |
| AUD-041 | **[S→R11]** spectral_flux_mean | frame-to-frame spectral change — *in eGeMAPS `spectralFlux`* | reduced expressivity |
| AUD-042 | **[S→R11]** zcr_mean | zero-crossing rate — *in eGeMAPS `zeroCrossingRate`* | breathiness proxy |
| AUD-043 | **[S→R11]** mfcc1..mfcc12_mean | mean MFCCs — *in eGeMAPS `mfcc1`–`mfcc12`* | broad timbre |
| AUD-044 | **[N]** mfcc1_delta_mean | MFCC1 trajectory (delta) — *deltas deprecated in opensmile≥2.0; recompute if wanted* | prosodic timbre movement |
| AUD-045 | **[S→R11]** formant_f1_mean | F1 — *in eGeMAPS `formantLogc-1`* | openness / jaw tension |
| AUD-046 | **[S→R11]** formant_f2_mean | F2 — *in eGeMAPS `formantLogc-2`* | |
| AUD-046b | **[S→R11]** formant_f3_mean, formant_f4_mean | F3/F4 — *in eGeMAPS `formantLogc-3/4`* | (was missing from old catalog; eGeMAPS already has them) |
| AUD-047 | **[N]** f1_f2_space_area | vowel-space area (F1×F2 convex hull) | articulatory undershoot → "mumbled" |
| AUD-048 | **[K]** f1_slope | F1 trend | |
| AUD-049 | **[N]** vocal_efficiency | HNR / energy | effortful vs efficient phonation |
| AUD-050 | **[N]** roughness | AM band energy 20–110 Hz | voice roughness |

### 1.4 Timing — pauses, rate, fluency, silences

| ID | Feature | Computation | Rationale / range |
|----|---------|-------------|-------------------|
| AUD-052 | **[K]** pause_ratio | silence speech-ratio | psychomotor hesitation |
| AUD-053 | **[K]** pause_mean / pause_count | avg # of pauses ≥ threshold | cognitive load |
| AUD-054 | **[N]** pause_median | median pause dur | |
| AUD-055 | **[N]** pause_std | pause duration variability | irregular hesitation |
| AUD-056 | **[N]** pause_max | **[K]** longest pause | stall at hard questions |
| AUD-057 | **[N]** pause_cv | std/mean | |
| AUD-058 | **[N]** long_pause_count | pauses > 2 s | marked stalling |
| AUD-059 | **[N]** verylong_pause_count | pauses > 5 s | avoidance/dissociation |
| AUD-060 | **[N]** pause_log_slope | reg(log pause_dur) on rank | Zipf-like pause distribution |
| AUD-061 | **[K]** speaking_rate | syllables or words / phonation min | retardation |
| AUD-062 | **[N]** speaking_rate_std | variance of instantaneous rate | |
| AUD-063 | **[N]** utterance_len_mean | mean run of speech between pauses | |
| AUD-064 | **[S→R17]** articulation_rate | syllables / (total−pauses) — *R17 names "articulation rate"; also eGeMAPS `articulationRate`* | true speed w/o pauses |
| AUD-065 | **[N]** phonation_ratio | **[K]** voiced/total | |
| AUD-066 | **[N]** voice_activity_ratio | **[K]** VAD fraction | |
| AUD-067 | **[N]** onset_delay_mean | gap from interviewer-end to participant-start | latency (also AUD-081) |
| AUD-068 | **[N]** turn_total_silence | sum of unvoiced across participant turns | |
| AUD-069 | **[N]** speech_fill_ratio | total phonation / interview duration | how much they talk at all |
| AUD-070 | **[N]** dysfluency_acoustic | filled-pause events / min | "um/uh" bursts |
| AUD-071 | **[N]** burst_speed | syllables/sec during fast runs vs overall | pressure vs retardation |

### 1.5 Respiratory / non-verbal vocalization

| ID | Feature | Computation | Rationale / range |
|----|---------|-------------|-------------------|
| AUD-073 | **[K]** breath_count | audible breath events | sighing respiration |
| AUD-074 | **[K]** breath_interval_mean/std/cv | spacing of breaths | |
| AUD-075 | **[N]** sigh_duration_ratio | sigh total / phonation | |
| AUD-076 | **[K]** sigh_count, sigh_rate_per_min | **[N]** sigh_total_duration | |
| AUD-077 | **[N]** swallow_count | throat-clear/swallow events | |
| AUD-078 | **[N]** laugh_count | voiced laughter bursts (also flat-vs-anhedonia) | |

### 1.6 Response latency (audio-timed)

| ID | Feature | Computation | Rationale / range |
|----|---------|-------------|-------------------|
| AUD-080 | **[K]** latency_mean | gap after each Ellie question | cognitive slowing |
| AUD-081 | **[K]** latency_max / latency_std | | variable processing |
| AUD-082 | **[N]** latency_median | | |
| AUD-083 | **[K]** slow_response_ratio | fraction of turns with latency > p75 | |
| AUD-084 | **[N]** latency_trend | slope of latency over turn index | fatigue / withdrawal |

---

## 2. TEXT — lexical, semantic, pragmatic, discourse

Depression signature: **more first-person & negative-emotion words, fewer
positive words, absolutist/dichotomous thinking, richer death/sadness/self-blame
semantics, reduced coherence, concrete rumination, low diversity of positive
topics.**

### 2.1 Sentiment / emotion (multi-lexicon, not one score)

| ID | Feature | Computation | Rationale |
|----|---------|-------------|-----------|
| TXT-001 | **[K]** sentiment_compound/neg/pos | VADER over participant turns | negative bias |
| TXT-002 | **[N]** sentiment_neg_minus_pos | net polarity | |
| TXT-003 | **[N]** sentiment_var | variance across turns | unstable affect |
| TXT-004 | **[N]** sentiment_trajectory | slope of turn-level compound over session | worsening/flat |
| TXT-005 | **[K]** emotion_sadness/anger/fear/joy | NRC / LIWC emotion sets | sad≠low-joy (anhedonia) |
| TXT-006 | **[N]** emotion_neg_emotion_total | anger+sadness+fear combined | |
| TXT-007 | **[N]** emotion_joy_minus_sadness | positive–negative gap | |
| TXT-008 | **[N]** dominant_emotion_entropy | entropy across 8 NRC emotions | undifferentiated affect |
| TXT-009 | **[N]** anxiety_words_ratio | LIWC anxiety set | |
| TXT-010 | **[N]** sadness_words_ratio | | |
| TXT-011 | **[N]** death_suicide_markers | rate of death/die/kill/hopeless/self-harm | clinical risk signal |

### 2.2 Pronouns / self-reference

| ID | Feature | Computation | Rationale |
|----|---------|-------------|-----------|
| TXT-013 | **[K]** first_person_ratio | I/me/my/mine per word | self-focus (Tausczik & Pennebaker) |
| TXT-014 | **[N]** first_person_plural_ratio | we/us/our | social disconnection |
| TXT-015 | **[N]** second_person_ratio | you/your | other-orientation |
| TXT-016 | **[N]** third_person_ratio | he/she/they | |
| TXT-017 | **[N]** i_you_ratio | self vs other focus | |
| TXT-018 | **[N]** impersonal_pronoun_ratio | it/what/there | depersonalization |

### 2.3 Cognitive style / distortions

| ID | Feature | Computation | Rationale |
|----|---------|-------------|-----------|
| TXT-020 | **[K]** absolutist_ratio | always/never/everything/nothing | dichotomous thinking |
| TXT-021 | **[K]** negative_ratio / positive_ratio | LIWC neg/pos | |
| TXT-022 | **[K]** cognitive_ratio | cognitive-ops words | |
| TXT-023 | **[N]** cognitive_mechanism | insight/reason/consider | reduced reflective processing |
| TXT-024 | **[N]** tentatives_ratio | maybe/sort of/guess | low confidence |
| TXT-025 | **[N]** exclusions_ratio | but/except/without | |
| TXT-026 | **[N]** negations_ratio | no/not/never | |
| TXT-027 | **[N]** rumination_markers | repetitive past-self references, wallowing terms | |
| TXT-028 | **[N]** self_blame_ratio | fault/guilty/ashamed/blame | |
| TXT-029 | **[N]** temporal_orientation_ratio | past vs present vs future words | future-negativity |

### 2.4 Lexical richness / complexity

| ID | Feature | Computation | Rationale |
|----|---------|-------------|-----------|
| TXT-031 | **[K]** word_count | | |
| TXT-032 | **[K]** lexical_diversity | type-token ratio (and MTLD where possible) | reduced verbosity |
| TXT-033 | **[N]** unique_words / rare_words_ratio | | |
| TXT-034 | **[K]** avg_word_length, avg_sentence_length | | |
| TXT-035 | **[K]** automated_readability | Flesch-Kincaid grade | simplification |
| TXT-036 | **[N]** flesch_reading_ease trend | slope over turns | |
| TXT-037 | **[N]** lexical_hard_words_ratio | beyond 5000-freq list | |
| TXT-038 | **[N]** noun_verb_ratio | POS-based | |
| TXT-039 | **[N]** adverb_adjective_ratio | reduced modification (sparse speech) | |
| TXT-040 | **[N]** content_word_ratio | nouns+verbs+adj vs function | |

### 2.5 Discourse / coherence / topic

| ID | Feature | Computation | Rationale |
|----|---------|-------------|-----------|
| TXT-042 | **[N]** topic_shift_rate | cosine drop between consecutive turn embeddings | loose associates |
| TXT-043 | **[N]** semantic_coherence | mean similarity of each turn to session centroid | tangentiality |
| TXT-044 | **[N]** repetition_ngram_ratio | repeated 3-grams / total | ruminative loops |
| TXT-045 | **[N]** topical_concentration | entropy over LDA/embedding clusters | narrow interests (anhedonia) |
| TXT-046 | **[N]** self_story_length | words per self-referential unit | verbosity of rumination |
| TXT-047 | **[N]** social_topic_ratio | family/friends/social words | social withdrawal |
| TXT-048 | **[N]** work_activity_ratio | job/school/activity | |
| TXT-049 | **[N]** sleep_words_ratio | **[K]** sleep words | somatic sleep signal |
| TXT-050 | **[N]** appetite_weight_words | eat/weight/appetite | somatic |
| TXT-051 | **[N]** energy_words_ratio | tired/energy/exhausted | fatigue |
| TXT-052 | **[N]** concentration_markers | focus/forget/decide | |
| TXT-053 | **[N]** hopelessness_ratio | hopeless/helpless/no point/worthless | |
| TXT-054 | **[N]** filler_count_ratio | um/uh/like/you know | disfluency |

### 2.6 Conversation / turn (text side)

| ID | Feature | Computation | Rationale |
|----|---------|-------------|-----------|
| TXT-056 | **[K]** talk_ratio | participant words / total | withdrawal |
| TXT-057 | **[K]** turn_count, words_per_turn | | |
| TXT-058 | **[K]** turn_length_mean/std/min/max | | |
| TXT-059 | **[K]** turn_length_trend | slope of turn length | shortening replies |
| TXT-060 | **[N]** min_turn_len_ratio | ultra-short (1–2 word) replies | |
| TXT-061 | **[N]** monosyllabic_answer_rate | bare yes/no | |
| TXT-062 | **[K]** engagement_slope | | |

---

## 3. VIDEO — motion, posture, scene, head dynamics

Depression signature: **reduced head/body motion (retardation) OR agitation,
slumped posture, stillness, low frame entropy.**

| ID | Feature | Computation | Rationale |
|----|---------|-------------|-----------|
| VID-001 | **[K]** flow_mean | mean optical-flow magnitude | psychomotor |
| VID-002 | **[K]** flow_std, flow_max | | |
| VID-003 | **[K]** motion_score | active-frame fraction | |
| VID-004 | **[N]** motion_energy | **[K]** per-frame pixel change mean | |
| VID-005 | **[N]** stillness_ratio | frames below motion threshold | |
| VID-006 | **[N]** motion_entropy | entropy of motion-magnitude bins | |
| VID-007 | **[N]** head_pose_yaw/pitch/roll_mean | **[K]** mean orientation | downcast gaze/posture |
| VID-008 | **[N]** head_yaw_std, head_pitch_std | variability | |
| VID-009 | **[N]** head_nod_rate | pitch oscillations / min | agreement/motor |
| VID-010 | **[N]** head_shake_rate | yaw oscillations / min | |
| VID-011 | **[N]** head_movement_velocity | **[K]** mean angular speed | |
| VID-012 | **[N]** head_forward_offset | **[K]** tilt toward/down camera | |
| VID-013 | **[N]** head_stillness_ratio | % time head motion < threshold | |
| VID-014 | **[N]** body_motion_bands | motion in upper vs lower frame region | |
| VID-015 | **[N]** face_region_motion | motion localized to face box | |
| VID-016 | **[N]** posture_drift | head-center translation start vs end | slumping over time |
| VID-017 | **[N]** blink_motion_coupling | correlation of head motion with blinks | |
| VID-018 | **[N]** scene_brightness_mean/std | gray-level stats | |
| VID-019 | **[K]** frame_quality_ratio, blur_score, brightness | | |
| VID-020 | **[N]** discrete_blur_ratio, dark_ratio | **[K]** | |
| VID-021 | **[N]** overexposed_ratio | | |
| VID-022 | **[N]** motion_sync_with_speech | cross-corr(head motion, energy envelope) | gesturing |
| VID-023 | **[N]** activity_onset_count | bursts of motion (fidgets) | agitation |
| VID-024 | **[N]** periodic_motion_peak | spectral peak of motion trace | rocking/tremor |

---

## 4. FACE — action units, expression, gaze, blink, geometry

Depression signature: **reduced smile (AU12), increased inner-brow draw (AU1/AU4),
reduced AU6, flattened expression, increased blink OR slowed, gaze aversion,
asymmetric/blunted response, longer emotional-decay time.**

### 4.1 Action units (per AU + composites)

> **Collision note:** the individual per-AU intensities/presences below
> (FAC-002–012) are **exactly** what spec steps **R44** (AU binary detection)
> and **R45** (AU intensity 0–5) already emit from OpenFace. They are tagged
> **[S→R45]** — read them from the AU table, do not re-derive. Only the
> *derived* composites/timings below them are new.

| ID | Feature | Computation | Rationale |
|----|---------|-------------|-----------|
| FAC-001 | **[K]** au_intensity (per AU vector) | **[S→R45]** OpenFace AU01..AU25 intensities | |
| FAC-002 | **[S→R45]** AU1_mean | inner brow raise intensity | sadness |
| FAC-003 | **[S→R45]** AU2_mean | outer brow raise | |
| FAC-004 | **[S→R45]** AU4_mean | brow lowerer | tension/anger |
| FAC-005 | **[S→R45]** AU6_mean | cheek raise (Duchenne) | genuine smile |
| FAC-006 | **[S→R45]** AU12_mean | lip corner pull (smile) | reduced in depression |
| FAC-007 | **[S→R45]** AU10_mean | upper lip raise | disgust |
| FAC-008 | **[S→R45]** AU15_mean | lip corner depress | sadness |
| FAC-009 | **[S→R45]** AU23_mean | lip tightener | tension |
| FAC-010 | **[S→R45]** AU24_mean | lip pressor | |
| FAC-011 | **[S→R45]** AU25/AU26/AU27 | jaw open / lip part / jaw drop | |
| FAC-012 | **[S→R45]** AU43_mean | eye close | |
| FAC-013 | **[K]** au_sadness_composite | f(AU1,AU4,AU15) | |
| FAC-014 | **[K]** au_happiness_composite | f(AU6,AU12) | |
| FAC-015 | **[K]** au_depression_indicator | curated AU1+AU4+AU15−AU12 | |
| FAC-016 | **[N]** Duchenne_ratio | AU6 present when AU12 present | genuine vs fake smiles |
| FAC-017 | **[N]** au_variability_mean | per-AU temporal std | flattened affect |
| FAC-018 | **[N]** au_active_ratio | AU>0.05 frame fraction | facial inactivity |
| FAC-019 | **[N]** au_range_mean | P95−P05 per AU | |
| FAC-020 | **[N]** au_symmetry | AU_left vs AU_right diffs | |
| FAC-021 | **[S→R49]** au_smile_timing | smile onset→peak — *R49 micro-expression onset/offset* | |
| FAC-022 | **[S→R49]** au_smile_decay | smile offset/"emotional lag" — *R49 explicitly says emotional lag* | emotional inertia |
| FAC-023 | **[N]** au_entropy | entropy over AU-presence vector | |

### 4.2 Expression / emotion (face-derived)

| ID | Feature | Computation | Rationale |
|----|---------|-------------|-----------|
| FAC-025 | **[K]** expression_variability | std of face emotion vector | |
| FAC-026 | emotion_fear_face/sad/angry/happy/neutral | per-emotion face means | |
| FAC-027 | neutral_face_ratio | frames labeled "neutral" dominant | flat affect |
| FAC-028 | **[K]** micro_expression_rate | brief (<300 ms) AU onsets | |
| FAC-029 | expression_change_rate | transitions between dominant emotions/min | |
| FAC-030 | expression_entropy | entropy of emotion label histogram | |

### 4.3 Gaze

| ID | Feature | Computation | Rationale |
|----|---------|-------------|-----------|
| FAC-032 | **[K]** gaze_direct_ratio | fraction facing camera | |
| FAC-033 | **[K]** gaze_averted_ratio | | |
| FAC-034 | **[K]** gaze_x/y_mean, gaze_avg_x/y | mean eye position (normalized) | downward gaze |
| FAC-035 | **[K]** gaze_mean_angle, gaze_std_angle | | |
| FAC-036 | **[N]** gaze_down_ratio | looking below horizontal | |
| FAC-037 | **[N]** gaze_saccade_rate | eye-move events / min | |
| FAC-038 | **[N]** eye_contact_stability | run-lengths of camera-facing | |
| FAC-039 | **[N]** gaze_aversion_onset | increase in aversion right after questions | |

### 4.4 Blink

| ID | Feature | Computation | Rationale |
|----|---------|-------------|-----------|
| FAC-041 | **[K]** blink_rate / blink_rate_per_min | blinks / min | |
| FAC-042 | **[K]** blink_count, avg_blink_duration_ms | | |
| FAC-043 | **[N]** blink_std | variability of interval | |
| FAC-044 | **[N]** long_blink_ratio | blinks > 300 ms | fatigue/drowsiness |
| FAC-045 | **[N]** blink_rate_trend | slope over session | |

### 4.5 Geometry / appearance / detection

| ID | Feature | Computation | Rationale |
|----|---------|-------------|-----------|
| FAC-047 | **[K]** face_detection_rate, conf_avg | | |
| FAC-048 | head_width_ratio / face_size | landmark face box vs frame | posture / proximity |
| FAC-049 | eye_aperture_mean | eyelid openness | psychomotor/parking |
| FAC-050 | mouth_openness_mean | landmark distance | |
| FAC-051 | brow_corner_drop | inner vs outer brow y | sad brows |
| FAC-052 | face_symmetry_score | landmark left-right | |
| FAC-053 | skin_tone_variance | lighting/health proxy (use cautiously) | |
| FAC-054 | face_present_ratio | frames with a detected face | |
| FAC-055 | gaze_shift_to_head_coupling | correlation(gaze, head yaw) | eye vs head leading |

---

## 5. TABULAR / SESSION / CONTEXT — metadata & derived

Depression signature at the interview level: **long latency to answer, short
session, high turn silence, and known PHQ-derived somatic/cognitive splits.**

| ID | Feature | Computation | Rationale |
|----|---------|-------------|-----------|
| TAB-001 | interview_duration | last−first timestamp | |
| TAB-002 | total_participant_time | sum of participant speech | |
| TAB-003 | interviewer_participant_ratio | Ellie vs user total | |
| TAB-004 | num_questions_answered | turns matched to Ellie prompts | |
| TAB-005 | unanswered_question_count | questions with no participant audio | avoidance |
| TAB-006 | avg_answer_length_words | | |
| TAB-007 | avg_answer_length_sec | | |
| TAB-008 | **[K]** phq8_somatic_score | somatic item sum (PHQ items 3,4 +…) | |
| TAB-009 | **[K]** phq8_cognitive_score | (PHQ items 2,6,7,8) | |
| TAB-010 | **[K]** phq8_anhedonia | item 1 | |
| TAB-011 | **[K]** phq8_fatigue | item 4 | |
| TAB-012 | **[K]** phq8_anxiety | item 5 | |
| TAB-013 | **[K]** phq8_total_estimate | sum | |
| TAB-014 | session_start_hour | time of interview | diurnal mood |
| TAB-015 | **[N]** audio_sr_detected | detected sample rate | |
| TAB-016 | **[N]** gender (from audio/text if provided) | | |
| TAB-017 | **[N]** age_estimate_face | from face landmarks | |

> Note: TAB-008..013 are **label-derived** and leak the target. Keep them
> OUT of the model input; use only for analysis/quarantine. Flagged in §7.

---

## 6. CROSS-MODAL CONGRUENCE — the richest manual signal

Depression-specific interactions: **flat voice over sad words (blunted), smiling
while describing hopelessness ("masked"), speech-motion decoupling.**

| ID | Feature | Computation | Rationale |
|----|---------|-------------|-----------|
| XM-001 | **[K]** congruence_audio_text_match | corr(voice emotion, text sentiment) | incongruence = masked affect |
| XM-002 | **[K]** congruence_masking_detected | positive smile + negative voice/text flag | |
| XM-003 | incongruence_smile_sadness | AU12 high while text sadness high | |
| XM-004 | flat_voice_sad_words | low f0_std + high neg-emotion | |
| XM-005 | **[K]** audio_face_match | corr(energy envelope, motion) | gesturing coupling |
| XM-006 | text_face_match | text emotion vs face emotion agreement | |
| XM-007 | latency_vs_answer_length | corr(turn latency, answer length) | |
| XM-008 | gaze_vs_negative_topic | aversion increase on negative-content turns | |
| XM-009 | pitch_drop_at_negative | ΔF0 around neg words | |
| XM-010 | energy_drop_on_self_ref | Δloudness on first-person words | |
| XM-011 | pause_on_hard_questions | longer pauses on symptom questions | |
| XM-012 | smile_rate_per_min | AU12 onsets / min | |
| XM-013 | response_verbosity_trend | slope of answer length | withdrawal |
| XM-014 | prosody_entropy | joint entropy pitch×loudness×tempo | expressivity |
| XM-015 | crossmodal_sync | **[K]** avg audio-visual temporal sync | |
| XM-016 | affect_variance | variance of fused emotion trace | instability |
| XM-017 | neg_in_pos_ratio | negative lexicon inside positive-sentiment frames | |

---

## 7. SIGNAL QUALITY / RELIABILITY — per-feature trust

The last run produced silent zeros/NaNs. **Every feature in this catalog is
emitted with a paired reliability gate** so downstream can mask, not trust:

| ID | Feature | Computation |
|----|---------|-------------|
| Q-001 | modality_present | bool: enough non-silent audio / nonempty text / valid frames |
| Q-002 | snr_db **[K]** | signal-to-noise |
| Q-003 | clipping_rate **[K]** | samples at ±full-scale |
| Q-004 | audio_dur_valid | voiced seconds ≥ floor |
| Q-005 | face_det_coverage | fraction frames with valid face |
| Q-006 | frame_quality_ratio **[K]** | good frames / total |
| Q-007 | transcript_align_ratio | words aligned to audio / expected |
| Q-008 | feature_nan_flag | per-vector: any NaN |
| Q-009 | feature_constant_flag | per-vector: near-zero across all participants |
| Q-010 | modality_imputed | True if this modality was filled from others |

---

## 8. Normalization / packaging rules (apply to every feature above)

1. **Emit NaN, never zero**, when a modality is absent; store `Q-001`.
2. **Pair raw + speaker-normalized** for all F0/loudness/rate features
   (z-score within participant, robust median/MAD across).
3. **Fixed column order + schema version** in the H5; validate on write
   (this is what the notebook Cell-14 quarantine should enforce).
4. **Per-modality sub-vectors** grouped so the fusion/MoE gate can read the
   `Q-0xx` flags and drop dead modalities instead of mixing zeros.
5. **Reproducible**: any stochastic step (sub-sampling, augmentation) seeded
   (SEED=42) — the adapters that were random this run must be seeded or trained.

---

## 9. Coverage summary (candidates before de-duplication)

| Modality | Candidate rows |
|----------|----------------|
| Audio (AUD) | 79 |
| Text (TXT) | 57 |
| Video (VID) | 24 |
| Face (FAC) | 51 |
| Tabular/session (TAB) | 17 |
| Cross-modal (XM) | 17 |
| Quality/reliability (Q) | 10 |
| **Total candidate rows** | **255** |

Tag tally across all rows: **[K]** 68 (already shipped), **[S→step]** 29
(duplicate a 108-step output — source, don't recompute), **[N]** 118 (explicitly
new), 40 untagged. §10 folds the untagged + [K] + [S] together into the real
collision picture, because a *collision* is defined against the 108-step spec,
not against the 64 shipped scalars.

**Bottom line: ~99 of the 255 rows map onto an existing 108-step output and must
be read from that step, never re-derived. ~156 rows have no spec-step parent —
that is the genuine net-new contribution.**

---

## 10. Collision reconciliation with the 108-step spec

Reference: `docs/ml_pipeline/108_step_specification.md` (P1–P40, R1–R59,
ADV1–ADV9). **Rule:** if a spec step already produces a signal, the catalog must
reference that step's output by name and must **not** define a second column for
the same quantity — a duplicate column is the "collision feature" failure that
let the last run mix zeros/redundancies into fusion.

### 10.1 Steps that already cover catalog rows (must be SOURCED, not recomputed)

| 108-step | Already emits | Colliding catalog rows | Action |
|----------|---------------|------------------------|--------|
| **P10 / R11 eGeMAPSv02 (88 fn)** | loudness, F0 (Hz+semitone) mean/std/range/quantiles, jitter, shimmer, HNR, CPP, spectral centroid/slope/flux/zcr, MFCC1–12, formants F1–F4 + bandwidth, speechRate, articulationRate, pauseTimeRatio, voiced/unvoiced counts | AUD-001…005, 018, 020…022, 030…048, 052/053/056, 061, 064…066 | **[S→R11]** read named functionals; do not recompute with librosa/praat |
| **R12 Pitch_F0_Tracking** | f0_mean/std/range | AUD-001, AUD-002, AUD-003 | **[S→R12]** |
| **R13 Jitter_Shimmer** | jitter, shimmer | AUD-030, AUD-033 (note: code computes these **twice** — opensmile *and* a Praat `JitterShimmerAnalyzer`; pick one) | **[S→R13]** dedup the Praat path |
| **R14 Formant_Analysis F1–F4** | F1..F4 freq + bandwidth | AUD-045, AUD-046, AUD-046b | **[S→R14]** |
| **R15/P11 Respiratory+sigh** | breath groups, sigh, speaking rate, pause ratio | AUD-052, 061, 073, 075/076 | **[S→R15]** |
| **R16 Pause_Analysis** | pause duration/frequency | AUD-052…060 | **[S→R16]** |
| **R17 Speaking_Rate** | syllables/sec + articulation rate | AUD-061, AUD-064 | **[S→R17]** |
| **ADV1 Response_Latency** | ms gap interviewer→participant | AUD-080…084, TXT-058 | **[S→ADV1]** |
| **ADV5 Breath_Interval_Variability** | std of breath intervals | AUD-074 | **[S→ADV5]** |
| **ADV3 Prosodic_Fingerprint (32-d)** | learned rhythm/pause embedding | AUD-011 (autocorr), contour shape | overlaps — keep only non-learned scalars |
| **ADV7 Temporal_Trajectory** | slope + curvature of features over session | AUD-010, AUD-024, TXT-004, TXT-036, TXT-059, AUD-084 | **[S→ADV7]** one generic trend operator, not per-family |
| **P17/R26 LIWC** | first-person, neg/pos emotion, absolutist, cognitive | TXT-005…013, 020…029 | **[S→R26]** |
| **P18/R27 Lexical_Diversity** | TTR, hapax, word/avg length | TXT-032…034 | **[S→R27]** |
| **R28 Readability** | Flesch-Kincaid, Gunning Fog | TXT-035, TXT-036 | **[S→R28]** |
| **P19/R29 Sentiment** | VADER/DistilBERT compound/pos/neg | TXT-001…003 | **[S→R29]** |
| **R30 Emotion_Labeling** | anger/sadness/fear/joy | TXT-005…011 | **[S→R30]** |
| **R31/P20 Turn_Taking** | talk ratio, turn count, latency, engagement slope | TXT-056…062 | **[S→R31]** |
| **P26/R38 Optical_Flow** | flow mean/std/magnitude | VID-001…003, VID-004 | **[S→R38]** |
| **P22/R33/R34 Quality filter** | blur, brightness, exposure ratio | VID-019…021 | **[S→R33/34]** |
| **ADV2 Kinematics_Posture** | head velocity + slumping trend | VID-011, VID-012, VID-016 | **[S→ADV2]** |
| **P31/R43 POSTER_v2** | face emotion embedding | FAC-025…030 | overlaps learned face; keep only label-derived scalars |
| **R44/R45 AU detection+intensity** | 17+ AU presence + 0–5 intensity | FAC-001…015 | **[S→R45]** (already inline-tagged) |
| **R46 Blink** | blink rate + duration (EAR) | FAC-041…042 | **[S→R46]** |
| **R47 Gaze** | direct/averted ratio, gaze vector | FAC-032…035 | **[S→R47]** |
| **R48 Head_Pose** | yaw/pitch/roll | FAC (head), VID-007…008 | **[S→R48]** |
| **P34/R49 Micro_Expression_Timing** | onset/offset, emotional lag | FAC-021, 022, 028 | **[S→R49]** (already inline-tagged) |
| **P39/ADV4 Symptom clustering** | PHQ-8 somatic/cognitive/anhedonia/fatigue/anxiety | TAB-008…013 | **[S→ADV4]** — and label-leak: see §5 note |
| **ADV6 Cross-Modal Congruence** | text-sentiment vs audio-valence alignment | XM-001, 002, 006, 015 | **[S→ADV6]** |
| **R55 Word_Level_Alignment** | word-aligned audio/face features | XM-009, XM-010, TXT word_* | **[S→R55]** |
| **P40/R59/ADV8 Quality scoring** | SNR, detection confidence, gated fusion weights | Q-002, 003, 006, 010 | **[S→R59]** |

### 10.2 The genuinely NEW, non-colliding additions (no 108-step parent)

These are the catalog's real contribution — signals the 108-step spec does **not**
define. Roughly **156 rows**, concentrated here:

- **Audio contour *shape* (not just moments):** f0 delta/rate, pitch reset,
  question-rise, syllable-peak count, spectral rolloff/bandwidth, whisper ratio,
  roughness (AM modulation), vocal-efficiency ratio, pause log-slope, long/verylong
  pause counts, onset-delay, acoustic dysfluency bursts, burst-speed.
- **Text discourse *dynamics*:** turn-to-turn topic shift, semantic coherence to
  centroid, n-gram repetition (ruminative loops), topical concentration, and the
  PHQ-item symptom lexicons (hopelessness, energy/fatigue, concentration, appetite,
  social-withdrawal) beyond generic LIWC.
- **Video behavioral motion:** motion entropy, stillness ratio, activity-onset
  (fidgets), periodic-motion peak (rocking/tremor), head nod/shake *rates*,
  blink-motion coupling, posture drift over session.
- **Face derived-AU & geometry:** Duchenne ratio, AU variability/range/entropy,
  AU symmetry, neutral-face ratio, expression change-rate & entropy, gaze
  down-ratio/saccade-rate/contact-stability/aversion-onset, long-blink ratio &
  blink trend, facial geometry (aperture, mouth openness, brow-drop, symmetry).
- **Cross-modal masking interactions:** incongruence_smile_sadness,
  flat_voice_sad_words, gaze_aversion on negative topics, prosody entropy
  (joint pitch×loud×tempo), neg-in-positve ratio, affect variance.
- **Reliability gates:** modality_present, audio_dur_valid, face_det_coverage,
  transcript_align_ratio, feature_nan_flag, feature_constant_flag — the mechanism
  that would have caught the last run's silent zeros/NaNs.

### 10.3 De-duplication rules to apply before implementing

1. **One source per quantity.** Any row in §10.1 must be populated from its named
   spec-step output. If a step (e.g. eGeMAPS R11) already gives it, delete the
   librosa/praat duplicate.
2. **eGeMAPS-first for audio.** Emit the 88 eGeMAPS functionals as named columns
   first; only add the §10.2 contour/shape features on top. This alone removes
   ~34 audio collisions.
3. **Fix the double jitter/shimmer.** `pipeline_audio.py` computes jitter/shimmer
   in both `EGeMAPSExtractor` (R11) and `JitterShimmerAnalyzer` (R13) — collapse
   to one; the current duplication is a live collision.
4. **Generic trend operator (ADV7), not per-family.** Implement ADV7 once as
   slope+curvature over any time-stamped feature; drop the six hand-rolled
   `*_trend`/`*_slope`/`*_trajectory` columns.
5. **Label-derived stays out.** TAB-008…013 (ADV4/P39 PHQ sub-scores) collide with
   the target and leak; keep them in an analysis-only table, not model input.
6. **Version the schema.** Each output column carries `(name, spec_step, source)`
   so a collision is a build error, not a silent duplicate.

---

Plus paired speaker-normalized variants and per-feature reliability gates, the
effective tensor width is roughly 2× the **net-new ~156**. This is the
"imagine a person talking in an interview" surface — every kept feature is a
hand-designed, explainable signal tied to a depression marker, not a learned
embedding, and none of it collides with the 108-step specification.
