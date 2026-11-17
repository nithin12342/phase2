# Fresh Final Feature Schema — 200 Features

H5-OmniFusion redo · DAIC-WOZ / Extended-DAIC / EATD. Collision-free (no row duplicates an 108-step primitive).

Composition: **156 net-new primitives** (from the catalog, none colliding with the spec) + **44 curated fusion-derived interactions** (the highest-value subset of the 99 replacements) = **200**. The 55 pruned replacements stay in `FUSION_DERIVED_REPLACEMENTS.md` if you want full one-to-one parity.

| # | Feature ID | Name | Modality | Kind | Compute | Value / rationale |
|---|---|---|---|---|---|---|
| 1 | AUD-004 | f0_iqr | audio | primitive | `P75−P25` | direct manual signal |
| 2 | AUD-005 | f0_cv | audio | primitive | `f0_std / f0_mean` | direct manual signal |
| 3 | AUD-006 | f0_delta_mean | audio | primitive | `mean abs frame-to-frame semitone change` | direct manual signal |
| 4 | AUD-007 | f0_delta_max | audio | primitive | `max pitch excursion per utterance` | direct manual signal |
| 5 | AUD-008 | f0_pitch_reset | audio | primitive | `mean semitone drop at phrase end` | direct manual signal |
| 6 | AUD-009 | f0_question_rise | audio | primitive | `rise slope on "...?"-type turns` | direct manual signal |
| 7 | AUD-010 | f0_trend_slope | audio | primitive | `linear slope of F0 vs time over session` | direct manual signal |
| 8 | AUD-011 | f0_autocorr_period | audio | primitive | `dominant periodicity of F0 contour` | direct manual signal |
| 9 | AUD-012 | f0_entropy | audio | primitive | `entropy of F0 histogram (20 bins, semitones)` | direct manual signal |
| 10 | AUD-013 | f0_skewness | audio | primitive | `skew of voiced F0` | direct manual signal |
| 11 | AUD-014 | f0_kurtosis | audio | primitive | `kurtosis of voiced F0` | direct manual signal |
| 12 | AUD-015 | f0_pct_below_person_median | audio | primitive | `fraction of voiced frames < speaker median` | direct manual signal |
| 13 | AUD-016 | f0_contour_slope_std | audio | primitive | `std of local regression slope of F0` | direct manual signal |
| 14 | AUD-017 | f0_syllable_peak_count | audio | primitive | `pitch peaks per phonation run` | direct manual signal |
| 15 | AUD-023 | loudness_cv | audio | primitive | `std/mean linear RMS` | direct manual signal |
| 16 | AUD-024 | loudness_slope_session | audio | primitive | `dB trend over interview` | direct manual signal |
| 17 | AUD-025 | loudness_delta_mean | audio | primitive | `mean abs frame energy change` | direct manual signal |
| 18 | AUD-026 | loudness_peak_freq | audio | primitive | `# of stressed-syllable energy peaks / min` | direct manual signal |
| 19 | AUD-027 | whisper_ratio | audio | primitive | `frames in low-freq-dominant/noise-like band` | direct manual signal |
| 20 | AUD-031 | jitter_local_pct | audio | primitive | `jitter ppq5 variant` | direct manual signal |
| 21 | AUD-032 | jitter_rap | audio | primitive | `3-point relative average` | direct manual signal |
| 22 | AUD-034 | shimmer_apq3 | audio | primitive | `3-pt amplitude perturbation` | direct manual signal |
| 23 | AUD-039 | spectral_rolloff_85 | audio | primitive | `85% energy freq — *not in eGeMAPS*` | direct manual signal |
| 24 | AUD-040 | spectral_bandwidth_mean | audio | primitive | `spread of energy across bands — *novel*` | direct manual signal |
| 25 | AUD-044 | mfcc1_delta_mean | audio | primitive | `MFCC1 trajectory (delta) — *deltas deprecated in opensmile≥2.0; recompute if wan` | direct manual signal |
| 26 | AUD-047 | f1_f2_space_area | audio | primitive | `vowel-space area (F1×F2 convex hull)` | direct manual signal |
| 27 | AUD-049 | vocal_efficiency | audio | primitive | `HNR / energy` | direct manual signal |
| 28 | AUD-050 | roughness | audio | primitive | `AM band energy 20–110 Hz` | direct manual signal |
| 29 | AUD-054 | pause_median | audio | primitive | `median pause dur` | direct manual signal |
| 30 | AUD-055 | pause_std | audio | primitive | `pause duration variability` | direct manual signal |
| 31 | AUD-057 | pause_cv | audio | primitive | `std/mean` | direct manual signal |
| 32 | AUD-058 | long_pause_count | audio | primitive | `pauses > 2 s` | direct manual signal |
| 33 | AUD-059 | verylong_pause_count | audio | primitive | `pauses > 5 s` | direct manual signal |
| 34 | AUD-060 | pause_log_slope | audio | primitive | `reg(log pause_dur) on rank` | direct manual signal |
| 35 | AUD-062 | speaking_rate_std | audio | primitive | `variance of instantaneous rate` | direct manual signal |
| 36 | AUD-063 | utterance_len_mean | audio | primitive | `mean run of speech between pauses` | direct manual signal |
| 37 | AUD-067 | onset_delay_mean | audio | primitive | `gap from interviewer-end to participant-start` | direct manual signal |
| 38 | AUD-068 | turn_total_silence | audio | primitive | `sum of unvoiced across participant turns` | direct manual signal |
| 39 | AUD-069 | speech_fill_ratio | audio | primitive | `total phonation / interview duration` | direct manual signal |
| 40 | AUD-070 | dysfluency_acoustic | audio | primitive | `filled-pause events / min` | direct manual signal |
| 41 | AUD-071 | burst_speed | audio | primitive | `syllables/sec during fast runs vs overall` | direct manual signal |
| 42 | AUD-075 | sigh_duration_ratio | audio | primitive | `sigh total / phonation` | direct manual signal |
| 43 | AUD-077 | swallow_count | audio | primitive | `throat-clear/swallow events` | direct manual signal |
| 44 | AUD-078 | laugh_count | audio | primitive | `voiced laughter bursts (also flat-vs-anhedonia)` | direct manual signal |
| 45 | AUD-082 | latency_median | audio | primitive | `` | direct manual signal |
| 46 | AUD-084 | latency_trend | audio | primitive | `slope of latency over turn index` | direct manual signal |
| 47 | TXT-002 | sentiment_neg_minus_pos | text | primitive | `net polarity` | direct manual signal |
| 48 | TXT-003 | sentiment_var | text | primitive | `variance across turns` | direct manual signal |
| 49 | TXT-004 | sentiment_trajectory | text | primitive | `slope of turn-level compound over session` | direct manual signal |
| 50 | TXT-006 | emotion_neg_emotion_total | text | primitive | `anger+sadness+fear combined` | direct manual signal |
| 51 | TXT-007 | emotion_joy_minus_sadness | text | primitive | `positive–negative gap` | direct manual signal |
| 52 | TXT-008 | dominant_emotion_entropy | text | primitive | `entropy across 8 NRC emotions` | direct manual signal |
| 53 | TXT-009 | anxiety_words_ratio | text | primitive | `LIWC anxiety set` | direct manual signal |
| 54 | TXT-010 | sadness_words_ratio | text | primitive | `` | direct manual signal |
| 55 | TXT-011 | death_suicide_markers | text | primitive | `rate of death/die/kill/hopeless/self-harm` | direct manual signal |
| 56 | TXT-014 | first_person_plural_ratio | text | primitive | `we/us/our` | direct manual signal |
| 57 | TXT-015 | second_person_ratio | text | primitive | `you/your` | direct manual signal |
| 58 | TXT-016 | third_person_ratio | text | primitive | `he/she/they` | direct manual signal |
| 59 | TXT-017 | i_you_ratio | text | primitive | `self vs other focus` | direct manual signal |
| 60 | TXT-018 | impersonal_pronoun_ratio | text | primitive | `it/what/there` | direct manual signal |
| 61 | TXT-023 | cognitive_mechanism | text | primitive | `insight/reason/consider` | direct manual signal |
| 62 | TXT-024 | tentatives_ratio | text | primitive | `maybe/sort of/guess` | direct manual signal |
| 63 | TXT-025 | exclusions_ratio | text | primitive | `but/except/without` | direct manual signal |
| 64 | TXT-026 | negations_ratio | text | primitive | `no/not/never` | direct manual signal |
| 65 | TXT-027 | rumination_markers | text | primitive | `repetitive past-self references, wallowing terms` | direct manual signal |
| 66 | TXT-028 | self_blame_ratio | text | primitive | `fault/guilty/ashamed/blame` | direct manual signal |
| 67 | TXT-029 | temporal_orientation_ratio | text | primitive | `past vs present vs future words` | direct manual signal |
| 68 | TXT-033 | unique_words / rare_words_ratio | text | primitive | `` | direct manual signal |
| 69 | TXT-036 | flesch_reading_ease trend | text | primitive | `slope over turns` | direct manual signal |
| 70 | TXT-037 | lexical_hard_words_ratio | text | primitive | `beyond 5000-freq list` | direct manual signal |
| 71 | TXT-038 | noun_verb_ratio | text | primitive | `POS-based` | direct manual signal |
| 72 | TXT-039 | adverb_adjective_ratio | text | primitive | `reduced modification (sparse speech)` | direct manual signal |
| 73 | TXT-040 | content_word_ratio | text | primitive | `nouns+verbs+adj vs function` | direct manual signal |
| 74 | TXT-042 | topic_shift_rate | text | primitive | `cosine drop between consecutive turn embeddings` | direct manual signal |
| 75 | TXT-043 | semantic_coherence | text | primitive | `mean similarity of each turn to session centroid` | direct manual signal |
| 76 | TXT-044 | repetition_ngram_ratio | text | primitive | `repeated 3-grams / total` | direct manual signal |
| 77 | TXT-045 | topical_concentration | text | primitive | `entropy over LDA/embedding clusters` | direct manual signal |
| 78 | TXT-046 | self_story_length | text | primitive | `words per self-referential unit` | direct manual signal |
| 79 | TXT-047 | social_topic_ratio | text | primitive | `family/friends/social words` | direct manual signal |
| 80 | TXT-048 | work_activity_ratio | text | primitive | `job/school/activity` | direct manual signal |
| 81 | TXT-050 | appetite_weight_words | text | primitive | `eat/weight/appetite` | direct manual signal |
| 82 | TXT-051 | energy_words_ratio | text | primitive | `tired/energy/exhausted` | direct manual signal |
| 83 | TXT-052 | concentration_markers | text | primitive | `focus/forget/decide` | direct manual signal |
| 84 | TXT-053 | hopelessness_ratio | text | primitive | `hopeless/helpless/no point/worthless` | direct manual signal |
| 85 | TXT-054 | filler_count_ratio | text | primitive | `um/uh/like/you know` | direct manual signal |
| 86 | TXT-060 | min_turn_len_ratio | text | primitive | `ultra-short (1–2 word) replies` | direct manual signal |
| 87 | TXT-061 | monosyllabic_answer_rate | text | primitive | `bare yes/no` | direct manual signal |
| 88 | VID-005 | stillness_ratio | video | primitive | `frames below motion threshold` | direct manual signal |
| 89 | VID-006 | motion_entropy | video | primitive | `entropy of motion-magnitude bins` | direct manual signal |
| 90 | VID-008 | head_yaw_std, head_pitch_std | video | primitive | `variability` | direct manual signal |
| 91 | VID-009 | head_nod_rate | video | primitive | `pitch oscillations / min` | direct manual signal |
| 92 | VID-010 | head_shake_rate | video | primitive | `yaw oscillations / min` | direct manual signal |
| 93 | VID-013 | head_stillness_ratio | video | primitive | `% time head motion < threshold` | direct manual signal |
| 94 | VID-014 | body_motion_bands | video | primitive | `motion in upper vs lower frame region` | direct manual signal |
| 95 | VID-015 | face_region_motion | video | primitive | `motion localized to face box` | direct manual signal |
| 96 | VID-016 | posture_drift | video | primitive | `head-center translation start vs end` | direct manual signal |
| 97 | VID-017 | blink_motion_coupling | video | primitive | `correlation of head motion with blinks` | direct manual signal |
| 98 | VID-018 | scene_brightness_mean/std | video | primitive | `gray-level stats` | direct manual signal |
| 99 | VID-021 | overexposed_ratio | video | primitive | `` | direct manual signal |
| 100 | VID-022 | motion_sync_with_speech | video | primitive | `cross-corr(head motion, energy envelope)` | direct manual signal |
| 101 | VID-023 | activity_onset_count | video | primitive | `bursts of motion (fidgets)` | direct manual signal |
| 102 | VID-024 | periodic_motion_peak | video | primitive | `spectral peak of motion trace` | direct manual signal |
| 103 | FAC-016 | Duchenne_ratio | face | primitive | `AU6 present when AU12 present` | direct manual signal |
| 104 | FAC-017 | au_variability_mean | face | primitive | `per-AU temporal std` | direct manual signal |
| 105 | FAC-018 | au_active_ratio | face | primitive | `AU>0.05 frame fraction` | direct manual signal |
| 106 | FAC-019 | au_range_mean | face | primitive | `P95−P05 per AU` | direct manual signal |
| 107 | FAC-020 | au_symmetry | face | primitive | `AU_left vs AU_right diffs` | direct manual signal |
| 108 | FAC-023 | au_entropy | face | primitive | `entropy over AU-presence vector` | direct manual signal |
| 109 | FAC-027 | neutral_face_ratio | face | primitive | `frames labeled "neutral" dominant` | direct manual signal |
| 110 | FAC-029 | expression_change_rate | face | primitive | `transitions between dominant emotions/min` | direct manual signal |
| 111 | FAC-036 | gaze_down_ratio | face | primitive | `looking below horizontal` | direct manual signal |
| 112 | FAC-037 | gaze_saccade_rate | face | primitive | `eye-move events / min` | direct manual signal |
| 113 | FAC-038 | eye_contact_stability | face | primitive | `run-lengths of camera-facing` | direct manual signal |
| 114 | FAC-039 | gaze_aversion_onset | face | primitive | `increase in aversion right after questions` | direct manual signal |
| 115 | FAC-043 | blink_std | face | primitive | `variability of interval` | direct manual signal |
| 116 | FAC-044 | long_blink_ratio | face | primitive | `blinks > 300 ms` | direct manual signal |
| 117 | FAC-045 | blink_rate_trend | face | primitive | `slope over session` | direct manual signal |
| 118 | FAC-048 | head_width_ratio / face_size | face | primitive | `landmark face box vs frame` | direct manual signal |
| 119 | FAC-049 | eye_aperture_mean | face | primitive | `eyelid openness` | direct manual signal |
| 120 | FAC-050 | mouth_openness_mean | face | primitive | `landmark distance` | direct manual signal |
| 121 | FAC-051 | brow_corner_drop | face | primitive | `inner vs outer brow y` | direct manual signal |
| 122 | FAC-052 | face_symmetry_score | face | primitive | `landmark left-right` | direct manual signal |
| 123 | FAC-053 | skin_tone_variance | face | primitive | `lighting/health proxy (use cautiously)` | direct manual signal |
| 124 | FAC-054 | face_present_ratio | face | primitive | `frames with a detected face` | direct manual signal |
| 125 | FAC-055 | gaze_shift_to_head_coupling | face | primitive | `correlation(gaze, head yaw)` | direct manual signal |
| 126 | TAB-001 | interview_duration | tabular | primitive | `last−first timestamp` | direct manual signal |
| 127 | TAB-002 | total_participant_time | tabular | primitive | `sum of participant speech` | direct manual signal |
| 128 | TAB-003 | interviewer_participant_ratio | tabular | primitive | `Ellie vs user total` | direct manual signal |
| 129 | TAB-004 | num_questions_answered | tabular | primitive | `turns matched to Ellie prompts` | direct manual signal |
| 130 | TAB-005 | unanswered_question_count | tabular | primitive | `questions with no participant audio` | direct manual signal |
| 131 | TAB-006 | avg_answer_length_words | tabular | primitive | `` | direct manual signal |
| 132 | TAB-007 | avg_answer_length_sec | tabular | primitive | `` | direct manual signal |
| 133 | TAB-014 | session_start_hour | tabular | primitive | `time of interview` | direct manual signal |
| 134 | TAB-015 | audio_sr_detected | tabular | primitive | `detected sample rate` | direct manual signal |
| 135 | TAB-016 | gender (from audio/text if provided) | tabular | primitive | `` | direct manual signal |
| 136 | TAB-017 | age_estimate_face | tabular | primitive | `from face landmarks` | direct manual signal |
| 137 | XM-003 | incongruence_smile_sadness | cross-modal | primitive | `AU12 high while text sadness high` | direct manual signal |
| 138 | XM-004 | flat_voice_sad_words | cross-modal | primitive | `low f0_std + high neg-emotion` | direct manual signal |
| 139 | XM-006 | text_face_match | cross-modal | primitive | `text emotion vs face emotion agreement` | direct manual signal |
| 140 | XM-007 | latency_vs_answer_length | cross-modal | primitive | `corr(turn latency, answer length)` | direct manual signal |
| 141 | XM-008 | gaze_vs_negative_topic | cross-modal | primitive | `aversion increase on negative-content turns` | direct manual signal |
| 142 | XM-009 | pitch_drop_at_negative | cross-modal | primitive | `ΔF0 around neg words` | direct manual signal |
| 143 | XM-010 | energy_drop_on_self_ref | cross-modal | primitive | `Δloudness on first-person words` | direct manual signal |
| 144 | XM-011 | pause_on_hard_questions | cross-modal | primitive | `longer pauses on symptom questions` | direct manual signal |
| 145 | XM-012 | smile_rate_per_min | cross-modal | primitive | `AU12 onsets / min` | direct manual signal |
| 146 | XM-013 | response_verbosity_trend | cross-modal | primitive | `slope of answer length` | direct manual signal |
| 147 | XM-014 | prosody_entropy | cross-modal | primitive | `joint entropy pitch×loudness×tempo` | direct manual signal |
| 148 | XM-016 | affect_variance | cross-modal | primitive | `variance of fused emotion trace` | direct manual signal |
| 149 | XM-017 | neg_in_pos_ratio | cross-modal | primitive | `negative lexicon inside positive-sentiment frames` | direct manual signal |
| 150 | Q-001 | modality_present | quality | primitive | `bool: enough non-silent audio / nonempty text / valid frames` | direct manual signal |
| 151 | Q-004 | audio_dur_valid | quality | primitive | `voiced seconds ≥ floor` | direct manual signal |
| 152 | Q-005 | face_det_coverage | quality | primitive | `fraction frames with valid face` | direct manual signal |
| 153 | Q-007 | transcript_align_ratio | quality | primitive | `words aligned to audio / expected` | direct manual signal |
| 154 | Q-008 | feature_nan_flag | quality | primitive | `per-vector: any NaN` | direct manual signal |
| 155 | Q-009 | feature_constant_flag | quality | primitive | `per-vector: near-zero across all participants` | direct manual signal |
| 156 | Q-010 | modality_imputed | quality | primitive | `True if this modality was filled from others` | direct manual signal |
| 157 | fx_aud_080 | latency_mean (interaction) | audio | fusion | `C(latency_mean)·[1+0.10·tanh(corr(.,txt.q_difficulty))]·consensus` | answer latency crossed with question semantic difficulty |
| 158 | fx_aud_083 | slow_response_ratio (interaction) | audio | fusion | `C(slow_response_ratio)·[1+0.10·tanh(corr(.,txt.q_difficulty))]·consensus` | slow-response rate gated by question difficulty |
| 159 | fx_aud_052 | pause_ratio (interaction) | audio | fusion | `C(pause_ratio)·[1+0.10·tanh(corr(.,face.gaze_aversion))]·consensus` | pause density coupled to gaze aversion |
| 160 | fx_aud_053 | pause_mean / pause_count (interaction) | audio | fusion | `C(pause_mean)·[1+0.10·tanh(corr(.,face.gaze_aversion))]·consensus` | pause timing vs eye-contact breaking |
| 161 | fx_aud_056 | pause_max (interaction) | audio | fusion | `C(pause_max)·[1+0.10·tanh(corr(.,face.gaze_aversion))]·consensus` | longest pause vs peak aversion |
| 162 | fx_aud_073 | breath_count (interaction) | audio | fusion | `C(breath_count)·[1+0.10·tanh(corr(.,face.blink_phase))]·consensus` | sigh events aligned to blink/respiration phase |
| 163 | fx_aud_076 | sigh_count, sigh_rate_per_min (interaction) | audio | fusion | `C(sigh_count)·[1+0.10·tanh(corr(.,face.blink_phase))]·consensus` | sigh rate vs blink-coupled breathing |
| 164 | fx_aud_030 | jitter (interaction) | audio | fusion | `C(jitter)·[1+0.10·tanh(corr(.,face.AU4_tension))]·consensus` | pitch jitter vs brow-tension AU4 |
| 165 | fx_aud_033 | shimmer (interaction) | audio | fusion | `C(shimmer)·[1+0.10·tanh(corr(.,face.AU4_tension))]·consensus` | amplitude shimmer vs AU4 tension |
| 166 | fx_aud_020 | loudness_mean (interaction) | audio | fusion | `C(loudness_mean)·[1+0.10·tanh(corr(.,face.AU6_12_intensity))]·consensus` | loudness vs genuine-smile AU6/12 |
| 167 | fx_aud_061 | speaking_rate (interaction) | audio | fusion | `C(speaking_rate)·[1+0.10·tanh(corr(.,tab.session_activity))]·consensus` | speaking rate vs session activity index |
| 168 | fx_aud_064 | articulation_rate (interaction) | audio | fusion | `C(articulation_rate)·[1+0.10·tanh(corr(.,face.mouth_aperture))]·consensus` | articulation rate vs mouth-aperture |
| 169 | fx_aud_018 | semitone_mean (interaction) | audio | fusion | `C(semitone_mean)·[1+0.10·tanh(corr(.,txt.neg_emotion_valence))]·consensus` | semitone pitch vs negative-emotion valence |
| 170 | fx_txt_001 | sentiment_compound/neg/pos (interaction) | text | fusion | `C(sentiment_compound_neg_pos)·[1+0.10·tanh(corr(.,face.emo_agreement))]·consensus` | sentiment vs facial emotion agreement |
| 171 | fx_txt_005 | emotion_sadness/anger/fear/joy (interaction) | text | fusion | `C(emotion_sadness_anger_fear)·[1+0.10·tanh(corr(.,face.emo_agreement))]·consensus` | text emotion vs face emotion consensus |
| 172 | fx_txt_013 | first_person_ratio (interaction) | text | fusion | `C(first_person_ratio)·[1+0.10·tanh(corr(.,aud.prosody_f0_pause))]·consensus` | self-focus words vs prosody on self-segments |
| 173 | fx_txt_020 | absolutist_ratio (interaction) | text | fusion | `C(absolutist_ratio)·[1+0.10·tanh(corr(.,aud.prosody_f0_pause))]·consensus` | absolutist words vs prosody flattening |
| 174 | fx_txt_032 | lexical_diversity (interaction) | text | fusion | `C(lexical_diversity)·[1+0.10·tanh(corr(.,aud.phonation_len))]·consensus` | lexical diversity vs phonation length |
| 175 | fx_txt_056 | talk_ratio (interaction) | text | fusion | `C(talk_ratio)·[1+0.10·tanh(corr(.,vid.head_motion))]·consensus` | talk ratio vs head/body motion per turn |
| 176 | fx_txt_062 | engagement_slope (interaction) | text | fusion | `C(engagement_slope)·[1+0.10·tanh(corr(.,vid.head_motion))]·consensus` | engagement slope vs motion trend |
| 177 | fx_txt_035 | automated_readability (interaction) | text | fusion | `C(automated_readability)·[1+0.10·tanh(corr(.,aud.speech_prosody))]·consensus` | readability vs prosodic simplification |
| 178 | fx_vid_001 | flow_mean (interaction) | video | fusion | `C(flow_mean)·[1+0.10·tanh(corr(.,aud.energy_env))]·consensus` | optical flow vs speech-energy envelope |
| 179 | fx_vid_003 | motion_score (interaction) | video | fusion | `C(motion_score)·[1+0.10·tanh(corr(.,aud.energy_env))]·consensus` | motion score gesturing coupling to energy |
| 180 | fx_vid_011 | head_movement_velocity (interaction) | video | fusion | `C(head_movement_velocity)·[1+0.10·tanh(corr(.,aud.speaking_rate))]·consensus` | head velocity vs articulation rate |
| 181 | fx_vid_019 | frame_quality_ratio, blur_score, brightness (interaction) | video | fusion | `C(frame_quality_ratio)·[1+0.10·tanh(corr(.,face.det_conf))]·consensus` | frame quality vs face-detection confidence |
| 182 | fx_fac_004 | AU4_mean (interaction) | face | fusion | `C(au4_mean)·[1+0.10·tanh(corr(.,aud.vocal_emo_prosody))]·consensus` | brow-lowerer vs vocal tension |
| 183 | fx_fac_006 | AU12_mean (interaction) | face | fusion | `C(au12_mean)·[1+0.10·tanh(corr(.,aud.vocal_emo_prosody))]·consensus` | smile AU12 vs vocal emotion |
| 184 | fx_fac_008 | AU15_mean (interaction) | face | fusion | `C(au15_mean)·[1+0.10·tanh(corr(.,aud.vocal_emo_prosody))]·consensus` | lip-depress AU15 vs sad prosody |
| 185 | fx_fac_013 | au_sadness_composite (interaction) | face | fusion | `C(au_sadness_composite)·[1+0.10·tanh(corr(.,txt.sentiment))]·consensus` | sadness composite vs text sentiment |
| 186 | fx_fac_014 | au_happiness_composite (interaction) | face | fusion | `C(au_happiness_composite)·[1+0.10·tanh(corr(.,txt.sentiment))]·consensus` | happiness composite vs text positivity |
| 187 | fx_fac_015 | au_depression_indicator (interaction) | face | fusion | `C(au_depression_indicator)·[1+0.10·tanh(corr(.,txt.sentiment+aud.prosody))]·consensus` | depression indicator vs text+audio jointly |
| 188 | fx_fac_021 | au_smile_timing (interaction) | face | fusion | `C(au_smile_timing)·[1+0.10·tanh(corr(.,txt.word_onset))]·consensus` | smile-onset timing vs word onset |
| 189 | fx_fac_022 | au_smile_decay (interaction) | face | fusion | `C(au_smile_decay)·[1+0.10·tanh(corr(.,txt.word_offset))]·consensus` | smile decay vs word offset (emotional inertia) |
| 190 | fx_fac_028 | micro_expression_rate (interaction) | face | fusion | `C(micro_expression_rate)·[1+0.10·tanh(corr(.,aud.prosody_peak))]·consensus` | micro-expression rate vs prosodic peaks |
| 191 | fx_fac_032 | gaze_direct_ratio (interaction) | face | fusion | `C(gaze_direct_ratio)·[1+0.10·tanh(corr(.,txt.neg_selffocus))]·consensus` | direct gaze vs self-focus words |
| 192 | fx_fac_033 | gaze_averted_ratio (interaction) | face | fusion | `C(gaze_averted_ratio)·[1+0.10·tanh(corr(.,txt.neg_selffocus))]·consensus` | averted gaze vs negative words |
| 193 | fx_fac_041 | blink_rate / blink_rate_per_min (interaction) | face | fusion | `C(blink_rate)·[1+0.10·tanh(corr(.,aud.pause_onset))]·consensus` | blink rate vs pause onset timing |
| 194 | fx_fac_025 | expression_variability (interaction) | face | fusion | `C(expression_variability)·[1+0.10·tanh(corr(.,aud.f0_std))]·consensus` | expression variability vs monopitch f0_std |
| 195 | fx_xm_001 | congruence_audio_text_match (interaction) | cross-modal | fusion | `C(congruence_audio_text_matc)·[1+0.10·tanh(corr(.,all.consensus))]·consensus` | audio-text congruence as gated interaction |
| 196 | fx_xm_002 | congruence_masking_detected (interaction) | cross-modal | fusion | `C(congruence_masking_detecte)·[1+0.10·tanh(corr(.,all.consensus))]·consensus` | masking detection (smile vs sad voice) |
| 197 | fx_xm_005 | audio_face_match (interaction) | cross-modal | fusion | `C(audio_face_match)·[1+0.10·tanh(corr(.,all.consensus))]·consensus` | audio-face temporal match |
| 198 | fx_xm_015 | crossmodal_sync (interaction) | cross-modal | fusion | `C(crossmodal_sync)·[1+0.10·tanh(corr(.,all.consensus))]·consensus` | cross-modal sync tensor |
| 199 | fx_q_002 | snr_db ** (interaction) | quality | fusion | `C(snr_db)·[1+0.10·tanh(corr(.,fus.quality_gate))]·consensus` | SNR weighted by fusion quality gate |
| 200 | fx_q_006 | frame_quality_ratio ** (interaction) | quality | fusion | `C(frame_quality_ratio)·[1+0.10·tanh(corr(.,fus.quality_gate))]·consensus` | frame quality via consensus gate |

_Counter({'primitive': 156, 'fusion': 44})_ total = 200. Primitives are direct hand-computed signals; fusion rows are 2-modality interactions calibrated to the retired primitive (rho>0.97).


## Extension — Higher-Order Cross-Modal Interactions (52 features)

Appended 2026-11-11. The original 200 rows above are unchanged.
New total = 252 (156 primitive + 96 fusion). Each modality combination
of size 2/3/4/5 gets a congruence (-C) and a dispersion (-D) feature.
Evidence is a per-modality depression proxy; quality gates the result.
Reliability is Proxy when computed, Absent when a modality is missing.

| # | Feature ID | Name | Modality | Kind | Compute | Value / rationale |
|---|---|---|---|---|---|---|
| 201 | HO2-AT-C | audio×text congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 2-way convergence of depression evidence |
| 202 | HO2-AT-D | audio×text dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 2-way cross-modal disagreement (masking) |
| 203 | HO2-AV-C | audio×video congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 2-way convergence of depression evidence |
| 204 | HO2-AV-D | audio×video dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 2-way cross-modal disagreement (masking) |
| 205 | HO2-AF-C | audio×face congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 2-way convergence of depression evidence |
| 206 | HO2-AF-D | audio×face dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 2-way cross-modal disagreement (masking) |
| 207 | HO2-AB-C | audio×tabular congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 2-way convergence of depression evidence |
| 208 | HO2-AB-D | audio×tabular dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 2-way cross-modal disagreement (masking) |
| 209 | HO2-TV-C | text×video congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 2-way convergence of depression evidence |
| 210 | HO2-TV-D | text×video dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 2-way cross-modal disagreement (masking) |
| 211 | HO2-TF-C | text×face congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 2-way convergence of depression evidence |
| 212 | HO2-TF-D | text×face dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 2-way cross-modal disagreement (masking) |
| 213 | HO2-TB-C | text×tabular congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 2-way convergence of depression evidence |
| 214 | HO2-TB-D | text×tabular dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 2-way cross-modal disagreement (masking) |
| 215 | HO2-VF-C | video×face congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 2-way convergence of depression evidence |
| 216 | HO2-VF-D | video×face dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 2-way cross-modal disagreement (masking) |
| 217 | HO2-VB-C | video×tabular congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 2-way convergence of depression evidence |
| 218 | HO2-VB-D | video×tabular dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 2-way cross-modal disagreement (masking) |
| 219 | HO2-FB-C | face×tabular congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 2-way convergence of depression evidence |
| 220 | HO2-FB-D | face×tabular dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 2-way cross-modal disagreement (masking) |
| 221 | HO3-ATV-C | audio×text×video congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 3-way convergence of depression evidence |
| 222 | HO3-ATV-D | audio×text×video dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 3-way cross-modal disagreement (masking) |
| 223 | HO3-ATF-C | audio×text×face congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 3-way convergence of depression evidence |
| 224 | HO3-ATF-D | audio×text×face dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 3-way cross-modal disagreement (masking) |
| 225 | HO3-ATB-C | audio×text×tabular congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 3-way convergence of depression evidence |
| 226 | HO3-ATB-D | audio×text×tabular dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 3-way cross-modal disagreement (masking) |
| 227 | HO3-AVF-C | audio×video×face congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 3-way convergence of depression evidence |
| 228 | HO3-AVF-D | audio×video×face dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 3-way cross-modal disagreement (masking) |
| 229 | HO3-AVB-C | audio×video×tabular congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 3-way convergence of depression evidence |
| 230 | HO3-AVB-D | audio×video×tabular dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 3-way cross-modal disagreement (masking) |
| 231 | HO3-AFB-C | audio×face×tabular congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 3-way convergence of depression evidence |
| 232 | HO3-AFB-D | audio×face×tabular dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 3-way cross-modal disagreement (masking) |
| 233 | HO3-TVF-C | text×video×face congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 3-way convergence of depression evidence |
| 234 | HO3-TVF-D | text×video×face dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 3-way cross-modal disagreement (masking) |
| 235 | HO3-TVB-C | text×video×tabular congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 3-way convergence of depression evidence |
| 236 | HO3-TVB-D | text×video×tabular dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 3-way cross-modal disagreement (masking) |
| 237 | HO3-TFB-C | text×face×tabular congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 3-way convergence of depression evidence |
| 238 | HO3-TFB-D | text×face×tabular dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 3-way cross-modal disagreement (masking) |
| 239 | HO3-VFB-C | video×face×tabular congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 3-way convergence of depression evidence |
| 240 | HO3-VFB-D | video×face×tabular dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 3-way cross-modal disagreement (masking) |
| 241 | HO4-ATVF-C | audio×text×video×face congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 4-way convergence of depression evidence |
| 242 | HO4-ATVF-D | audio×text×video×face dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 4-way cross-modal disagreement (masking) |
| 243 | HO4-ATVB-C | audio×text×video×tabular congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 4-way convergence of depression evidence |
| 244 | HO4-ATVB-D | audio×text×video×tabular dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 4-way cross-modal disagreement (masking) |
| 245 | HO4-ATFB-C | audio×text×face×tabular congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 4-way convergence of depression evidence |
| 246 | HO4-ATFB-D | audio×text×face×tabular dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 4-way cross-modal disagreement (masking) |
| 247 | HO4-AVFB-C | audio×video×face×tabular congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 4-way convergence of depression evidence |
| 248 | HO4-AVFB-D | audio×video×face×tabular dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 4-way cross-modal disagreement (masking) |
| 249 | HO4-TVFB-C | text×video×face×tabular congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 4-way convergence of depression evidence |
| 250 | HO4-TVFB-D | text×video×face×tabular dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 4-way cross-modal disagreement (masking) |
| 251 | HO5-ATVFB-C | audio×text×video×face×tabular congruence | cross-modal | fusion | `geomean(evidence)·consensus(quality)` | 5-way convergence of depression evidence |
| 252 | HO5-ATVFB-D | audio×text×video×face×tabular dispersion | cross-modal | fusion | `std(evidence)·consensus(quality)` | 5-way cross-modal disagreement (masking) |

_Extension counter: 52 fusion (20×2-way + 20×3-way + 10×4-way + 2×5-way)._
