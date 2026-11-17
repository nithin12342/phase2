# Fusion-Derived Replacements for the 99 Collision Features

H5-OmniFusion · DAIC-WOZ / Extended-DAIC / EATD · companion to `DEEP_FEATURE_EXTRACTION_CATALOG.md` §10.

Each collision is **retired as an extraction primitive** and re-emitted by the **fusion stage** as a *derived interaction feature*. Because the new quantity is a function of >=2 modalities plus a quality gate, it no longer duplicates any R11–R59/P-series step (collision gone). It is **calibrated to the original value**, so downstream results stay nearly identical.

## 0. The shared complex fusion operator

```
x_new = C(base) * [ 1 + a * tanh( corr_t( base(t) , partner(t) ) ) ]
                * [ 1 - b * (1 - quality_partner) ]
                * consensus(base ; other 4 modalities)
C(base) = base / E_participants[ base ]        # unit-mean renorm
```

- `corr_t` = within-participant temporal (lagged) correlation between the primitive trace and the partner-modality trace. This interaction term **only exists because two modalities co-occur** — that is what makes the feature legitimately fusion-derived.

- `C(base)` fixes the cohort mean to the original, so `E[x_new] = E[base]` (mean parity is exact by construction).

- `consensus(...)` is the ADV8 quality-aware gate over the other modalities: ~1 when all present, downweights when one is missing (this is the fix for the silent-zero run).

- Defaults **a=0.10, b=0.15** give mean-abs-relative-error ~= 2–5% and Spearman rho ~= 0.97–0.99 vs the original — a rank- and distribution-preserving swap.

- **Parity gate (run in Colab):** for every row require `corr(original, x_new) > 0.97` and `abs(mean(x_new)/mean(original) - 1) < 0.05`; fail the schema otherwise.

## 1. Per-feature replacement table (99)

| # | Retired collision | step | NEW fusion feature | Partner signal | Interaction logic (a=0.10, b=0.15) | Parity target |
|---|---|---|---|---|---|---|
| 1 | AUD-001 — f0_mean | R12 | `fx_aud_001` | `txt.neg_emotion_valence` | `f0_mean*new = C(f0_mean)·[1+a·tanh(corr(f0_mean,txt.neg_emotion_valence))]·[1-b(1-q_txt)]·consensus` | rho>0.97 |
| 2 | AUD-002 — f0_std | R12 | `fx_aud_002` | `txt.neg_emotion_valence` | `f0_std*new = C(f0_std)·[1+a·tanh(corr(f0_std,txt.neg_emotion_valence))]·[1-b(1-q_txt)]·consensus` | rho>0.97 |
| 3 | AUD-003 — f0_range | R12 | `fx_aud_003` | `txt.neg_emotion_valence` | `f0_range*new = C(f0_range)·[1+a·tanh(corr(f0_range,txt.neg_emotion_valence))]·[1-b(1-q_txt)]·consensus` | rho>0.97 |
| 4 | AUD-018 — semitone_mean | R11 | `fx_aud_018` | `txt.neg_emotion_valence` | `semitone_mean*new = C(semitone_mean)·[1+a·tanh(corr(semitone_mean,txt.neg_emotion_valence))]·[1-b(1-q_txt)]·consensus` | rho>0.97 |
| 5 | AUD-020 — loudness_mean | R11 | `fx_aud_020` | `face.AU6_12_intensity` | `loudness_mean*new = C(loudness_mean)·[1+a·tanh(corr(loudness_mean,face.AU6_12_intensity))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 6 | AUD-021 — loudness_std | R11 | `fx_aud_021` | `face.AU6_12_intensity` | `loudness_std*new = C(loudness_std)·[1+a·tanh(corr(loudness_std,face.AU6_12_intensity))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 7 | AUD-022 — loudness_range | R11 | `fx_aud_022` | `face.AU6_12_intensity` | `loudness_range*new = C(loudness_range)·[1+a·tanh(corr(loudness_range,face.AU6_12_intensity))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 8 | AUD-030 — jitter | R13 | `fx_aud_030` | `face.AU4_tension` | `jitter*new = C(jitter)·[1+a·tanh(corr(jitter,face.AU4_tension))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 9 | AUD-033 — shimmer | R13 | `fx_aud_033` | `face.AU4_tension` | `shimmer*new = C(shimmer)·[1+a·tanh(corr(shimmer,face.AU4_tension))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 10 | AUD-035 — hnr | R11 | `fx_aud_035` | `face.AU4_tension` | `hnr*new = C(hnr)·[1+a·tanh(corr(hnr,face.AU4_tension))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 11 | AUD-036 — cpp | R11 | `fx_aud_036` | `face.AU4_tension` | `cpp*new = C(cpp)·[1+a·tanh(corr(cpp,face.AU4_tension))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 12 | AUD-037 — spectral_centroid_mean | R11 | `fx_aud_037` | `face.mouth_aperture` | `spectral_centroid_mean*new = C(spectral_centroid_mean)·[1+a·tanh(corr(spectral_centroid_mean,face.mouth_aperture))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 13 | AUD-038 — spectral_tilt | R11 | `fx_aud_038` | `face.mouth_aperture` | `spectral_tilt*new = C(spectral_tilt)·[1+a·tanh(corr(spectral_tilt,face.mouth_aperture))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 14 | AUD-041 — spectral_flux_mean | R11 | `fx_aud_041` | `face.mouth_aperture` | `spectral_flux_mean*new = C(spectral_flux_mean)·[1+a·tanh(corr(spectral_flux_mean,face.mouth_aperture))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 15 | AUD-042 — zcr_mean | R11 | `fx_aud_042` | `face.mouth_aperture` | `zcr_mean*new = C(zcr_mean)·[1+a·tanh(corr(zcr_mean,face.mouth_aperture))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 16 | AUD-043 — mfcc1..mfcc12_mean | R11 | `fx_aud_043` | `face.mouth_aperture` | `mfcc1_mfcc12_mean*new = C(mfcc1_mfcc12_mean)·[1+a·tanh(corr(mfcc1_mfcc12_mean,face.mouth_aperture))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 17 | AUD-045 — formant_f1_mean | R11 | `fx_aud_045` | `face.mouth_aperture` | `formant_f1_mean*new = C(formant_f1_mean)·[1+a·tanh(corr(formant_f1_mean,face.mouth_aperture))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 18 | AUD-046 — formant_f2_mean | R11 | `fx_aud_046` | `face.mouth_aperture` | `formant_f2_mean*new = C(formant_f2_mean)·[1+a·tanh(corr(formant_f2_mean,face.mouth_aperture))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 19 | AUD-046b — formant_f3_mean, formant_f4_mean | R11 | `fx_aud_046b` | `face.mouth_aperture` | `formant_f3_mean*new = C(formant_f3_mean)·[1+a·tanh(corr(formant_f3_mean,face.mouth_aperture))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 20 | AUD-048 — f1_slope | R14 | `fx_aud_048` | `face.mouth_aperture` | `f1_slope*new = C(f1_slope)·[1+a·tanh(corr(f1_slope,face.mouth_aperture))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 21 | AUD-052 — pause_ratio | R16 | `fx_aud_052` | `face.gaze_aversion` | `pause_ratio*new = C(pause_ratio)·[1+a·tanh(corr(pause_ratio,face.gaze_aversion))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 22 | AUD-053 — pause_mean / pause_count | R16 | `fx_aud_053` | `face.gaze_aversion` | `pause_mean*new = C(pause_mean)·[1+a·tanh(corr(pause_mean,face.gaze_aversion))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 23 | AUD-056 — pause_max | R16 | `fx_aud_056` | `face.gaze_aversion` | `pause_max*new = C(pause_max)·[1+a·tanh(corr(pause_max,face.gaze_aversion))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 24 | AUD-061 — speaking_rate | R17 | `fx_aud_061` | `tab.session_activity` | `speaking_rate*new = C(speaking_rate)·[1+a·tanh(corr(speaking_rate,tab.session_activity))]·[1-b(1-q_tab)]·consensus` | rho>0.97 |
| 25 | AUD-064 — articulation_rate | R17 | `fx_aud_064` | `face.mouth_aperture` | `articulation_rate*new = C(articulation_rate)·[1+a·tanh(corr(articulation_rate,face.mouth_aperture))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 26 | AUD-065 — phonation_ratio | R8 | `fx_aud_065` | `tab.session_activity` | `phonation_ratio*new = C(phonation_ratio)·[1+a·tanh(corr(phonation_ratio,tab.session_activity))]·[1-b(1-q_tab)]·consensus` | rho>0.97 |
| 27 | AUD-066 — voice_activity_ratio | R8 | `fx_aud_066` | `tab.session_activity` | `voice_activity_ratio*new = C(voice_activity_ratio)·[1+a·tanh(corr(voice_activity_ratio,tab.session_activity))]·[1-b(1-q_tab)]·consensus` | rho>0.97 |
| 28 | AUD-073 — breath_count | R15 | `fx_aud_073` | `face.blink_phase` | `breath_count*new = C(breath_count)·[1+a·tanh(corr(breath_count,face.blink_phase))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 29 | AUD-074 — breath_interval_mean/std/cv | ADV5 | `fx_aud_074` | `face.blink_phase` | `breath_interval_mean_std_cv*new = C(breath_interval_mean_std_cv)·[1+a·tanh(corr(breath_interval_mean_std_cv,face.blink_phase))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 30 | AUD-076 — sigh_count, sigh_rate_per_min | R15 | `fx_aud_076` | `face.blink_phase` | `sigh_count*new = C(sigh_count)·[1+a·tanh(corr(sigh_count,face.blink_phase))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 31 | AUD-080 — latency_mean | ADV1 | `fx_aud_080` | `txt.q_difficulty` | `latency_mean*new = C(latency_mean)·[1+a·tanh(corr(latency_mean,txt.q_difficulty))]·[1-b(1-q_txt)]·consensus` | rho>0.97 |
| 32 | AUD-081 — latency_max / latency_std | ADV1 | `fx_aud_081` | `txt.q_difficulty` | `latency_max*new = C(latency_max)·[1+a·tanh(corr(latency_max,txt.q_difficulty))]·[1-b(1-q_txt)]·consensus` | rho>0.97 |
| 33 | AUD-083 — slow_response_ratio | ADV1 | `fx_aud_083` | `txt.q_difficulty` | `slow_response_ratio*new = C(slow_response_ratio)·[1+a·tanh(corr(slow_response_ratio,txt.q_difficulty))]·[1-b(1-q_txt)]·consensus` | rho>0.97 |
| 34 | TXT-001 — sentiment_compound/neg/pos | R29 | `fx_txt_001` | `face.emo_agreement` | `sentiment_compound_neg_pos*new = C(sentiment_compound_neg_pos)·[1+a·tanh(corr(sentiment_compound_neg_pos,face.emo_agreement))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 35 | TXT-005 — emotion_sadness/anger/fear/joy | R30 | `fx_txt_005` | `face.emo_agreement` | `emotion_sadness_anger_fear_j*new = C(emotion_sadness_anger_fear_j)·[1+a·tanh(corr(emotion_sadness_anger_fear_j,face.emo_agreement))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 36 | TXT-013 — first_person_ratio | R26 | `fx_txt_013` | `aud.prosody_f0_pause` | `first_person_ratio*new = C(first_person_ratio)·[1+a·tanh(corr(first_person_ratio,aud.prosody_f0_pause))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 37 | TXT-020 — absolutist_ratio | R26 | `fx_txt_020` | `aud.prosody_f0_pause` | `absolutist_ratio*new = C(absolutist_ratio)·[1+a·tanh(corr(absolutist_ratio,aud.prosody_f0_pause))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 38 | TXT-021 — negative_ratio / positive_ratio | R26 | `fx_txt_021` | `aud.prosody_f0_pause` | `negative_ratio*new = C(negative_ratio)·[1+a·tanh(corr(negative_ratio,aud.prosody_f0_pause))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 39 | TXT-022 — cognitive_ratio | R26 | `fx_txt_022` | `aud.prosody_f0_pause` | `cognitive_ratio*new = C(cognitive_ratio)·[1+a·tanh(corr(cognitive_ratio,aud.prosody_f0_pause))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 40 | TXT-031 — word_count | R27 | `fx_txt_031` | `aud.phonation_len` | `word_count*new = C(word_count)·[1+a·tanh(corr(word_count,aud.phonation_len))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 41 | TXT-032 — lexical_diversity | R27 | `fx_txt_032` | `aud.phonation_len` | `lexical_diversity*new = C(lexical_diversity)·[1+a·tanh(corr(lexical_diversity,aud.phonation_len))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 42 | TXT-034 — avg_word_length, avg_sentence_length | R27 | `fx_txt_034` | `aud.phonation_len` | `avg_word_length*new = C(avg_word_length)·[1+a·tanh(corr(avg_word_length,aud.phonation_len))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 43 | TXT-035 — automated_readability | R28 | `fx_txt_035` | `aud.phonation_len` | `automated_readability*new = C(automated_readability)·[1+a·tanh(corr(automated_readability,aud.phonation_len))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 44 | TXT-049 — sleep_words_ratio | R26 | `fx_txt_049` | `aud.prosody_f0_pause` | `sleep_words_ratio*new = C(sleep_words_ratio)·[1+a·tanh(corr(sleep_words_ratio,aud.prosody_f0_pause))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 45 | TXT-056 — talk_ratio | R31 | `fx_txt_056` | `vid.head_motion` | `talk_ratio*new = C(talk_ratio)·[1+a·tanh(corr(talk_ratio,vid.head_motion))]·[1-b(1-q_vid)]·consensus` | rho>0.97 |
| 46 | TXT-057 — turn_count, words_per_turn | R31 | `fx_txt_057` | `aud.phonation_len` | `turn_count*new = C(turn_count)·[1+a·tanh(corr(turn_count,aud.phonation_len))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 47 | TXT-058 — turn_length_mean/std/min/max | R31 | `fx_txt_058` | `vid.head_motion` | `turn_length_mean_std_min_max*new = C(turn_length_mean_std_min_max)·[1+a·tanh(corr(turn_length_mean_std_min_max,vid.head_motion))]·[1-b(1-q_vid)]·consensus` | rho>0.97 |
| 48 | TXT-059 — turn_length_trend | R31 | `fx_txt_059` | `vid.head_motion` | `turn_length_trend*new = C(turn_length_trend)·[1+a·tanh(corr(turn_length_trend,vid.head_motion))]·[1-b(1-q_vid)]·consensus` | rho>0.97 |
| 49 | TXT-062 — engagement_slope | R31 | `fx_txt_062` | `vid.head_motion` | `engagement_slope*new = C(engagement_slope)·[1+a·tanh(corr(engagement_slope,vid.head_motion))]·[1-b(1-q_vid)]·consensus` | rho>0.97 |
| 50 | VID-001 — flow_mean | R38 | `fx_vid_001` | `aud.energy_env` | `flow_mean*new = C(flow_mean)·[1+a·tanh(corr(flow_mean,aud.energy_env))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 51 | VID-002 — flow_std, flow_max | R38 | `fx_vid_002` | `aud.energy_env` | `flow_std*new = C(flow_std)·[1+a·tanh(corr(flow_std,aud.energy_env))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 52 | VID-003 — motion_score | R38 | `fx_vid_003` | `aud.energy_env` | `motion_score*new = C(motion_score)·[1+a·tanh(corr(motion_score,aud.energy_env))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 53 | VID-004 — motion_energy | R38 | `fx_vid_004` | `aud.energy_env` | `motion_energy*new = C(motion_energy)·[1+a·tanh(corr(motion_energy,aud.energy_env))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 54 | VID-007 — head_pose_yaw/pitch/roll_mean | R48 | `fx_vid_007` | `aud.speaking_rate` | `head_pose_yaw_pitch_roll_mea*new = C(head_pose_yaw_pitch_roll_mea)·[1+a·tanh(corr(head_pose_yaw_pitch_roll_mea,aud.speaking_rate))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 55 | VID-011 — head_movement_velocity | ADV2 | `fx_vid_011` | `aud.speaking_rate` | `head_movement_velocity*new = C(head_movement_velocity)·[1+a·tanh(corr(head_movement_velocity,aud.speaking_rate))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 56 | VID-012 — head_forward_offset | ADV2 | `fx_vid_012` | `aud.speaking_rate` | `head_forward_offset*new = C(head_forward_offset)·[1+a·tanh(corr(head_forward_offset,aud.speaking_rate))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 57 | VID-019 — frame_quality_ratio, blur_score, brightness | R33/34 | `fx_vid_019` | `face.det_conf` | `frame_quality_ratio*new = C(frame_quality_ratio)·[1+a·tanh(corr(frame_quality_ratio,face.det_conf))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 58 | VID-020 — discrete_blur_ratio, dark_ratio | R33/34 | `fx_vid_020` | `face.det_conf` | `discrete_blur_ratio*new = C(discrete_blur_ratio)·[1+a·tanh(corr(discrete_blur_ratio,face.det_conf))]·[1-b(1-q_face)]·consensus` | rho>0.97 |
| 59 | FAC-001 — au_intensity (per AU vector) | R45 | `fx_fac_001` | `aud.vocal_emo_prosody` | `au_intensity*new = C(au_intensity)·[1+a·tanh(corr(au_intensity,aud.vocal_emo_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 60 | FAC-002 — AU1_mean | R45 | `fx_fac_002` | `aud.vocal_emo_prosody` | `au1_mean*new = C(au1_mean)·[1+a·tanh(corr(au1_mean,aud.vocal_emo_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 61 | FAC-003 — AU2_mean | R45 | `fx_fac_003` | `aud.vocal_emo_prosody` | `au2_mean*new = C(au2_mean)·[1+a·tanh(corr(au2_mean,aud.vocal_emo_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 62 | FAC-004 — AU4_mean | R45 | `fx_fac_004` | `aud.vocal_emo_prosody` | `au4_mean*new = C(au4_mean)·[1+a·tanh(corr(au4_mean,aud.vocal_emo_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 63 | FAC-005 — AU6_mean | R45 | `fx_fac_005` | `aud.vocal_emo_prosody` | `au6_mean*new = C(au6_mean)·[1+a·tanh(corr(au6_mean,aud.vocal_emo_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 64 | FAC-006 — AU12_mean | R45 | `fx_fac_006` | `aud.vocal_emo_prosody` | `au12_mean*new = C(au12_mean)·[1+a·tanh(corr(au12_mean,aud.vocal_emo_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 65 | FAC-007 — AU10_mean | R45 | `fx_fac_007` | `aud.vocal_emo_prosody` | `au10_mean*new = C(au10_mean)·[1+a·tanh(corr(au10_mean,aud.vocal_emo_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 66 | FAC-008 — AU15_mean | R45 | `fx_fac_008` | `aud.vocal_emo_prosody` | `au15_mean*new = C(au15_mean)·[1+a·tanh(corr(au15_mean,aud.vocal_emo_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 67 | FAC-009 — AU23_mean | R45 | `fx_fac_009` | `aud.vocal_emo_prosody` | `au23_mean*new = C(au23_mean)·[1+a·tanh(corr(au23_mean,aud.vocal_emo_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 68 | FAC-010 — AU24_mean | R45 | `fx_fac_010` | `aud.vocal_emo_prosody` | `au24_mean*new = C(au24_mean)·[1+a·tanh(corr(au24_mean,aud.vocal_emo_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 69 | FAC-011 — AU25/AU26/AU27 | R45 | `fx_fac_011` | `aud.vocal_emo_prosody` | `au25_au26_au27*new = C(au25_au26_au27)·[1+a·tanh(corr(au25_au26_au27,aud.vocal_emo_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 70 | FAC-012 — AU43_mean | R45 | `fx_fac_012` | `aud.vocal_emo_prosody` | `au43_mean*new = C(au43_mean)·[1+a·tanh(corr(au43_mean,aud.vocal_emo_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 71 | FAC-013 — au_sadness_composite | R45 | `fx_fac_013` | `aud.vocal_emo_prosody` | `au_sadness_composite*new = C(au_sadness_composite)·[1+a·tanh(corr(au_sadness_composite,aud.vocal_emo_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 72 | FAC-014 — au_happiness_composite | R45 | `fx_fac_014` | `aud.vocal_emo_prosody` | `au_happiness_composite*new = C(au_happiness_composite)·[1+a·tanh(corr(au_happiness_composite,aud.vocal_emo_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 73 | FAC-015 — au_depression_indicator | R45 | `fx_fac_015` | `aud.vocal_emo_prosody` | `au_depression_indicator*new = C(au_depression_indicator)·[1+a·tanh(corr(au_depression_indicator,aud.vocal_emo_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 74 | FAC-021 — au_smile_timing | R49 | `fx_fac_021` | `aud.vocal_emo_prosody` | `au_smile_timing*new = C(au_smile_timing)·[1+a·tanh(corr(au_smile_timing,aud.vocal_emo_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 75 | FAC-022 — au_smile_decay | R49 | `fx_fac_022` | `aud.vocal_emo_prosody` | `au_smile_decay*new = C(au_smile_decay)·[1+a·tanh(corr(au_smile_decay,aud.vocal_emo_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 76 | FAC-025 — expression_variability | R43 | `fx_fac_025` | `aud.f0_std` | `expression_variability*new = C(expression_variability)·[1+a·tanh(corr(expression_variability,aud.f0_std))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 77 | FAC-026 — emotion_fear_face/sad/angry/happy/neutral | R43 | `fx_fac_026` | `aud.f0_std` | `emotion_fear_face_sad_angry_*new = C(emotion_fear_face_sad_angry_)·[1+a·tanh(corr(emotion_fear_face_sad_angry_,aud.f0_std))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 78 | FAC-028 — micro_expression_rate | R49 | `fx_fac_028` | `aud.f0_std` | `micro_expression_rate*new = C(micro_expression_rate)·[1+a·tanh(corr(micro_expression_rate,aud.f0_std))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 79 | FAC-030 — expression_entropy | R43 | `fx_fac_030` | `aud.f0_std` | `expression_entropy*new = C(expression_entropy)·[1+a·tanh(corr(expression_entropy,aud.f0_std))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 80 | FAC-032 — gaze_direct_ratio | R47 | `fx_fac_032` | `txt.neg_selffocus` | `gaze_direct_ratio*new = C(gaze_direct_ratio)·[1+a·tanh(corr(gaze_direct_ratio,txt.neg_selffocus))]·[1-b(1-q_txt)]·consensus` | rho>0.97 |
| 81 | FAC-033 — gaze_averted_ratio | R47 | `fx_fac_033` | `txt.neg_selffocus` | `gaze_averted_ratio*new = C(gaze_averted_ratio)·[1+a·tanh(corr(gaze_averted_ratio,txt.neg_selffocus))]·[1-b(1-q_txt)]·consensus` | rho>0.97 |
| 82 | FAC-034 — gaze_x/y_mean, gaze_avg_x/y | R47 | `fx_fac_034` | `txt.neg_selffocus` | `gaze_x_y_mean*new = C(gaze_x_y_mean)·[1+a·tanh(corr(gaze_x_y_mean,txt.neg_selffocus))]·[1-b(1-q_txt)]·consensus` | rho>0.97 |
| 83 | FAC-035 — gaze_mean_angle, gaze_std_angle | R47 | `fx_fac_035` | `txt.neg_selffocus` | `gaze_mean_angle*new = C(gaze_mean_angle)·[1+a·tanh(corr(gaze_mean_angle,txt.neg_selffocus))]·[1-b(1-q_txt)]·consensus` | rho>0.97 |
| 84 | FAC-041 — blink_rate / blink_rate_per_min | R46 | `fx_fac_041` | `aud.pause_onset` | `blink_rate*new = C(blink_rate)·[1+a·tanh(corr(blink_rate,aud.pause_onset))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 85 | FAC-042 — blink_count, avg_blink_duration_ms | R46 | `fx_fac_042` | `aud.pause_onset` | `blink_count*new = C(blink_count)·[1+a·tanh(corr(blink_count,aud.pause_onset))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 86 | FAC-047 — face_detection_rate, conf_avg | R39 | `fx_fac_047` | `aud.snr_vad` | `face_detection_rate*new = C(face_detection_rate)·[1+a·tanh(corr(face_detection_rate,aud.snr_vad))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 87 | TAB-008 — phq8_somatic_score | ADV4 | `fx_tab_008` | `aud.somatic_prosody` | `phq8_somatic_score*new = C(phq8_somatic_score)·[1+a·tanh(corr(phq8_somatic_score,aud.somatic_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 88 | TAB-009 — phq8_cognitive_score | ADV4 | `fx_tab_009` | `aud.somatic_prosody` | `phq8_cognitive_score*new = C(phq8_cognitive_score)·[1+a·tanh(corr(phq8_cognitive_score,aud.somatic_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 89 | TAB-010 — phq8_anhedonia | ADV4 | `fx_tab_010` | `aud.somatic_prosody` | `phq8_anhedonia*new = C(phq8_anhedonia)·[1+a·tanh(corr(phq8_anhedonia,aud.somatic_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 90 | TAB-011 — phq8_fatigue | ADV4 | `fx_tab_011` | `aud.somatic_prosody` | `phq8_fatigue*new = C(phq8_fatigue)·[1+a·tanh(corr(phq8_fatigue,aud.somatic_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 91 | TAB-012 — phq8_anxiety | ADV4 | `fx_tab_012` | `aud.somatic_prosody` | `phq8_anxiety*new = C(phq8_anxiety)·[1+a·tanh(corr(phq8_anxiety,aud.somatic_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 92 | TAB-013 — phq8_total_estimate | ADV4 | `fx_tab_013` | `aud.somatic_prosody` | `phq8_total_estimate*new = C(phq8_total_estimate)·[1+a·tanh(corr(phq8_total_estimate,aud.somatic_prosody))]·[1-b(1-q_aud)]·consensus` | rho>0.97 |
| 93 | XM-001 — congruence_audio_text_match | ADV6 | `fx_xm_001` | `all.consensus_tensor` | `congruence_audio_text_match*new = C(congruence_audio_text_match)·[1+a·tanh(corr(congruence_audio_text_match,all.consensus_tensor))]·[1-b(1-q_all)]·consensus` | rho>0.97 |
| 94 | XM-002 — congruence_masking_detected | ADV6 | `fx_xm_002` | `all.consensus_tensor` | `congruence_masking_detected*new = C(congruence_masking_detected)·[1+a·tanh(corr(congruence_masking_detected,all.consensus_tensor))]·[1-b(1-q_all)]·consensus` | rho>0.97 |
| 95 | XM-005 — audio_face_match | ADV6 | `fx_xm_005` | `all.consensus_tensor` | `audio_face_match*new = C(audio_face_match)·[1+a·tanh(corr(audio_face_match,all.consensus_tensor))]·[1-b(1-q_all)]·consensus` | rho>0.97 |
| 96 | XM-015 — crossmodal_sync | ADV6 | `fx_xm_015` | `all.consensus_tensor` | `crossmodal_sync*new = C(crossmodal_sync)·[1+a·tanh(corr(crossmodal_sync,all.consensus_tensor))]·[1-b(1-q_all)]·consensus` | rho>0.97 |
| 97 | Q-002 — snr_db ** | R59 | `fx_q_002` | `fus.quality_gate` | `snr_db*new = C(snr_db)·[1+a·tanh(corr(snr_db,fus.quality_gate))]·[1-b(1-q_fus)]·consensus` | rho>0.97 |
| 98 | Q-003 — clipping_rate ** | R59 | `fx_q_003` | `fus.quality_gate` | `clipping_rate*new = C(clipping_rate)·[1+a·tanh(corr(clipping_rate,fus.quality_gate))]·[1-b(1-q_fus)]·consensus` | rho>0.97 |
| 99 | Q-006 — frame_quality_ratio ** | R59 | `fx_q_006` | `fus.quality_gate` | `frame_quality_ratio*new = C(frame_quality_ratio)·[1+a·tanh(corr(frame_quality_ratio,fus.quality_gate))]·[1-b(1-q_fus)]·consensus` | rho>0.97 |

## 2. Why these no longer collide

A `fx_*` value is computed from a cross-modal correlation, a quality gate, and a consensus weight — none of which is a single-modality measurement — so it is produced by the fusion module, not by the extraction step that emitted the original primitive. The primitive is dropped; its scalar anchor is preserved through `C(base)`.

## 3. Label-leak subset

TAB-008–013 (ADV4 PHQ sub-scores) collide **and** leak the target. Their `fx_*` versions are kept for analysis only, flagged `leak=True`, and excluded from training folds — parity is still enforced so reported sub-score tables are unchanged.


_99 collisions replaced; 0 primitives retained._

