//! Preprocessing application layer. Use cases that orchestrate the domains.
use h5_schema::{FeatureValue, Reliability};

pub mod ho_combos;

/// Use case: derive the voiced F0 contour (Hz) from raw WAV bytes. Moves the
/// pitch DSP off Python/librosa into the std-only Rust kernel.
pub fn derive_f0(wav_bytes: &[u8]) -> Vec<f64> {
    let (sr, samples) = h5_audio::pitch::parse_wav(wav_bytes);
    h5_audio::pitch::f0_contour(&samples, sr)
}

/// Use case: derive the motion trace from a raw grayscale plane stream. Moves
/// the frame-diff off Python/cv2 into the std-only Rust kernel.
pub fn derive_motion(raw: &[u8], width: usize, height: usize) -> Vec<f64> {
    h5_video::motion::motion_from_raw(raw, width, height)
}

/// Port: raw per-participant inputs supplied by an infrastructure adapter.
pub trait ParticipantInput {
    fn f0_frames(&self) -> Vec<f64>;
    fn motion(&self) -> Vec<f64>;
}

/// Use case: extract the audio F0 coefficient of variation for one participant.
pub fn extract_f0_cv(input: &dyn ParticipantInput) -> FeatureValue {
    h5_audio::f0_cv(&input.f0_frames())
}

/// Use case: extract the video stillness ratio for one participant.
pub fn extract_stillness(input: &dyn ParticipantInput) -> FeatureValue {
    h5_video::stillness_ratio(&input.motion(), 0.5)
}

/// The decoded per-participant signals an adapter can supply. Empty trace or
/// `None` scalar means the signal is absent; the feature emits NaN + Absent.
#[derive(Debug, Default, Clone)]
pub struct SignalSet {
    pub f0: Vec<f64>,
    pub motion: Vec<f64>,
    pub au12: Vec<f64>,
    pub unique: Option<f64>,
    pub total: Option<f64>,
    pub first_ts: Option<f64>,
    pub last_ts: Option<f64>,
    pub speech_ratio: Option<f64>,
}

/// Use case: compute every currently-implemented feature from a signal set.
///
/// Returns (feature_id, value) pairs. Unimplemented features are not returned;
/// the assembler fills them from the 200-catalog as NaN + Absent.
pub fn extract_from_signals(s: &SignalSet) -> Vec<(&'static str, FeatureValue)> {
    let mut out = Vec::new();
    out.push(("AUD-005", h5_audio::f0_cv(&s.f0)));
    out.push(("VID-005", h5_video::stillness_ratio(&s.motion, 0.5)));
    out.push(("FAC-018", h5_face::au_active_ratio(&s.au12)));
    let diversity = match (s.unique, s.total) {
        (Some(u), Some(t)) if t > 0.0 => {
            h5_text::lexical_diversity(u as usize, t as usize)
        }
        _ => FeatureValue::absent(),
    };
    out.push(("TXT-033", diversity));
    let duration = match (s.first_ts, s.last_ts) {
        (Some(a), Some(b)) => h5_tabular::interview_duration(a, b),
        _ => FeatureValue::absent(),
    };
    out.push(("TAB-001", duration));
    out
}

/// Reliability helper re-exported for adapters.
pub fn absent() -> FeatureValue {
    FeatureValue::new(f64::NAN, Reliability::Absent)
}

/// Map a reliability flag to a quality weight for the consensus gate.
fn rel_q(r: Reliability) -> f64 {
    match r {
        Reliability::Exact => 1.0,
        Reliability::Proxy => 0.6,
        Reliability::Absent => f64::NAN,
    }
}

/// Tag a higher-order result: Proxy when computed, Absent when NaN.
fn ho_value(v: f64) -> FeatureValue {
    if v.is_nan() {
        FeatureValue::absent()
    } else {
        FeatureValue::new(v, Reliability::Proxy)
    }
}

/// Per-modality depression evidence in [0,1], order A,T,V,F,B (indices 0..4).
///
/// Each is a documented proxy transform of one primitive feature. A missing
/// modality stays NaN, so any combination that needs it emits Absent.
fn modality_evidence(
    s: &SignalSet,
    prims: &[(&'static str, FeatureValue)],
) -> ([f64; 5], [f64; 5]) {
    let lookup = |id: &str| prims.iter().find(|(fid, _)| *fid == id).map(|(_, v)| *v);
    let mut evidence = [f64::NAN; 5];
    let mut quality = [f64::NAN; 5];
    let mut set = |i: usize, fv: FeatureValue, transform: fn(f64) -> f64| {
        if !fv.is_absent() {
            evidence[i] = transform(fv.value).clamp(0.0, 1.0);
            quality[i] = rel_q(fv.reliability);
        }
    };
    if let Some(fv) = lookup("AUD-005") { set(0, fv, |cv| 1.0 / (1.0 + cv)); }
    if let Some(fv) = lookup("TXT-033") { set(1, fv, |d| 1.0 - d); }
    if let Some(fv) = lookup("VID-005") { set(2, fv, |st| st); }
    if let Some(fv) = lookup("FAC-018") { set(3, fv, |a| 1.0 - a); }
    if let Some(sr) = s.speech_ratio {
        evidence[4] = (1.0 - sr).clamp(0.0, 1.0);
        quality[4] = 1.0;
    }
    (evidence, quality)
}

/// Compute all 52 higher-order cross-modal features from the primitives.
pub fn higher_order_features(
    s: &SignalSet,
    prims: &[(&'static str, FeatureValue)],
) -> Vec<(&'static str, FeatureValue)> {
    let (evidence, quality) = modality_evidence(s, prims);
    let mut out = Vec::new();
    for combo in ho_combos::COMBOS {
        let ev: Vec<f64> = combo.mods.iter().map(|&i| evidence[i]).collect();
        let q: Vec<f64> = combo.mods.iter().map(|&i| quality[i]).collect();
        out.push((combo.cong, ho_value(h5_kernel::ho::congruence(&ev, &q))));
        out.push((combo.disp, ho_value(h5_kernel::ho::dispersion(&ev, &q))));
    }
    out
}

/// Use case: primitives plus all higher-order interactions for one participant.
pub fn extract_all(s: &SignalSet) -> Vec<(&'static str, FeatureValue)> {
    let prims = extract_from_signals(s);
    let ho = higher_order_features(s, &prims);
    let mut out = prims;
    out.extend(ho);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Demo;
    impl ParticipantInput for Demo {
        fn f0_frames(&self) -> Vec<f64> { vec![100.0, 110.0, 90.0] }
        fn motion(&self) -> Vec<f64> { vec![0.0, 1.0] }
    }

    #[test]
    fn use_case_runs() {
        assert!(extract_f0_cv(&Demo).value > 0.0);
    }

    #[test]
    fn signals_full() {
        let s = SignalSet {
            f0: vec![100.0, 110.0, 90.0],
            motion: vec![0.0, 1.0, 0.0, 1.0],
            au12: vec![0.0, 0.1, 0.2, 0.0],
            unique: Some(5.0),
            total: Some(10.0),
            first_ts: Some(10.0),
            last_ts: Some(70.0),
            speech_ratio: Some(0.5),
        };
        let rows = extract_from_signals(&s);
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[2].1.value, 0.5); // FAC-018
        assert_eq!(rows[3].1.value, 0.5); // TXT-033
        assert_eq!(rows[4].1.value, 60.0); // TAB-001
    }

    #[test]
    fn signals_empty_are_absent() {
        let rows = extract_from_signals(&SignalSet::default());
        assert!(rows.iter().all(|(_, v)| v.is_absent()));
    }

    fn full_signals() -> SignalSet {
        SignalSet {
            f0: vec![100.0, 110.0, 90.0],
            motion: vec![0.0, 1.0, 0.0, 1.0],
            au12: vec![0.0, 0.1, 0.2, 0.0],
            unique: Some(5.0),
            total: Some(10.0),
            first_ts: Some(10.0),
            last_ts: Some(70.0),
            speech_ratio: Some(0.5),
        }
    }

    #[test]
    fn extract_all_emits_57() {
        assert_eq!(extract_all(&full_signals()).len(), 5 + 52);
    }

    #[test]
    fn five_way_present_is_proxy() {
        let rows = extract_all(&full_signals());
        let cong = rows.iter().find(|(id, _)| *id == "HO5-ATVFB-C").unwrap();
        assert_eq!(cong.1.reliability, Reliability::Proxy);
        assert!(cong.1.value > 0.0);
        let disp = rows.iter().find(|(id, _)| *id == "HO5-ATVFB-D").unwrap();
        assert!(disp.1.value > 0.0);
    }

    #[test]
    fn five_way_absent_without_tabular() {
        let mut s = full_signals();
        s.speech_ratio = None;
        let rows = extract_all(&s);
        let cong = rows.iter().find(|(id, _)| *id == "HO5-ATVFB-C").unwrap();
        assert!(cong.1.is_absent());
    }

    #[test]
    fn empty_signals_all_absent() {
        let rows = extract_all(&SignalSet::default());
        assert_eq!(rows.len(), 57);
        assert!(rows.iter().all(|(_, v)| v.is_absent()));
    }
}