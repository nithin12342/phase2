//! Preprocessing interface layer. Thin CLI over the application use cases.
//!
//! Modes:
//!   --demo <out.csv>                 Run a built-in demo participant.
//!   --features <input.csv> <out.csv> Read long-format signals, emit features.
//!   --derive <wav> <raw> <w> <h> <pid> <out.csv>
//!                                    Derive f0 + motion from raw media.
use h5_preprocess::ParticipantInput;
use h5_schema::FeatureValue;
use std::path::Path;

struct DemoInput;
impl ParticipantInput for DemoInput {
    fn f0_frames(&self) -> Vec<f64> { vec![100.0, 110.0, 90.0, 105.0] }
    fn motion(&self) -> Vec<f64> { vec![0.0, 1.0, 0.2, 0.0] }
}

fn run_demo(out: &str) {
    let input = DemoInput;
    let rows: Vec<(&str, FeatureValue)> = vec![
        ("AUD-005", h5_preprocess::extract_f0_cv(&input)),
        ("VID-005", h5_preprocess::extract_stillness(&input)),
    ];
    match h5_preprocess_infra::write_features_csv(Path::new(out), &rows) {
        Ok(()) => println!("wrote {}", out),
        Err(e) => eprintln!("write failed: {}", e),
    }
}

fn run_features(input: &str, out: &str) {
    let sets = match h5_preprocess_infra::read_signals_csv(Path::new(input)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("read failed: {}", e);
            std::process::exit(1);
        }
    };
    let mut rows: Vec<(String, &'static str, FeatureValue)> = Vec::new();
    for (pid, signals) in &sets {
        for (id, fv) in h5_preprocess::extract_all(signals) {
            rows.push((pid.clone(), id, fv));
        }
    }
    match h5_preprocess_infra::write_participant_features_csv(Path::new(out), &rows) {
        Ok(()) => println!("wrote {} ({} participants, {} rows)",
                           out, sets.len(), rows.len()),
        Err(e) => {
            eprintln!("write failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_derive(wav: &str, raw: &str, w: usize, h: usize, pid: &str, out: &str) {
    match h5_preprocess_infra::derive_signals(
        Path::new(wav), Path::new(raw), w, h, pid, Path::new(out),
    ) {
        Ok(()) => println!("derived {} -> {}", pid, out),
        Err(e) => {
            eprintln!("derive failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("--demo") => {
            let out = args.get(2).cloned().unwrap_or_else(|| "features.csv".into());
            run_demo(&out);
        }
        Some("--features") => {
            let input = args.get(2).cloned().unwrap_or_else(|| "signals.csv".into());
            let out = args.get(3).cloned().unwrap_or_else(|| "features.csv".into());
            run_features(&input, &out);
        }
        Some("--derive") => {
            let wav = args.get(2).cloned().unwrap_or_default();
            let raw = args.get(3).cloned().unwrap_or_default();
            let w = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
            let h = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(64);
            let pid = args.get(6).cloned().unwrap_or_else(|| "0".into());
            let out = args.get(7).cloned().unwrap_or_else(|| "derived.csv".into());
            run_derive(&wav, &raw, w, h, &pid, &out);
        }
        _ => {
            println!("h5-preprocess-cli --demo <out.csv>");
            println!("h5-preprocess-cli --features <input.csv> <out.csv>");
            println!("h5-preprocess-cli --derive <wav> <raw> <w> <h> <pid> <out.csv>");
        }
    }
}