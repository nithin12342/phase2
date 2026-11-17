//! Preprocessing infrastructure. Adapters: CSV I/O + atomic Drive writes.
use h5_preprocess::SignalSet;
use h5_schema::FeatureValue;
use std::fs;
use std::io;
use std::path::Path;

/// Write feature rows as CSV: id,value,reliability. NaN stays "NaN".
pub fn write_features_csv(path: &Path, rows: &[(&str, FeatureValue)])
    -> io::Result<()> {
    let mut out = String::from("id,value,reliability\n");
    for (id, fv) in rows {
        out.push_str(&format!("{},{},{:?}\n", id, fv.value, fv.reliability));
    }
    atomic_write(path, out.as_bytes())
}

/// Write per-participant feature rows: participant,id,value,reliability.
pub fn write_participant_features_csv(
    path: &Path,
    rows: &[(String, &'static str, FeatureValue)],
) -> io::Result<()> {
    let mut out = String::from("participant,id,value,reliability\n");
    for (pid, id, fv) in rows {
        out.push_str(&format!("{},{},{},{:?}\n", pid, id, fv.value, fv.reliability));
    }
    atomic_write(path, out.as_bytes())
}

/// Atomic write: temp file then rename. A crash never leaves a partial file.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)
}

/// Read long-format signals: participant,signal,v0,v1,...
///
/// Returns one SignalSet per participant, in first-seen order. Unknown signals
/// are ignored. Unparseable values are skipped, never panicked on.
pub fn read_signals_csv(path: &Path) -> io::Result<Vec<(String, SignalSet)>> {
    let text = fs::read_to_string(path)?;
    let mut order: Vec<String> = Vec::new();
    let mut sets: Vec<SignalSet> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if i == 0 && cols.first() == Some(&"participant") {
            continue;
        }
        if cols.len() < 2 {
            continue;
        }
        let pid = cols[0].trim().to_string();
        let signal = cols[1].trim();
        let vals: Vec<f64> = cols[2..].iter()
            .filter_map(|v| v.trim().parse::<f64>().ok())
            .collect();
        let idx = match order.iter().position(|p| p == &pid) {
            Some(k) => k,
            None => {
                order.push(pid.clone());
                sets.push(SignalSet::default());
                sets.len() - 1
            }
        };
        let s = &mut sets[idx];
        match signal {
            "f0" => s.f0 = vals,
            "motion" => s.motion = vals,
            "au12" => s.au12 = vals,
            "unique" => s.unique = vals.first().copied(),
            "total" => s.total = vals.first().copied(),
            "first_ts" => s.first_ts = vals.first().copied(),
            "last_ts" => s.last_ts = vals.first().copied(),
            "speech_ratio" => s.speech_ratio = vals.first().copied(),
            _ => {}
        }
    }
    Ok(order.into_iter().zip(sets).collect())
}

/// Write one participant's derived f0 + motion traces in long format:
/// participant,signal,v0,v1,... Compatible with read_signals_csv.
pub fn write_derived_signals(
    path: &Path, pid: &str, f0: &[f64], motion: &[f64],
) -> io::Result<()> {
    let mut out = String::from("participant,signal,values\n");
    for (name, vals) in [("f0", f0), ("motion", motion)] {
        out.push_str(&format!("{},{}", pid, name));
        for v in vals {
            out.push_str(&format!(",{}", v));
        }
        out.push('\n');
    }
    atomic_write(path, out.as_bytes())
}

/// Derive f0 + motion for one participant from raw media and write the signals
/// CSV. The wav is 16-bit PCM; the raw stream is grayscale width*height planes.
pub fn derive_signals(
    wav: &Path, raw: &Path, width: usize, height: usize, pid: &str, out: &Path,
) -> io::Result<()> {
    let f0 = h5_preprocess::derive_f0(&fs::read(wav)?);
    let motion = h5_preprocess::derive_motion(&fs::read(raw)?, width, height);
    write_derived_signals(out, pid, &f0, &motion)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_roundtrip() {
        let p = std::env::temp_dir().join("h5_atomic_test.txt");
        atomic_write(&p, b"hello").unwrap();
        assert_eq!(fs::read(&p).unwrap(), b"hello");
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn read_signals_groups_by_participant() {
        let p = std::env::temp_dir().join("h5_signals_test.csv");
        let csv = "participant,signal,v0,v1\n\
                   300,f0,100,110\n\
                   300,total,10\n\
                   301,motion,0,1\n";
        fs::write(&p, csv).unwrap();
        let sets = read_signals_csv(&p).unwrap();
        assert_eq!(sets.len(), 2);
        assert_eq!(sets[0].0, "300");
        assert_eq!(sets[0].1.f0, vec![100.0, 110.0]);
        assert_eq!(sets[0].1.total, Some(10.0));
        assert_eq!(sets[1].1.motion, vec![0.0, 1.0]);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn derived_signals_roundtrip() {
        let p = std::env::temp_dir().join("h5_derived_test.csv");
        write_derived_signals(&p, "7", &[100.0, 110.0], &[0.0, 1.0]).unwrap();
        let sets = read_signals_csv(&p).unwrap();
        assert_eq!(sets.len(), 1);
        assert_eq!(sets[0].0, "7");
        assert_eq!(sets[0].1.f0, vec![100.0, 110.0]);
        assert_eq!(sets[0].1.motion, vec![0.0, 1.0]);
        let _ = fs::remove_file(&p);
    }
}