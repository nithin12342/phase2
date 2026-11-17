"""Generate the higher-order (HO) cross-modal feature block.

For every modality combination of size k in {2,3,4,5}, emit two features:
  -C congruence: geometric mean of depression evidence x quality consensus.
  -D dispersion: std of evidence x consensus (cross-modal disagreement/masking).

Modalities (canonical order): A=audio, T=text, V=video, F=face, B=tabular.
52 features total -> schema grows 200 -> 252 (156 primitive + 96 fusion).

Patches, idempotently:
  FINAL_200_FEATURE_SCHEMA.md        (append a new section, original 200 intact)
  rust/.../h5-schema/src/catalog.rs  (insert entries + update totals)
  preprocessing/schema_catalog.py    (insert entries + update total)
  rust/.../h5-preprocess/src/ho_combos.rs  (regenerate combo table)

Run: python tools/gen_ho_features.py
"""
import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LETTERS = "ATVFB"
NAMES = {"A": "audio", "T": "text", "V": "video", "F": "face", "B": "tabular"}
MODALITY_IDX = {c: i for i, c in enumerate(LETTERS)}


def combos():
    """Yield (k, letters, indices) for every combination, k=2..5."""
    for k in (2, 3, 4, 5):
        for idx in itertools.combinations(range(5), k):
            letters = "".join(LETTERS[i] for i in idx)
            yield k, letters, idx


def features():
    """Yield (id, name, compute, rationale, letters, idx, kind_tag)."""
    for k, letters, idx in combos():
        pretty = "×".join(NAMES[c] for c in letters)
        yield (f"HO{k}-{letters}-C", f"{pretty} congruence",
               "geomean(evidence)·consensus(quality)",
               f"{k}-way convergence of depression evidence", letters, idx, "C")
        yield (f"HO{k}-{letters}-D", f"{pretty} dispersion",
               "std(evidence)·consensus(quality)",
               f"{k}-way cross-modal disagreement (masking)", letters, idx, "D")


def patch_schema():
    path = ROOT / "FINAL_200_FEATURE_SCHEMA.md"
    text = path.read_text(encoding="utf-8")
    if "HO2-AT-C" in text:
        print("schema: already extended, skip")
        return
    rows = list(features())
    lines = [
        "",
        "## Extension — Higher-Order Cross-Modal Interactions (52 features)",
        "",
        "Appended 2026-11-11. The original 200 rows above are unchanged.",
        "New total = 252 (156 primitive + 96 fusion). Each modality combination",
        "of size 2/3/4/5 gets a congruence (-C) and a dispersion (-D) feature.",
        "Evidence is a per-modality depression proxy; quality gates the result.",
        "Reliability is Proxy when computed, Absent when a modality is missing.",
        "",
        "| # | Feature ID | Name | Modality | Kind | Compute | Value / rationale |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, (fid, name, compute, rationale, *_rest) in enumerate(rows, start=201):
        lines.append(
            f"| {n} | {fid} | {name} | cross-modal | fusion | `{compute}` | {rationale} |")
    lines += ["", "_Extension counter: 52 fusion (20×2-way + 20×3-way + 10×4-way + 2×5-way)._", ""]
    path.write_text(text + "\n".join(lines), encoding="utf-8")
    print(f"schema: appended {len(rows)} rows")


def patch_rust_catalog():
    path = ROOT / "rust/crates/h5-schema/src/catalog.rs"
    text = path.read_text(encoding="utf-8")
    if "HO2-AT-C" in text:
        print("rust catalog: already extended, skip")
        return
    entries = "".join(
        f'    CatalogEntry {{ id: "{fid}", modality: Modality::CrossModal, kind: Kind::Fusion }},\n'
        for fid, *_ in features())
    marker = "    CatalogEntry { id: \"fx_q_006\", modality: Modality::Quality, kind: Kind::Fusion },\n];"
    assert marker in text, "rust catalog closing marker not found"
    text = text.replace(
        marker,
        "    CatalogEntry { id: \"fx_q_006\", modality: Modality::Quality, kind: Kind::Fusion },\n"
        + entries + "];")
    text = text.replace("assert_eq!(count(), 200);", "assert_eq!(count(), 252);")
    text = text.replace("assert_eq!(fusion_count(), 44);", "assert_eq!(fusion_count(), 96);")
    path.write_text(text, encoding="utf-8")
    print("rust catalog: inserted 52 entries, totals 252/156/96")


def patch_python_catalog():
    path = ROOT / "preprocessing/schema_catalog.py"
    text = path.read_text(encoding="utf-8")
    if "HO2-AT-C" in text:
        print("python catalog: already extended, skip")
        return
    entries = "".join(f'    ("{fid}", "cross-modal", "fusion"),\n' for fid, *_ in features())
    marker = '    ("fx_q_006", "quality", "fusion"),\n]'
    assert marker in text, "python catalog closing marker not found"
    text = text.replace(marker, '    ("fx_q_006", "quality", "fusion"),\n' + entries + "]")
    text = text.replace("assert len(CATALOG) == 200, len(CATALOG)",
                        "assert len(CATALOG) == 252, len(CATALOG)")
    path.write_text(text, encoding="utf-8")
    print("python catalog: inserted 52 entries, total 252")


def write_combo_table():
    path = ROOT / "rust/crates/preprocess/h5-preprocess/src/ho_combos.rs"
    lines = [
        "//! Higher-order modality combinations. Generated by tools/gen_ho_features.py.",
        "/// Modality indices: 0=audio, 1=text, 2=video, 3=face, 4=tabular.",
        "pub struct HoCombo {",
        "    pub cong: &'static str,",
        "    pub disp: &'static str,",
        "    pub mods: &'static [usize],",
        "}",
        "",
        "/// All 52 higher-order combinations (2/3/4/5-way), each with -C and -D.",
        "pub const COMBOS: &[HoCombo] = &[",
    ]
    for k, letters, idx in combos():
        mods = ", ".join(str(i) for i in idx)
        lines.append(
            f'    HoCombo {{ cong: "HO{k}-{letters}-C", disp: "HO{k}-{letters}-D", mods: &[{mods}] }},')
    lines += ["];", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"combo table: wrote {sum(1 for _ in combos())} combos -> {path.name}")


def main():
    n = sum(1 for _ in features())
    assert n == 52, n
    patch_schema()
    patch_rust_catalog()
    patch_python_catalog()
    write_combo_table()
    print(f"done: {n} higher-order features; schema total 252")


if __name__ == "__main__":
    main()