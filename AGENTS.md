# AGENTS.md — H5-OmniFusion (phase2)

## Notebook editing rule (HARD — user-mandated)
- **Append-only.** When editing ANY Jupyter notebook (`.ipynb`), only APPEND new
  cells. NEVER delete, replace, reorder, clear, or overwrite existing cells or
  their outputs. This applies to every notebook in this repo and to any new
  Colab notebook created for this project.

## Colab reprocessing rule (HARD — user-mandated)
- All dataset downloads, processing, and feature outputs must be **batched and
  streamed to Google Drive** as each batch completes. Nothing critical may live
  only on the Colab VM disk. A VM disconnect/restart must NEVER lose completed
  work.
- Use a Drive-side manifest plus atomic writes (temp file → rename) so any run
  resumes idempotently and skips already-saved batches.

## Colab MCP operating procedure (HARD — user-mandated)
- Open ONE Colab browser tab. Connect ONE time. Reuse that connection.
- Do not open a new connection for each command. Each new connection gives a
  fresh empty notebook and loses the prior cells and runtime state.
- Keep one persistent driver process alive. Send it commands over a local
  command queue. Read its results from a local result queue.
- Keep all logic in the uploaded code (`colab/run.py`). Do not spread logic
  across many notebook cells.
- Run the code through ONE reused cell. Add a new cell only to install a
  package when the code cannot do it itself.
- Use a special upload cell to push corrected code into the session after an
  error. Upload the source zip directly (base64 cell). Do not require a manual
  upload.
- Check the connect result. Abort when it returns false. Do not call notebook
  tools after a failed connect.
- The append-only rule still protects project notebooks and existing cell
  outputs. The scratch Colab session reuses its own run cell.

## Environment conventions
- Target runtime: Google Colab. T4 **GPU** for pretrained-encoder inference;
  **CPU** for download/decode, the Rust kernel, scalar feature extraction, and
  Drive I/O. The user switches runtime manually between phases.
- Drive base path: `/content/drive/MyDrive/DAIC-WOZ_Datasets`
- Pretrained models: `.../DAIC-WOZ_Datasets/pretrained_models` (load local-only).
- New reprocessing runs write to a NEW dated subfolder, never over old outputs.

## Feature schema
- Use the collision-free 200-feature schema (`FINAL_200_FEATURE_SCHEMA.md`);
  emit NaN + a reliability flag for absent modalities, never silent zeros.

## Documentation standard (HARD — user-mandated)
- Write every project document in **ASD-STE100 Issue 9 (Simplified Technical
  English)** style. Apply these writing rules:
  - Use short sentences. One topic per sentence. Keep procedures to 20 words
    or fewer per sentence; keep descriptive text to 25 words or fewer.
  - Use the active voice. Use the simple present tense where possible.
  - Use articles (a, an, the). Use one approved term for one concept. Define
    each term in a glossary before first use. Do not switch terms.
  - Keep noun clusters to three nouns or fewer.
  - Put a WARNING or CAUTION before the step it applies to.
  - Use vertical lists and tables for complex data. Keep paragraphs to six
    sentences or fewer.
  - Keep code identifiers, file paths, and technical names exactly as written.
    STE permits technical names. Do not translate them.
- Every audit, plan, or report must include: a hierarchical tree, real file
  counts and lines-of-code (LOC) tables, and at least one diagram or chart
  (mermaid or ASCII). Support the points with bullet lists.
- The code-audit reference is `OPERATIONAL_CODE_AUDIT_ASD-STE100.md`. It classifies
  each module as KEEP, DELETE, or FIX for feature extraction AND training.
- An audit only recommends. Do not delete or change code without the user's
  explicit confirmation. Deletion is a separate, confirmed action.