# Detailed Corrections for PHASE 2 REPORT.docx

This document provides a comprehensive analysis of errors, conflicts, and fabrications found in the `PHASE 2 REPORT.docx` (based on its 49-page equivalent `PHASE_2_REPORT.md`) after cross-referencing with the actual project codebase and directory structure.

---

## 1. Backend & Infrastructure Discrepancies

### 1.1. Database Engine Mismatch

* **Report Claim:** The system uses a **PostgreSQL** database for production-grade reliability and complex queries.
* **Actual Code:** The backend (`backend/main.py`, `backend/database.py`, and `backend/models.py`) is hardcoded to use **SQLite** (`sqlite:///./backend/database/app.db`).
* **Correction:** Update the report to reflect the use of SQLite as the primary database, or implement PostgreSQL as claimed.

### 1.2. Missing Asynchronous Task Queue (Celery/Redis)

* **Report Claim:** Heavy multimodal processing is handled by a background queue using **Celery** with **Redis** as a message broker to prevent HTTP timeouts.
* **Actual Code:** Neither Celery nor Redis is present in the `backend/requirements.txt` or `backend/main.py`. The API handles processing synchronously or via simple FastAPI dependency injection, without any external task worker.
* **Correction:** Remove references to Celery and Redis from the architectural diagrams and text.

### 1.3. Fabricated "Production-Grade" Infrastructure

* **Report Claim:** The system is deployed on a robust Azure architecture featuring PostgreSQL managed instances, Redis caches, and A100 GPUs for low-latency inference.
* **Actual Code:** The `infrastructure/main.bicep` file only defines an **Azure Container App** and a **Container Registry**. There are no Bicep resources for PostgreSQL, Redis, or specialized GPU nodes. The Container App is configured with only **1 CPU and 2GB RAM**.
* **Correction:** Align the infrastructure description with the actual Bicep configuration (Container Apps only).

---

## 2. Machine Learning & Model Architecture Conflicts

### 2.1. The "108-Step Pipeline" is Non-Functional

* **Report Claim:** A sophisticated 108-step pipeline (Production, Research, Innovation tiers) is used for exhaustive feature extraction (SNR, blur, optical flow, etc.).
* **Actual Code:** While a specification document exists (`docs/ml_pipeline/108_step_specification.md`), the actual implementation in `backend/models.py` uses a **"lite" mode**. It delegates feature extraction to the **HuggingFace Inference API** and runs a single local fusion model. Most "advanced" steps (like SNR or blur gating) are missing from the inference flow or exist only as mocked scripts with hardcoded values.
* **Correction:** Clarify that the 108-step pipeline is a design specification, not fully implemented in the current production backend.

### 2.2. Fabricated Loss Functions

* **Report Claim:** The model uses **Huber Loss** for PHQ-8 score regression to handle outliers.
* **Actual Code:** A global search of the repository shows **zero instances** of Huber Loss. The model training scripts and heads use standard MSE or Binary Cross Entropy.
* **Correction:** Remove mention of Huber Loss from the mathematical methodology section.

### 2.3. Data Augmentation Hallucinations

* **Report Claim:** The model benefits from **SpecAugment** (audio) and **French Back-Translation** (text).
* **Actual Code:** There is no code or dependency for translation APIs or SpecAugment in the repository.
* **Correction:** Remove these claims from the Data Preprocessing and Augmentation sections.

### 2.4. Tabular Modality is a Mock

* **Report Claim:** A sophisticated **FT-Transformer** processes tabular clinical data, contributing 7% to the model's accuracy.
* **Actual Code:** In `backend/models.py`, the tabular embedding function is literally: `return np.zeros(768, dtype=np.float32), None`. It returns an empty vector and has no impact on the prediction.
* **Correction:** Correct the report to state that tabular features are currently placeholders.

---

## 3. Frontend & Visualization Inaccuracies

### 3.1. Missing Visualization Libraries

* **Report Claim:** The dashboard uses **Recharts** and **D3.js** to generate interactive radar charts and emotion trajectory graphs.
* **Actual Code:** `frontend/package.json` contains no dependencies for `recharts` or `d3`. The `frontend/src/App.js` code simply displays prediction results as raw text in a box.
* **Correction:** Remove the descriptions of interactive charts and trajectory graphs until they are implemented.

### 3.2. Missing MediaRecorder Integration

* **Report Claim:** The frontend captures real-time audio and video using the browser's **MediaRecorder API**.
* **Actual Code:** The frontend only provides standard file input fields (`<input type="file">`) for uploading pre-recorded files. There is no logic for live recording.
* **Correction:** Update the UI section to reflect that it is an upload-based system, not a live capture system.

---

## 4. Mathematical & Logical Inconsistencies

### 4.1. Impossible Confusion Matrix Samples

* **Report Claim:** The confusion matrix (Section 4.9) shows a total of **931 validation samples**.
* **The Conflict:** The DAIC-WOZ dataset used has only **189 participants**. Even with cross-validation, the sample counts reported are mathematically impossible for this dataset.
* **Correction:** Recalculate or provide the correct sample counts based on the 189-subject DAIC-WOZ dataset.

### 4.2. AUC-ROC Value Contradictions

* **Report Claim:** The report lists four different values for AUC-ROC: **0.87, 0.89, 0.8145, and 0.8445** across different sections.
* **Actual Code:** The evaluation logs (where available) show values closer to **0.8013**.
* **Correction:** Standardize the performance metrics across all sections of the report to match the actual experimental results.

### 4.3. Latency vs. Resource Paradox

* **Report Claim:** The system processes a 1-minute multimodal session in **2.5 seconds** and supports **50 concurrent sessions** on an A100.
* **The Conflict:** The actual deployment uses **1 CPU and 2GB RAM** (no GPU). Running VideoMAE, Wav2Vec2, and RoBERTa on such hardware for a 1-minute video would take minutes, not seconds, and would certainly not support 50 concurrent users.
* **Correction:** Update the performance and scalability claims to match the actual hardware constraints (Azure Container Apps).

---

## 5. Directory Structure & File References

* **Conflict:** Several sections refer to a `Table 2.1` or `Figure 3.4` inline, but the actual tables/figures are often appended at the very end of the document (Tables 1-59), breaking the flow and cross-referencing.
* **Correction:** Move relevant tables and figures inline to the sections where they are discussed.

---

**Summary:** The report describes an ideal "Champion" version of the project that significantly exceeds the current implementation's capabilities. To ensure technical integrity, the report must be revised to reflect the **"Lite" production environment** actually present in the codebase.
