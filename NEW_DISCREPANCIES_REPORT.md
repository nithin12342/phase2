# Final Report on Newly Discovered Discrepancies (Chapter 1 to End)

Following a "recursive" line-by-line analysis of the report against the codebase, I have identified several **new** technical mistakes and conflicting information that were not covered in previous summaries.

---

## 1. Machine Learning Architecture & Components

### 1.1. Explainability (SHAP & LIME) Fabrication
*   **Report Claim:** Chapter 4 and several tables (Tables 23, 43) explicitly mention using **SHAP** and **LIME** for post-hoc model interpretation and explainability.
*   **Actual Code:** Neither `shap` nor `lime` is present in `requirements.txt`. There are no scripts or functions in the repository that implement these techniques.
*   **Discrepancy:** The explainability section is purely theoretical and does not reflect the actual software capabilities.

### 1.2. Mamba vs. Transformer Context
*   **Report Claim:** The report emphasizes a "Hybrid Fusion" strategy using **Cross-Modal Attention**.
*   **Actual Code:** The encoders (specifically `audio_encoder.py` and `text_encoder.py`) actually utilize **Mamba blocks** (`MambaEncoder`) for temporal modeling. While Mamba is advanced, it is a State Space Model (SSM), not a standard Transformer-based attention mechanism as emphasized in some parts of the report's fusion descriptions.
*   **Discrepancy:** The report inconsistent describes the backbone architecture, sometimes claiming standard transformers and other times SSMs, but the implementation is heavily Mamba-reliant.

### 1.3. Tabular Backbone (FT-Transformer vs. TabPFN/KAN)
*   **Report Claim:** Table 55 explicitly lists the Tabular backbone as **FT-Transformer**.
*   **Actual Code:** The tabular encoder (`tabular_encoder.py`) uses **Kolmogorov-Arnold Networks (KAN)** and references to **TabPFN**. There is no FT-Transformer implementation.
*   **Discrepancy:** The reported architecture for tabular data is completely different from the actual code.

---

## 2. Dataset & Scale Discrepancies

### 2.1. Dataset Size Hallucination
*   **Report Claim:** Section 1.3 and Chapter 3 claim the project handles over **one terabyte (1TB)** of raw multimodal data.
*   **Actual Code:** The DAIC-WOZ dataset (the primary source used) is roughly 30-50GB. The Azure infrastructure defined in `main.bicep` only provides **2GB of RAM** and small storage volumes, which would be physically unable to process or store a 1TB dataset for training.
*   **Discrepancy:** The data scale is exaggerated by approximately 20-30x.

### 2.2. Cross-Dataset Validation (EATD-Corpus)
*   **Report Claim:** The report implies the system is primarily an English-based system for DAIC-WOZ.
*   **Actual Code:** There is significant, almost parallel implementation for the **EATD-Corpus** (Mandarin Chinese) in `chinese_support.py` and `eatd_corpus_processor.py`.
*   **Discrepancy:** The report fails to highlight the significant Mandarin Chinese support and EATD-Corpus integration which is actually present in the code.

---

## 3. Advanced Feature & Logic Conflicts

### 3.1. "Smiling Depression" (Duchenne Marker) Narrative
*   **Report Claim:** Section 4.3 details a clinical case study of "Subject 300" using **Duchenne markers** (correlation between AU6 and AU12) to detect fake smiles.
*   **Actual Code:** While `advanced.py` implements a simple `CrossModalCongruence` score (comparing text sentiment vs audio valence), there is **zero code** that analyzes the correlation between specific Action Units (AUs) like AU6 and AU12 to detect Duchenne markers.
*   **Discrepancy:** The "clinical audit" narrative is a fabricated story added to the report without underlying technical implementation.

### 3.2. Advanced Innovations (ADV1-ADV9) implementation
*   **Report Claim:** The report describes ADV1-ADV9 as "Advanced Innovations" (Steps 100-108).
*   **Actual Code:** Many of these (like ADV1: Response Latency or ADV7: Temporal Trajectory) are implemented as simple **heuristics** (e.g., `slope = np.polyfit(...)`) rather than the "Deep Advanced AI" implied in the report.
*   **Discrepancy:** The report presents basic statistical features as breakthrough "AI Innovations."

---

## 4. Hyperparameter & Training Mismatches

### 4.1. Batch Size & Epochs
*   **Report Claim:** The model was trained with a **Batch Size of 32** for **300 epochs**.
*   **Actual Code:** `training_config.py` explicitly sets `batch_size: 8` and `n_epochs: 100`.
*   **Discrepancy:** The training parameters in the report are significantly different from the configuration used to generate the weights.

### 4.2. Learning Rate & Optimization
*   **Report Claim:** The learning rate was "optimized via Optuna" to **2.34e-4**.
*   **Actual Code:** `training_config.py` hardcodes the learning rate to **5e-5**.
*   **Discrepancy:** The reported learning rate is nearly 5x higher than the actual code, and the Optuna optimization is missing.

---

## 5. Clinical Metric Impossibilities

### 5.1. MAE / RMSE Paradox
*   **Report Claim:** Table 56 claims a **Mean Absolute Error (MAE) of 2.15** on the PHQ-8 scale.
*   **Actual Code:** The `CompositeLoss` and `PHQ8RegressionLoss` are implemented, but the codebase lacks an actual evaluation script that outputs these clinical metrics in a standardized way matching the table.
*   **Discrepancy:** The clinical metrics in the table appear to be "best-guess" or fabricated numbers rather than derived from a verifiable evaluation run.

---

**Summary:** 
The report continues to describe a "future-state" or "ideal" version of the project. The most egregious new findings are the fabrication of **Explainability (SHAP/LIME)**, the **1TB data scale** claim, the **FT-Transformer** backbone, and the **Smiling Depression/Duchenne** clinical narrative which has no code support.
