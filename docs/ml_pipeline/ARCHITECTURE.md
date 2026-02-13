# H⁵-OmniFusion: SOTA Multimodal Architecture

## Architecture Overview

### System Design Philosophy
The H⁵-OmniFusion architecture is designed to address the "black box" problem in clinical AI while maximizing predictive performance. It adopts a **Hybrid Neuro-Symbolic** approach where deep learning components (Wav2Vec2, RoBERTa) provide rich feature representations, while structured fusion mechanisms (Hypergraph, MoE) provide interpretability and control.
*   **Modularity**: Each modality is processed by a dedicated "Expert" branch, allowing for independent upgrades (e.g., swapping OpenFace for py-feat).
*   **Resilience**: The MoE gating mechanism ensures the system functions even if one modality fails (e.g., corrupted video), by dynamically re-weighting the remaining available experts.
*   **Traceability**: Every intermediate tensor—from raw input to final probability—is logged and auditable, complying with the "Traceability Contract" for clinical software.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         H⁵-OMNIFUSION ARCHITECTURE                          │
│                    Depression Detection (DAIC-WOZ Dataset)                  │
└─────────────────────────────────────────────────────────────────────────────┘

                              RAW MULTIMODAL INPUT
    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │  AUDIO   │ │   TEXT   │ │  VIDEO   │ │   FACE   │ │ TABULAR  │
    │  (.wav)  │ │(transcript)│ │  (.mp4)  │ │ (frames) │ │(metadata)│
    └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
         │            │            │            │            │
         ▼            ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: SOTA FEATURE EXTRACTION                     │
├────────────────┬────────────────┬────────────────┬──────────────┬───────────┤
│ Wav2Vec2-Large │  MentalRoBERTa │   VideoMAE     │OpenFace 2.0  │  FT-Transformer│
│  + eGeMAPSS    │     (768d)     │    (768d)      │ + POSTER v2  │  (768d)   │
│    (768d)      │                │                │   (768d)     │           │
│  xlsr-53 +     │ mental-roberta │ videomae-base  │ AU + Spatial │ Pretrained│
│  opensmile     │     -base      │                │  Features    │ Tabular   │
│  Acc: 0.84     │  Acc: 0.91     │ Acc: 0.76-0.79 │  Acc: 0.79   │ Acc: 0.85 │
├────────────────┴────────────────┴────────────────┴──────────────┴───────────┤
│                           Output: 768-dim per modality                      │
└────────────────────────────────────────────────────────────────────────────┘
         │            │            │            │            │
         ▼            ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   STAGE 2: MODALITY-SPECIFIC ENCODERS                       │
├──────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐│
│  │AudioEncoder │ │ TextEncoder │ │VideoEncoder │ │ FaceEncoder │ │TabEncoder│
│  │  BiLSTM +   │ │  Temporal   │ │  BiLSTM +   │ │   BiLSTM    │ │   MLP   ││
│  │  Attention  │ │  Pooling    │ │  Attention  │ │  + Quality  │ │ + Proj  ││
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └────┬────┘│
│         │               │               │               │              │     │
│         └───────────────┴───────────────┼───────────────┴──────────────┘     │
│                                         ▼                                    │
│                              [5 × 768-dim Embeddings]                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   STAGE 3: THREE-LEVEL HYPERGRAPH FUSION                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │         LOCAL HYPERGRAPH (Intra-Modality Relationships)               │  │
│  │    ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐                   │  │
│  │    │Audio│◄─►│Text │◄─►│Video│◄─►│Face │◄─►│ Tab │                   │  │
│  │    └─────┘   └─────┘   └─────┘   └─────┘   └─────┘                   │  │
│  │    Message passing within local temporal windows                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                   │                                         │
│                                   ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │       MODALITY HYPERGRAPH (Cross-Modal Attention Fusion)              │  │
│  │                                                                       │  │
│  │              Audio ◄────────────────────────► Text                    │  │
│  │                 ▲                                ▲                    │  │
│  │                 │      ┌──────────────┐          │                    │  │
│  │                 └──────┤ HYPEREDGE    ├──────────┘                    │  │
│  │                        │ (All 5 mod)  │                               │  │
│  │                 ┌──────┤              ├──────┐                        │  │
│  │                 │      └──────────────┘      │                        │  │
│  │                 ▼                            ▼                        │  │
│  │              Video ◄────────────────────► Face                        │  │
│  │                              ▲                                        │  │
│  │                              │                                        │  │
│  │                           Tabular                                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                   │                                         │
│                                   ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │         LATENT PERCEIVER (Global Context Aggregation)                 │  │
│  │    ┌─────────────────────────────────────────────────────────────┐    │  │
│  │    │                  Learned Latent Queries                      │    │  │
│  │    │                        [64 × 768]                            │    │  │
│  │    │                            ▼                                 │    │  │
│  │    │    ┌─────────────────────────────────────────────────────┐   │    │  │
│  │    │    │        Cross-Attention with All Modalities          │   │    │  │
│  │    │    │           (Perceiver-style bottleneck)              │   │    │  │
│  │    │    └─────────────────────────────────────────────────────┘   │    │  │
│  │    └─────────────────────────────────────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 4: MIXTURE OF EXPERTS (MoE)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Quality-Aware Gating Network                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     Confidence Scores per Modality                  │   │
│   │     Audio: 0.84    Text: 0.91    Video: 0.78    Face: 0.79         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                   │                                         │
│                                   ▼                                         │
│   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │
│   │Expert  │ │Expert  │ │Expert  │ │Expert  │ │Expert  │ │Expert  │        │
│   │ Audio  │ │  Text  │ │ Video  │ │  Face  │ │Tabular │ │ Fusion │        │
│   │ (FFN)  │ │ (FFN)  │ │ (FFN)  │ │ (FFN)  │ │ (FFN)  │ │ (FFN)  │        │
│   └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘        │
│       │          │          │          │          │          │              │
│       └──────────┴──────────┴────┬─────┴──────────┴──────────┘              │
│                                  ▼                                          │
│                     Weighted Combination (Top-K=3)                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 5: OUTPUT HEADS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────┐   ┌─────────────────────────────┐        │
│   │    Binary Classification    │   │    PHQ-8 Score Regression   │        │
│   │    (Depressed/Not)          │   │    (0-24 continuous)        │        │
│   │                             │   │                             │        │
│   │    MLP → Sigmoid            │   │    MLP → Linear             │        │
│   │                             │   │                             │        │
│   │    Target: F1 > 0.75        │   │    Target: MAE < 3.0        │        │
│   │    AUC > 0.85               │   │                             │        │
│   └─────────────────────────────┘   └─────────────────────────────┘        │
│                                                                             │
│                        Combined Multi-Task Loss                             │
│             L = λ₁·BCEFocal + λ₂·MSE + λ₃·OrthogonalReg                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


## Deep Dive: Component Specifications

### 1. Stage 3: Three-Level Hypergraph Fusion Network
The Hypergraph Fusion Network (HFN) is the structural core of the model, replacing standard concatenation or cross-attention. It models relationships as **hyperedges** that can connect any number of nodes, not just pairs.
*   **Level 1: Local Hypergraph (Intra-Modality)**
    *   **Nodes**: Temporal segments of a single modality (e.g., Audio_t1, Audio_t2, Audio_t3).
    *   **Hyperedges**: Sliding windows connecting adjacent segments.
    *   **Function**: Captures short-term temporal dependencies and smooths noise within a modality before fusion.
*   **Level 2: Modality Hypergraph (Inter-Modality)**
    *   **Nodes**: The aggregated representations of each modality (Audio, Text, Video, Face, Tabular).
    *   **Hyperedges**: 
        *   *Semantic Edge*: Connects {Audio, Text} (Speech content + tone).
        *   *Visual Edge*: Connects {Video, Face} (Body language + Expression).
        *   *Global Edge*: Connects {All Modalities}.
    *   **Function**: Allows specific subgroups of modalities to interact. For example, the mismatch between a happy text ("I am fine") and sad audio (flat tone) is captured by the Semantic Edge.
*   **Level 3: Latent Perceiver (Global Context)**
    *   **Mechanism**: Uses a fixed set of latent queries (64 vectors) to attend to the variable-sized output of the Modality Hypergraph.
    *   **Benefit**: Decouples the computational cost from the input sequence length, allowing the model to handle long clinical interviews efficiently.

### 2. Stage 4: Quality-Aware Mixture of Experts (MoE)
The MoE layer acts as the final decision maker, weighing the contributions of each modality based on *data quality* rather than just *feature salience*.
*   **Gating Network**: A lightweight MLP that takes quality metrics (SNR, Face Confidence, Transcript Confidence) as input.
*   **Logic**:
    *   If `Audio_SNR < 10dB` (Noisy), the Audio Expert weight is penalized.
    *   If `Face_Confidence < 0.5` (Face turned away), the Face Expert weight is set to near-zero.
    *   If `Tabular_Missing_Rate > 0.5`, the Tabular Expert is down-weighted.
*   **Top-K Selection**: Only the top K=3 experts are activated for the final prediction, reducing noise from weak modalities.
*   **Expert Architecture**: Each "Expert" is a specialized Feed-Forward Network (FFN) with residual connections, fine-tuned to extract depression-relevant signals from its specific modality embedding.

## Model Specifications

| Modality | Backbone Model | HuggingFace ID | Output Dim | Reported Acc |
|----------|---------------|----------------|------------|--------------|
| Audio | Wav2Vec2-Large + eGeMAPSS | facebook/wav2vec2-large-xlsr-53 + opensmile | 768 | 0.84 |
| Text | MentalRoBERTa | mental/mental-roberta-base | 768 | 0.91 |
| Video | VideoMAE-Base | MCG-NJU/videomae-base | 768 | 0.76-0.79 |
| Face | OpenFace 2.0 + POSTER v2 | OpenFace + poster_v2 | 768 | 0.79 |
| Tabular | FT-Transformer | - | 768 | 0.87 |
| **Fusion** | Cross-Modal Attention | - | 768 | **0.86-0.90** |


## Key Innovations

1. **MentalRoBERTa**: Pre-trained on mental health text for depression-specific semantics
2. **Wav2Vec2-XLSR**: Multi-lingual audio features + eGeMAPSS acoustic features
3. **POSTER v2**: State-of-the-art facial expression recognition
4. **FT-Transformer**: Feature Tokenizer Transformer for robust tabular embeddings
5. **Three-Level Hypergraph**: Captures complex multi-modal interactions
6. **Quality-Aware MoE**: Dynamically weights experts based on input quality
