"""
H5-OmniFusion Configuration Module
Contains all configuration parameters for the preprocessing pipeline.
"""
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import os


@dataclass
class Config:
    """
    Central configuration for H5-OmniFusion Pipeline.
    All paths, parameters, and thresholds are defined here.
    """
    
    DAIC_WOZ_PATH: str = '/content/drive/MyDrive/DAIC-WOZ_Datasets'
    EATD_CORPUS_PATH: str = '/content/drive/MyDrive/EATD-Corpus'
    PRETRAINED_PATH: str = '/content/drive/MyDrive/pretrained_models'
    OUTPUT_PATH: str = '/content/drive/MyDrive/DAIC-WOZ_Datasets/H5_OmniFusion_Output'
    LABELS_PATH: str = '/content/drive/MyDrive/DAIC-WOZ_Datasets/merged_labels.csv'
    TEMP_PATH: str = '/tmp/h5_omnifusion'
    
    SAMPLE_RATE: int = 16000  # Required for Wav2Vec2
    WINDOW_SEC: float = 10.0  # Segmentation window
    OVERLAP: float = 0.5  # 50% overlap
    TARGET_LUFS: float = -23.0  # EBU R128 loudness
    NOISE_PROP_DECREASE: float = 0.8  # Noise reduction strength
    VAD_TOP_DB: int = 30  # Voice activity threshold
    
    PAUSE_MIN_DURATION_MS: float = 200  # Minimum pause duration
    PAUSE_ENERGY_THRESHOLD_DB: float = -30  # Energy threshold for pause
    SIGH_MIN_DURATION_SEC: float = 1.0
    SIGH_MAX_DURATION_SEC: float = 3.0
    SIGH_MAX_FREQ_HZ: float = 500  # Low-frequency dominance
    
    F0_MIN_HZ: float = 75
    F0_MAX_HZ: float = 500
    
    TARGET_FPS: int = 5
    NUM_FRAMES: int = 16  # For VideoMAE
    FRAME_SIZE: Tuple[int, int] = (224, 224)
    
    BLUR_THRESHOLD: float = 50.0  # Laplacian variance
    BRIGHTNESS_MIN: int = 80
    BRIGHTNESS_MAX: int = 180
    
    IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    FACE_CONFIDENCE_THRESHOLD: float = 0.8
    FACE_MARGIN: float = 0.2  # 20% expansion
    
    GAZE_DIRECT_MAX: float = 10.0
    GAZE_INDIRECT_MAX: float = 15.0
    
    EAR_THRESHOLD: float = 0.2  # Eye Aspect Ratio
    NORMAL_BLINK_RATE_MIN: int = 15  # blinks per minute
    NORMAL_BLINK_RATE_MAX: int = 20
    
    MAX_TOKEN_LENGTH: int = 512
    
    FILLERS: List[str] = field(default_factory=lambda: [
        'um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean'
    ])
    
    FIRST_PERSON: List[str] = field(default_factory=lambda: [
        'i', 'me', 'my', 'myself', 'mine'
    ])
    ABSOLUTIST: List[str] = field(default_factory=lambda: [
        'always', 'never', 'nothing', 'everything', 'completely', 'totally', 'absolutely'
    ])
    NEGATIVE_EMOTION: List[str] = field(default_factory=lambda: [
        'sad', 'depressed', 'hopeless', 'worthless', 'anxious', 'afraid', 'terrible'
    ])
    
    AUDIO_SNR_MIN_DB: float = 15.0
    AUDIO_CLIPPING_MAX_RATIO: float = 0.01  # 1%
    AUDIO_VAD_MIN_RATIO: float = 0.4  # 40%
    VIDEO_FACE_DETECTION_MIN_RATIO: float = 0.8
    
    EMBED_DIM: int = 768  # Universal embedding dimension
    
    WAV2VEC2_MODEL: str = 'facebook/wav2vec2-large-xlsr-53'
    MENTAL_ROBERTA_MODEL: str = 'mental/mental-roberta-base'
    CHINESE_BERT_MODEL: str = 'hfl/chinese-roberta-wwm-ext'
    VIDEOMAE_MODEL: str = 'MCG-NJU/videomae-base'
    XLM_ROBERTA_MODEL: str = 'xlm-roberta-base'
    
    d_model: int = 768
    dropout: float = 0.1
    n_classes: int = 2
    predict_vad: bool = False
    
    @dataclass
    class AudioConfig:
        input_dim: int = 768
        backbone_dim: int = 1024  # Wav2Vec2 output dim
        egemaps_dim: int = 88    # eGeMAPS features
        use_egemaps: bool = False  # Using pre-extracted features
        use_mamba: bool = True
        use_kan: bool = True
        n_mamba_layers: int = 2
        n_layers: int = 2
    
    @dataclass 
    class TextConfig:
        input_dim: int = 768
        backbone_dim: int = 768  # RoBERTa output dim
        use_mamba: bool = True
        use_kan: bool = True
        n_mamba_layers: int = 2
        n_layers: int = 2
    
    @dataclass
    class VideoConfig:
        input_dim: int = 768
        backbone_dim: int = 768  # ViT output dim
        use_mamba: bool = True
        use_kan: bool = True
        use_timesformer: bool = True
        n_mamba_layers: int = 2
        n_layers: int = 2
    
    @dataclass
    class FaceConfig:
        input_dim: int = 768
        au_dim: int = 35  # 17 AUs * 2 (intensity + presence) + gaze
        use_mamba: bool = False
        use_kan: bool = True
        n_mamba_layers: int = 1
        n_lstm_layers: int = 2
        bidirectional: bool = True
        n_layers: int = 1
    
    @dataclass
    class TabularConfig:
        input_dim: int = 768
        hidden_dim: int = 512
        n_features: int = 20  # Number of tabular features
        use_kan: bool = True
        kan_grid_size: int = 5
        kan_spline_order: int = 3
        n_layers: int = 2
    
    @dataclass
    class FusionConfig:
        local_n_heads: int = 8
        local_dropout: float = 0.1
        modality_n_heads: int = 8
        modality_n_layers: int = 2
        use_ms2: bool = True
        shared_ratio: float = 0.5
        n_latents: int = 16
        n_perceiver_blocks: int = 2
        perceiver_n_heads: int = 8
        use_mamba_in_perceiver: bool = True
    
    @dataclass
    class MoEConfig:
        expert_hidden_dim: int = 512
        n_quality_features: int = 5
    
    audio: AudioConfig = field(default_factory=AudioConfig)
    text: TextConfig = field(default_factory=TextConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    tabular: TabularConfig = field(default_factory=TabularConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    
    @dataclass
    class OptimizerConfig:
        lr: float = 1e-4
        weight_decay: float = 0.01
    
    @dataclass
    class SchedulerConfig:
        warmup_ratio: float = 0.1
    
    @dataclass
    class LossConfig:
        lambda_cls: float = 1.0
        lambda_phq: float = 0.3
        lambda_orth: float = 0.05
        focal_alpha: float = 0.25
        focal_gamma: float = 2.0
        label_smoothing: float = 0.1
    
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    
    n_epochs: int = 50
    patience: int = 15
    mixed_precision: bool = True
    
    BATCH_SIZE: int = 1  # Process one participant at a time for memory
    CHECKPOINT_FREQUENCY: int = 10  # Save every 10 participants
    USE_FP16: bool = True  # Use half precision for inference
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [self.OUTPUT_PATH, self.TEMP_PATH]:
            os.makedirs(path, exist_ok=True)
    
    def get_daic_woz_participant_path(self, pid: str) -> str:
        """Get path to a DAIC-WOZ participant folder/zip."""
        return os.path.join(self.DAIC_WOZ_PATH, f'{pid}.zip')
    
    def get_eatd_participant_path(self, pid: str) -> str:
        """Get path to an EATD-Corpus participant folder."""
        return os.path.join(self.EATD_CORPUS_PATH, pid)
    

    @dataclass
    class Training:
        LR: float = 1e-4
        WEIGHT_DECAY: float = 0.01
        BATCH_SIZE: int = 32  # Larger batch for stable gradients on embeddings
        GRADIENT_ACCUMULATION_STEPS: int = 1
        MAX_GRAD_NORM: float = 1.0
        
        OPTIMIZER: str = "adamw"  # Options: adamw, radam, sgd, ranger, etc.
        
        WARMUP_RATIO: float = 0.1
        EPOCHS: int = 50
        MIN_EPOCHS: int = 35
        PATIENCE: int = 15
        
        MIXUP_ALPHA: float = 0.2  # Beta(0.2, 0.2) for mixup
        MIXUP_PROB: float = 0.5   # Probability to apply mixup
        FEATURE_NOISE_STD: float = 0.01  # Gaussian noise for robustness
        DROPOUT: float = 0.3
        LABEL_SMOOTHING: float = 0.1
        
        LAMBDA_CLS: float = 1.0
        LAMBDA_PHQ: float = 0.3
        LAMBDA_ORTH: float = 0.05
    
    TRAIN: Training = field(default_factory=Training)  # Access via CFG.TRAIN

    @dataclass
    class RLNAT:
        ENABLED: bool = True
        
        USE_GEMINI: bool = False
        GEMINI_API_KEY: str = ""
        JOURNAL_PATH: str = "/content/drive/MyDrive/DAIC-WOZ_Datasets/training_journal_gemini.md"
        
        LLM_ADVISOR_PROVIDER: str = "agent_lightning"
        
        GEMINI_MODEL: str = "gemini-2.0-flash"  # Options: gemini-2.0-flash, gemini-2.5-flash
        LLM_TEMPERATURE: float = 0.3  # Lower = more deterministic
        LLM_MAX_RETRIES: int = 3
        
        COLLECT_TRAJECTORIES: bool = True
        TRAJECTORY_PATH: str = "/content/drive/MyDrive/Dysia/rl_trajectories.jsonl"
        
        USE_HOAC: bool = True
        L0_ENABLED: bool = True   # Meta-strategy (Decision Transformer)
        L1_ENABLED: bool = True   # Gating policy (IQL)
        L2_ENABLED: bool = True   # HP bandit (Thompson Sampling)
        
        LLM_FEEDBACK_INTERVAL: int = 1  # Get LLM advice every N epochs
        
        HARD_MINING_RATIO: float = 0.3  # Ensure 30% of batch are "hard" samples
        LOSS_EMA_ALPHA: float = 0.1     # For smoothing loss history
        
        CONTROLLER_EPSILON: float = 0.2 # Probability of random exploration
        ADAPTATION_INTERVAL: int = 1    # Epochs between adjustments
        
        LR_MIN: float = 1e-6
        LR_MAX: float = 5e-4
        MIXUP_MIN: float = 0.0
        MIXUP_MAX: float = 0.4
        
        REWARD_LAMBDA_F1: float = 1.0
        REWARD_LAMBDA_PHQ: float = 0.5
        REWARD_LAMBDA_ORTH: float = 0.1
        REWARD_LAMBDA_SMOOTH: float = 0.05
    
    RL: RLNAT = field(default_factory=RLNAT)  # Access via CFG.RL

    def get_pretrained_model_path(self, category: str, model_name: str) -> str:
        """Get path to a local pretrained model."""
        return os.path.join(self.PRETRAINED_PATH, category, model_name)


CFG = Config()
