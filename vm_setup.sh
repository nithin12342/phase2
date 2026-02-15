#!/bin/bash

# ==============================================================================
# H5-OmniFusion VM Setup Script (Ubuntu 22.04)
# Run this on your Azure B2ms VM to prepare the environment.
# ==============================================================================

echo "🔵 [1/6] Updating System & Installing Dependencies..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3.10-venv docker.io docker-compose-v2 git

# Enable Docker without sudo
sudo usermod -aG docker $USER
echo "Docker installed. Note: You may need to logout and login for group changes to take effect."

echo "🔵 [2/6] Setting up Project Directory..."
# Assuming you cloned the repo to ~/phase2
if [ ! -d ~/phase2 ]; then
    echo "❌ Directory ~/phase2 not found! Clone repo first."
    exit 1
fi
cd ~/phase2

# Install Python deps locally (optional, for debugging without docker)
pip install gdown

echo "🔵 [3/6] Creating Model Directories..."
mkdir -p ml_pipeline/h5_omnifusion/pretrained_models/audio/wav2vec2-large-xlsr-53
mkdir -p ml_pipeline/h5_omnifusion/pretrained_models/text/mental-roberta-base
mkdir -p ml_pipeline/h5_omnifusion/pretrained_models/video/videomae-base
mkdir -p ml_pipeline/h5_omnifusion/pretrained_models/face/dinov2-base
mkdir -p ml_pipeline/h5_omnifusion/pretrained_models/face/POSTER_V2
mkdir -p ml_pipeline/h5_omnifusion/checkpoints

echo "🔵 [4/6] Model Download Instructions"
echo "================================================================================"
echo "⚠️  IMPORTANT: You must download the models using your Google Drive IDs."
echo "    Replace <ID> with the actual File/Folder ID from your Drive."
echo "================================================================================"

echo "# 1. Wav2Vec2 (Audio)"
echo "gdown <DRIVE_ID_WAV2VEC2> -O ml_pipeline/h5_omnifusion/pretrained_models/audio/wav2vec2-large-xlsr-53/ --folder"
echo ""
echo "# 2. MentalRoBERTa (Text)"
echo "gdown <DRIVE_ID_ROBERTA> -O ml_pipeline/h5_omnifusion/pretrained_models/text/mental-roberta-base/ --folder"
echo ""
echo "# 3. VideoMAE (Video)"
echo "gdown <DRIVE_ID_VIDEOMAE> -O ml_pipeline/h5_omnifusion/pretrained_models/video/videomae-base/ --folder"
echo ""
echo "# 4. DINOv2 (Face)"
echo "gdown <DRIVE_ID_DINOV2> -O ml_pipeline/h5_omnifusion/pretrained_models/face/dinov2-base/ --folder"
echo ""
echo "# 5. POSTER V2 (Expression)"
echo "# (If not already present in repo)"
echo "gdown <DRIVE_ID_POSTER> -O ml_pipeline/h5_omnifusion/pretrained_models/face/POSTER_V2/ --folder"
echo ""
echo "# 6. H5 FUSION CHECKPOINT (Critical)"
echo "gdown <DRIVE_ID_CHECKPOINT> -O ml_pipeline/h5_omnifusion/checkpoints/h5_omnifusion_compliant.pt"
echo "================================================================================"

echo "🔵 [5/6] Launching Application"
echo "Once models are downloaded, run:"
echo "  docker compose -f docker-compose.prod.yml up -d --build"
echo ""
echo "Monitor logs with:"
echo "  docker compose -f docker-compose.prod.yml logs -f"

echo "🔵 [6/6] Configuring Auto-Shutdown (Cost Saving)"
echo "================================================================================"
echo "⚠️  CRITICAL FOE FREE TIER: Setting up auto-shutdown to save CPU hours."
echo "   The VM will automatically shut down at 23:00 (11 PM) local time every day."
echo "================================================================================"
# Add crontab entry to shut down at 23:00 (11 PM) every day
# We need pseudo-root or user crontab. 'sudo shutdown' needs root.
# We'll add to root crontab.
echo "0 23 * * * /sbin/shutdown -h now" | sudo tee -a /var/spool/cron/crontabs/root
sudo chmod 600 /var/spool/cron/crontabs/root
sudo service cron restart

echo "✅ Auto-shutdown scheduled for 23:00 daily."
echo "To remove this, run 'sudo crontab -e' and delete the line."
