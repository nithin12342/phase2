# H5-OmniFusion: Free Deployment on Hugging Face Spaces (Option C)

**Hugging Face Spaces** offers a "CPU Basic" tier with **2 vCPU and 16 GB RAM** for free. This is perfect for H5-OmniFusion (which needs ~3 GB RAM). It also provides a permanent URL: `https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME`.

## Steps to Deploy

### 1. Create a Space
1.  Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2.  **Space name**: `h5-omnifusion-demo`.
3.  **License**: MIT.
4.  **SDK**: Select **Docker**.
5.  **Hardware**: Keep **CPU Basic (Free)**.
6.  Click **Create Space**.

### 2. Prepare Your Files
You need to upload your code and models to the Space.
1.  Clone the Space locally:
    ```bash
    git clone https://huggingface.co/spaces/<YOUR_USERNAME>/h5-omnifusion-demo
    cd h5-omnifusion-demo
    ```
2.  Copy your project files into this folder:
    *   `backend/`
    *   `ml_pipeline/` (Including `h5_omnifusion` code)
    *   `Dockerfile.hf` (Rename this to `Dockerfile`)
    *   `requirements_azure.txt` (This is referenced in the Dockerfile)

### 3. Handle Large Models (LFS)
Since default git doesn't allow files >10MB, you need **Git LFS** for your models.
If you have downloaded the models to `ml_pipeline/h5_omnifusion/pretrained_models/`:

1.  **Install Git LFS**:
    ```bash
    git lfs install
    ```
2.  **Track large files**:
    ```bash
    git lfs track "*.bin"
    git lfs track "*.pt"
    git lfs track "*.pth"
    git lfs track "*.onnx"
    ```
3.  **Copy your downloaded models** into the `ml_pipeline/...` structure inside the Space folder.

### 4. Push to Deploy
1.  **Rename Dockerfile**:
    Ensure the file you created as `Dockerfile.hf` is named `Dockerfile` in the root of the Space repo.
2.  **Commit and Push**:
    ```bash
    git add .
    git commit -m "Deploy H5-OmniFusion"
    git push
    ```

### 5. Wait for Build
Go to your Space URL. You will see "Building".
Once done, your app will be live at:
`https://huggingface.co/spaces/<YOUR_USERNAME>/h5-omnifusion-demo`

## Important Notes
*   **Persistent URL**: This URL works 24/7 (wakes up on request).
*   **Privacy**: Your Space is public by default. Go to Settings -> "Make Private" if needed (Free tier still works).
*   **Model Upload**: If your internet upstream is slow, uploading models via Git LFS might take time.
