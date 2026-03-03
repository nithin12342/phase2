# H5-OmniFusion: Azure Deployment Setup

This document records the infrastructure layout and deployment strategy used to host the H5-OmniFusion project on Microsoft Azure.

---

## 🏗️ 1. Architecture Overview
The system relies on a hybrid execution strategy to minimize compute costs:
1.  **Frontend (Azure Static Web Apps):** Serves the React interface to the user.
2.  **Backend (Azure Container Apps):** A lightweight FastAPI server hosting the local `H5-OmniFusion` checkpoint.
3.  **Feature Extraction (Remote APIs):** The heavy multimedia extraction (Wav2Vec2, RoBERTa, VideoMAE, DinoV2) is outsourced dynamically to the free tier of the HuggingFace Inference API via the backend `hf_client.py`.

---

## 🌍 2. Global Configuration
All resources are grouped together under a single management boundary for cost monitoring and deletion ease.

*   **Resource Group:** `h5-manual-rg`
*   **Primary Location:** `Southeast Asia` (Selected for backend due to availability of consumption-plan Container Apps).

---

## ⚙️ 3. Backend Architecture (Compute Layer)
The backend is Dockerized and deployed using Serverless compute. 

### A. Azure Container Registry (ACR)
Stores the Docker images required by the Container App.
*   **Name:** `h5regmanual1653298546`
*   **Region:** `Southeast Asia`
*   **SKU:** Basic
*   **Cost Strategy:** Keep only the latest image (`v10`) and delete old tags to minimize storage fees.

### B. Azure Container Apps Environment
The underlying virtual network environment for the container.
*   **Name:** `h5-env-866460966`
*   **Region:** `Southeast Asia`

### C. Azure Container App (Backend API)
The actual running server that processes requests and runs the fusion prediction.
*   **Name:** `h5-backend`
*   **Region:** `Southeast Asia`
*   **Workload Profile:** Consumption (Serverless scale-down-to-zero).
*   **Compute Profile (Optimized):** `1.0 vCPU` / `2.0Gi Memory`
*   **Container Image:** `h5regmanual1653298546.azurecr.io/h5-backend:v10`
*   **Ingress Settings:** Enabled, Port 8000, Type External.
*   **Environment Variables:** Security tokens (`HUGGINGFACE_TOKEN`, `CORS_ORIGINS`, `MODEL_CACHE_DIR`) are passed here.

---

## 🖥️ 4. Frontend Architecture (Hosting Layer)
The React dashboard.

*   **Service Used:** Azure Static Web Apps
*   **Name:** `h5-frontend`
*   **Region:** `East Asia` (Static Web Apps have different regional availability; East Asia is closest to Southeast Asia).
*   **Plan:** Free Tier ($0/month).
*   **Permanent URL:** `https://blue-mushroom-094076100.1.azurestaticapps.net`

---

## 💰 5. Cost Management Strategy
To protect the subscription from runaway charges, the following rules are applied:
1.  **Pause When Idle:** Send CLI commands to set `min-replicas = 0` and disable Ingress when the backend is not actively being tested.
2.  **Delete Images:** Regularly purge old Container Registry uploads.
## ⚠️ 6. Disaster Recovery & Total Reset
If you need to completely delete the project to avoid all costs, or if you need to redeploy everything from scratch, follow these instructions.

### 6.1. Delete Everything
This deletes the Resource Group and all associated resources instantly dropping your cost to $0.
```powershell
az group delete --name h5-manual-rg --yes --no-wait
```

### 6.2. Recreate Resources From Scratch
If the resource group was deleted, you must follow this required order to bring the backend online again. Our `deploy_v10.ps1` script alone *cannot* recreate the resource group or the container registry.

**1. Recreate the Resource Group:**
```powershell
az group create --name h5-manual-rg --location southeastasia
```

**2. Recreate the Container Registry (ACR):**
```powershell
az acr create --resource-group h5-manual-rg --name h5regmanual1653298546 --sku Basic --admin-enabled true
```

**3. Run the Backend Deployment Script:**
Open PowerShell in the `phase 2` directory and run:
```powershell
.\deploy_v10.ps1
```
*(This script will build the Docker container locally, push it to your newly created ACR, create the Container App environment, and deploy the Container App itself).*

### 6.3. Recreate the Frontend
If you deleted the Azure Static Web App, you must go into the Azure Portal or use the Azure Static Web Apps VS Code extension to reconnect your GitHub repository (or local code) to a new Static Web App instance. The previous auto-generated permanent URL (`blue-mushroom...`) will be lost, and a new one will be generated for the frontend.
