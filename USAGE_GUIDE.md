# H5-OmniFusion Project Usage Guide

This guide explains how to manage the deployed H5-OmniFusion system using your Azure CLI.

## 🚀 Prerequisites
- A terminal (PowerShell, CMD, or Bash).
- Azure CLI installed (`az`).
- Logged in execution of `az login`.

---

## 🛑 How to STOP Resources (Save Cost)
When you are not using the system, you should "pause" it to avoid compute charges.

**Run this command:**
```powershell
az containerapp update --name h5-backend --resource-group h5-manual-rg --min-replicas 0 --max-replicas 1
az containerapp ingress disable --name h5-backend --resource-group h5-manual-rg
```

### 💰 Cost Analysis (Paused State)
- **Compute (Backend)**: **$0.00** (Scaled to 0 replicas).
- **Container Registry**: ~$0.17/day per GB (Storage for Docker images).
- **Static Web App**: Free (Free Tier) or ~$0.30/day (Standard).
- **Network/Ingress**: $0.00 (Disabled).

**Total Estimated Idle Cost:** < $5 - $10 per month (mostly for storage).

---

## ▶️ How to START Resources
When you want to demonstrate or use the app.

**Run this command:**
```powershell
az containerapp update --name h5-backend --resource-group h5-manual-rg --min-replicas 1 --max-replicas 1
az containerapp ingress enable --name h5-backend --resource-group h5-manual-rg --target-port 8000 --type external
```
*Wait ~30-60 seconds for the container to pull the image and start.*

---

## 🔗 How to Get the Frontend Link
The Frontend is hosted on Azure Static Web Apps. The URL is permanent.

**Run the correct command:**
```powershell
az staticwebapp show --name h5-frontend --resource-group h5-manual-rg --query "defaultHostname" --output tsv
```
**Or verify directly:**
- **Frontend URL:** `https://blue-mushroom-094076100.1.azurestaticapps.net`
- **Backend Health:** `https://h5-backend.bravepebble-1b927c90.southeastasia.azurecontainerapps.io/health`

---

## 🛠️ Troubleshooting
If the fusion model seems unavailable:
1. Check backend logs:
   ```powershell
   az containerapp logs show --name h5-backend --resource-group h5-manual-rg --tail 50
   ```
2. Verify `/health` endpoint returns `healthy`.

---
**Project Status (v10):** ✅ Deployed & Operational (Champion Model v2)

---

## 🗑️ Option 2: DELETE EVERYTHING (Zero Cost)
Delete all resources to pay **$0.00**.
**Trade-off:** You must re-deploy everything (~30 mins) next time.

**Run:**
```powershell
az group delete --name h5-manual-rg --yes --no-wait
```
*Resources will disappear in ~10 mins.*

### How to Restore from Zero
1.  **Create Resource Group:**
    ```powershell
    az group create --name h5-manual-rg --location southeastasia
    ```
2.  **Create Registry:**
    ```powershell
    az acr create --resource-group h5-manual-rg --name h5regmanual1653298546 --sku Basic --admin-enabled true
    ```
3.  **Run Deployment:**
    ```powershell
    .\deploy_v10.ps1
    ```

---

## 🧹 Intermediate: Reduce Storage Cost (Keep Latest Only)
If you chose Option 1 (Pause), delete old image versions to save space.

**Run:**
```powershell
# List tags
az acr repository show-tags --name h5regmanual1653298546 --repository h5-backend --output table

# Delete old versions (Replace vX)
az acr repository delete --name h5regmanual1653298546 --image h5-backend:vX --yes
```
*Keep only `v10`.*




az staticwebapp show --name h5-frontend --resource-group h5-manual-rg --query "defaultHostname" --output tsv
