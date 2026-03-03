#!/bin/bash

# ==============================================================================
# H5-OmniFusion Azure Deployment Script
# Automates the setup of Azure Container Apps (Backend) and Static Web Apps (Frontend)
# ==============================================================================

# Variables
RESOURCE_GROUP="h5-omnifusion-rg"
LOCATION="eastus"
ACR_NAME="h5registry$RANDOM"
ENV_NAME="h5-env"
BACKEND_APP_NAME="h5-backend"
FRONTEND_APP_NAME="h5-frontend"
STORAGE_NAME="h5storage$RANDOM"
GITHUB_REPO_URL="https://github.com/yourusername/h5-omnifusion" # UPDATE THIS

echo "🔵 [1/7] Logging into Azure..."
az login

echo "🔵 [2/7] Creating Resource Group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

echo "🔵 [3/7] Creating Container Apps Environment..."
az containerapp env create \
  --name $ENV_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

echo "🔵 [4/7] Building and Pushing Backend Image..."
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query "username" -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)

# Build image (assumes running from root)
az acr build --registry $ACR_NAME --image $BACKEND_APP_NAME:v1 --file backend/Dockerfile backend/

echo "🔵 [5/7] Deploying Backend Container App..."
az containerapp create \
  --name $BACKEND_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $ENV_NAME \
  --image $ACR_NAME.azurecr.io/$BACKEND_APP_NAME:v1 \
  --target-port 8000 \
  --ingress external \
  --cpu 1.0 --memory 2.0Gi \
  --min-replicas 0 --max-replicas 3 \
  --registry-server $ACR_NAME.azurecr.io \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --env-vars MODEL_CACHE_DIR=/tmp/models

# Get Backend URL
BACKEND_URL=$(az containerapp show --name $BACKEND_APP_NAME --resource-group $RESOURCE_GROUP --query "properties.configuration.ingress.fqdn" -o tsv)
echo "✅ Backend Deployed at: https://$BACKEND_URL"

echo "🔵 [6/7] Deploying Frontend (Static Web App)..."
# Note: For Static Web Apps to auto-build from GitHub, you usually need to link the repo interactively or via token.
# Here we create the resource and output instructions for linking.
az staticwebapp create \
  --name $FRONTEND_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --location "eastus2" \
  --source $GITHUB_REPO_URL \
  --branch main \
  --app-location "/frontend" \
  --output-location "build"

echo "⚠️  IMPORTANT: Go to Azure Portal -> Static Web Apps -> $FRONTEND_APP_NAME -> Configuration"
echo "    Add Application Setting: REACT_APP_API_URL = https://$BACKEND_URL"

echo "🔵 [7/7] Configuring Storage..."
az storage account create \
  --name $STORAGE_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS

az storage container create \
  --name interviews \
  --account-name $STORAGE_NAME \
  --public-access off

# Get SAS Token or Connection String logic would go here if needed for backend
STORAGE_CONN_STR=$(az storage account show-connection-string --name $STORAGE_NAME --resource-group $RESOURCE_GROUP --query connectionString -o tsv)

# Update Backend with Storage Connection String
az containerapp update \
  --name $BACKEND_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --set-env-vars AZURE_STORAGE_CONNECTION_STRING="$STORAGE_CONN_STR"

echo "✅ Deployment Complete!"
echo "   Backend: https://$BACKEND_URL"
echo "   Storage: $STORAGE_NAME"
echo "   Frontend: Check Azure Portal for URL"
