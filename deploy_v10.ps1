# Deploy H5-OmniFusion v10 (Fixed Torch Load)
# Run this script to apply the weights_only=False fix

echo "1. Logging into Azure Container Registry..."
az acr login --name h5regmanual1653298546

echo "2. Pushing v10 image (uploads 134MB layer one last time)..."
docker push h5regmanual1653298546.azurecr.io/h5-backend:v10

echo "3. Updating Azure Container App to use v10..."
az containerapp update --name h5-backend --resource-group h5-manual-rg --image h5regmanual1653298546.azurecr.io/h5-backend:v10 --min-replicas 1 --max-replicas 1
az containerapp ingress enable --name h5-backend --resource-group h5-manual-rg --target-port 8000 --type external

echo "✅ Deployment Complete! Fusion model should work now."
