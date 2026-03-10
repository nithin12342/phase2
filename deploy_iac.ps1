param(
    [Parameter(Mandatory=$true)]
    [string]$hfToken
)

Write-Host "Creating Resource Group h5-manual-rg..."
az group create --name h5-manual-rg --location southeastasia

Write-Host "Deploying Infrastructure as Code (Bicep)..."
az deployment group create `
    --resource-group h5-manual-rg `
    --template-file infrastructure/main.bicep `
    --parameters hfToken=$hfToken

Write-Host "Deployment Complete!"
