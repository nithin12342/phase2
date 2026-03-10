param location string = 'southeastasia'
param acrName string = 'h5regmanual1653298546'
param envName string = 'h5-env-866460966'
param appName string = 'h5-backend'
param imageName string = '${acrName}.azurecr.io/${appName}:v10'

@secure()
@description('The HuggingFace API token required for feature extraction.')
param hfToken string

// 1. Container Registry
resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

// 2. Container Apps Environment
resource managedEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: envName
  location: location
  properties: {
    zoneRedundant: false
  }
}

// 3. Container App
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: appName
  location: location
  properties: {
    managedEnvironmentId: managedEnv.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        {
          name: 'acr-password'
          value: acr.listCredentials().passwords[0].value
        }
        {
          name: 'huggingface-token'
          value: hfToken
        }
      ]
    }
    template: {
      containers: [
        {
          name: appName
          image: imageName
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: [
            {
              name: 'HUGGINGFACE_TOKEN'
              secretRef: 'huggingface-token'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 1
      }
    }
  }
}
