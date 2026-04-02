# Building and Deploying via GitHub Codespaces

Yes, you can absolutely build this image inside a GitHub Codespace! In fact, it is **highly recommended** because GitHub Codespaces run on backbone cloud servers with internet speeds up to 10 Gbps. It will download the massive PyTorch libraries in seconds instead of hours.

Here are the complete instructions to build, test, and push your container directly from a Codespace.

## Phase 1: Build the Image in Codespace

1. **Launch your Codespace:**
   - Go to your repository on GitHub.
   - Click the green **`<> Code`** button.
   - Switch to the **Codespaces** tab and click **Create codespace on main**.
   - Wait a minute or two for the web environment to initialize.

2. **Trigger the Fast Build:**
   Open the terminal inside your Codespace and run the multi-stage Docker build:
   ```bash
   docker build -t h5-backend:latest -f Dockerfile.backend .
   ```
   *(Notice how `apt-get` and `pip install` fly by instantly thanks to the cloud internet bandwidth!)*

## Phase 2: Test the Built Container

1. **Run the container locally inside the Codespace:**
   ```bash
   docker run -d -p 8000:8000 --name test_h5_backend h5-backend:latest
   ```

2. **Verify it works:**
   Check the logs to ensure the server actually started without crashing on the `typing_extensions` error:
   ```bash
   docker logs test_h5_backend
   ```
   *(You should see `Uvicorn running on http://0.0.0.0:8000`)*

## Phase 3: Push to Azure Cloud

Your Codespace terminal comes with the Azure CLI (`az`) accessible.

1. **Authenticate to Azure:**
   Because you are in a web browser, use the device code login method:
   ```bash
   az login --use-device-code
   ```
   *Follow the on-screen link and enter the code to securely log in.*

2. **Log into your Azure Container Registry (ACR):**
   ```bash
   az acr login --name <YourAzureRegistryName>
   ```

3. **Tag the Image:**
   ```bash
   docker tag h5-backend:latest <YourAzureRegistryName>.azurecr.io/h5-backend:latest
   ```

4. **Push the Image:**
   ```bash
   docker push <YourAzureRegistryName>.azurecr.io/h5-backend:latest
   ```

*Once pushed, your image is securely inside Azure and ready to be deployed to Web Apps or Container Instances!*
