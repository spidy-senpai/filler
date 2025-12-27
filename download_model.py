import os
import requests
from pathlib import Path

def download_model():
    """
    Download the colorization model file from cloud storage.
    
    For Vercel deployment, you should:
    1. Upload colorization_release_v2.caffemodel to cloud storage (S3, Google Cloud, or Vercel Blob)
    2. Replace MODEL_URL below with your actual URL
    3. This function will download the model on first request
    """
    
    MODEL_URL = "YOUR_MODEL_URL_HERE"  # Replace with actual URL
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
    MODEL_PATH = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
    
    # Check if model already exists
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return MODEL_PATH
    
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print(f"Downloading model from {MODEL_URL}...")
    
    try:
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        # Download with progress
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"Progress: {progress:.1f}%", end='\r')
        
        print(f"\nModel downloaded successfully to {MODEL_PATH}")
        return MODEL_PATH
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

if __name__ == "__main__":
    download_model()
