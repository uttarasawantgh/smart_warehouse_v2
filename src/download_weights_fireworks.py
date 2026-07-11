import os
import requests
from dotenv import load_dotenv

# Load your FIREWORKS_API_KEY from .env
load_dotenv()

def download_fireworks_lora():
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("❌ Missing FIREWORKS_API_KEY in your .env file.")

    account_id = os.getenv("FIREWORKS_ACCOUNT_ID", "default_placeholder")
    model_id = "ft-rotjdido29sbo"
    output_dir = f"./models/{model_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Official Fireworks endpoint to retrieve download URLs
    endpoint = f"https://api.fireworks.ai/v1/accounts/{account_id}/models/{model_id}:getDownloadEndpoint"
    
    print(f"📡 Requesting secure download URLs for {model_id}...")
    response = requests.get(endpoint, headers=headers)
    
    if response.status_code != 200:
        print(f"❌ Failed to get download endpoints. Status Code: {response.status_code}")
        print(response.text)
        return

    data = response.json()
    urls_map = data.get("filenameToSignedUrls", {})
    
    if not urls_map:
        print("⚠️ No downloadable files found for this model artifact.")
        return

    print(f"📦 Found {len(urls_map)} files to download.\n")

    for file_name, download_url in urls_map.items():
        print(f"📥 Downloading {file_name}...")
        file_response = requests.get(download_url, stream=True)
        
        if file_response.status_code == 200:
            target_path = os.path.join(output_dir, file_name)
            
            # ✨ FIX: Ensure the nested subdirectory structure exists before writing the file
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # Stream the file content down to disk
            with open(target_path, "wb") as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Saved: {target_path}\n")
        else:
            print(f"❌ Failed downloading {file_name}. Status: {file_response.status_code}")

    print(f"🎉 Success! All weights are stored at: {output_dir}")

if __name__ == "__main__":
    download_fireworks_lora()