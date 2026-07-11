import os
import requests
from dotenv import load_dotenv

# Load credentials
load_dotenv()

def upload_dataset():
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("❌ Missing FIREWORKS_API_KEY in your .env file.")
        
    file_path = "./data/processed/warehouse_train.jsonl"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Could not find training file at {file_path}")
        
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    # 1. Dynamically retrieve your profile/account ID info
    print("🔍 Authenticating profile info with Fireworks AI...")
    profile_url = "https://api.fireworks.ai/inference/v1/users/me"
    profile_response = requests.get(profile_url, headers=headers)
    
    if profile_response.status_code != 200:
        print("❌ Authentication failed. Please check your FIREWORKS_API_KEY inside .env.")
        print(profile_response.text)
        return
        
    # Standard profile setups map your username or email string as the primary account namespace
    account_id = profile_response.json().get("username")
    if not account_id:
        # Fallback to display name string if username field is structured differently
        account_id = profile_response.json().get("id")

    print(f"🆔 Authenticated successfully! Account Namespace: {account_id}")
    print("📤 Uploading warehouse_train.jsonl directly to account data storage...")
    
    # 2. Point to the absolute upload destination mapping your exact account context
    upload_url = f"https://api.fireworks.ai/v1/accounts/{account_id}/files"
    
    with open(file_path, "rb") as f:
        files = {
            "file": (os.path.basename(file_path), f, "application/jsonl")
        }
        data = {
            "purpose": "fine-tune"
        }
        
        response = requests.post(upload_url, headers=headers, files=files, data=data)
        
    if response.status_code == 200:
        res_data = response.json()
        print("\n✅ Upload Complete!")
        print(f"🆔 File ID: {res_data.get('name') or res_data.get('id')}")
        print("👉 Save this File ID string; we will use it next to deploy the training loop framework.")
    else:
        print(f"\n❌ Upload Failed with Status Code {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    upload_dataset()