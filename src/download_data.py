import os
from dotenv import load_dotenv
from roboflow import Roboflow

# Load environment variables from your local .env file
load_dotenv()

def run_download():
    # Grab your token securely from the local environment state
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("❌ Missing ROBOFLOW_API_KEY. Please ensure it is declared in your system environment or a .env file.")
    
    rf = Roboflow(api_key=api_key)
    
    # Since we are at C:\amd_wp, this path resolves safely to C:\amd_wp\data\raw
    # The short root directory prevents the Windows Max Path Limit from breaking the unzip process
    target_dir = "./data/raw"
    os.makedirs(target_dir, exist_ok=True)
    os.chdir(target_dir)
    
    print("📥 Initiating dataset download from Roboflow Universe...")
    print(" -> Targeting workspace: large-benchmark-datasets")
    print(" -> Fetching project: logistics-sz9jr")
    
    # Connect directly to the specific global benchmark archive
    warehouse_project = rf.workspace("large-benchmark-datasets").project("logistics-sz9jr")
    
    # Downloading the optimized version 2 configuration in COCO JSON format
    warehouse_project.version(2).download("coco")
    
    print(f"\n✅ Extraction complete! All benchmark logistics assets are saved locally inside: {target_dir}")

if __name__ == "__main__":
    run_download()