# process_batch.py
import os
import json
import csv
from datetime import datetime, timedelta
import random
#from src.engine import analyze_warehouse_frame
from engine_gemma4 import analyze_warehouse_frame

# --- CONFIGURATION ---
DATASET_DIR = "data/raw/Logistics-2/test"
LEDGER_JSON_PATH = os.getenv("LEDGER_JSON_PATH", "data/compliance_time_series_ledger.json")
LEDGER_CSV_PATH = os.getenv("LEDGER_CSV_PATH", "data/compliance_alerts_log.csv")

def init_ledgers():
    """Ensures storage layers exist before writing records."""
    os.makedirs("data", exist_ok=True)
    
    # Initialize JSON ledger as an empty array if it doesn't exist
    if not os.path.exists(LEDGER_JSON_PATH):
        with open(LEDGER_JSON_PATH, "w") as f:
            json.dump([], f)
            
    # Initialize CSV alert ledger with explicit headers if it doesn't exist
    if not os.path.exists(LEDGER_CSV_PATH):
        with open(LEDGER_CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Asset_Name", "Alert_Type", 
                "Risk_Level", "Description", "Total_Inventory_Items"
            ])

def run_batch_pipeline():
    init_ledgers()
    
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: Target dataset directory '{DATASET_DIR}' not found.")
        return

    # Gather all image frames in target logistics camera path
    # valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
    # image_files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(valid_extensions)]
    # Gather all image frames in target logistics camera path
    valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
    all_image_files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(valid_extensions)]
    all_image_files.sort()  # Ensuring chronological or alphabetical order

    # Apply Option 2: Sample every 15th frame to stay safely within the 1200s timeout
    image_files = all_image_files[::15]

    print(f"✂️ Spatial Downsampling Active: Reduced payload from {len(all_image_files)} to {len(image_files)} frames.")
    
    total_files = len(image_files)
    if total_files == 0:
        print(f"⚠️ No matching image assets found in {DATASET_DIR}")
        return

    print(f"📦 Found {total_files} warehouse camera frames to batch process.")
    print(f"🕒 Time-Series Ledger Targets:\n  ➡️ JSON: {LEDGER_JSON_PATH}\n  ➡️ CSV Alerts: {LEDGER_CSV_PATH}\n")
    print("=" * 80)

    base_time = datetime.now()

    for idx, filename in enumerate(image_files, start=1):
        image_path = os.path.join(DATASET_DIR, filename)
        frame_timestamp = (base_time + timedelta(minutes=idx * 5)).strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"🎥 [{idx}/{total_files}] Processing Frame: {filename} ({frame_timestamp})")
        
        # 1. Execute Multi-Modal VLM Inference Core
        payload = analyze_warehouse_frame(image_path)
        
        # Guard against hard pipeline failures or empty evaluations
        if not payload or "error" in payload:
            print(f"  ❌ Skipping row due to engine error: {payload.get('error', 'Empty response')}")
            continue

        # Extract sub-entities gracefully with default fallback targets
        inventory_block = payload.get("inventory_tracking", {})
        if isinstance(inventory_block, dict):
            inventory_list = inventory_block.get("structural_inventory", [])
        else:
            inventory_list = []

        compliance_block = payload.get("life_safety_compliance", {})
        if isinstance(compliance_block, dict):
            violations = compliance_block.get("violations", [])
            compliances = compliance_block.get("compliances", [])
        else:
            violations = []
            compliances = []

        # --- HARDENED PROCESSING LAYER ---
        # Robust item counting handling dictionaries, strings, and bad value types
        total_items_counted = 0
        if isinstance(inventory_list, list):
            for item in inventory_list:
                if isinstance(item, dict):
                    raw_qty = item.get("quantity", 1)
                    try:
                        # Force string values like "2" into integers cleanly
                        total_items_counted += int(raw_qty)
                    except (TypeError, ValueError):
                        # Handle text anomalies like "unknown" or None gracefully
                        total_items_counted += 1
                elif isinstance(item, str):
                    total_items_counted += 1  # Fallback if it's just a raw label string

        high_risk_count = 0
        valid_violations = []
        if isinstance(violations, list):
            for v in violations:
                if isinstance(v, dict):
                    valid_violations.append(v)
                    if str(v.get("risk_level", "")).lower() == "high":
                        high_risk_count += 1
                elif isinstance(v, str):
                    # Handle case where model spits back plain string warnings
                    valid_violations.append({"type": "Safety Alert", "description": v, "risk_level": "Medium"})

        # 2. Build Unified Time-Series Ledger Transaction Object
        ledger_entry = {
            "timestamp": frame_timestamp,
            "asset_identity": filename,
            "summary_metrics": {
                "total_inventory_items": total_items_counted,
                "total_violations_detected": len(valid_violations),
                "total_compliant_indicators": len(compliances) if isinstance(compliances, list) else 0,
                "critical_safety_breach": high_risk_count > 0
            },
            "raw_inference_payload": payload
        }

        # 3. Write Atomic Entry to JSON Time-Series Document Array
        try:
            with open(LEDGER_JSON_PATH, "r+") as f:
                data = json.load(f)
                data.append(ledger_entry)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
        except Exception as e:
            print(f"  ⚠️ Failed appending to JSON ledger: {str(e)}")

        # 4. Stream High/Medium Risk Safety Violations directly into flat CSV Ledger
        if valid_violations:
            try:
                with open(LEDGER_CSV_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    for violation in valid_violations:
                        writer.writerow([
                            frame_timestamp,
                            filename,
                            violation.get("type", "Unknown Breach"),
                            violation.get("risk_level", "Medium"),
                            violation.get("description", ""),
                            total_items_counted
                        ])
            except Exception as e:
                print(f"  ⚠️ Failed logging CSV row: {str(e)}")

        # Print inline process update
        print(f"  📊 Counted: {total_items_counted} items | Violations: {len(valid_violations)} (🔥 {high_risk_count} CRITICAL)")
        print("-" * 80)

    print(f"\n✅ Batch Execution Completed! Processed {total_files} frames.")
    print(f"💾 Comprehensive historical metrics logged cleanly to your storage volumes.")

if __name__ == "__main__":
    run_batch_pipeline()