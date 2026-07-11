import os
import json
import csv
import base64
from datetime import datetime, timedelta
from openai import OpenAI

# --- STRICT PATH MATCHING (Preserves your existing workspace tree) ---
DATASET_DIR = "data/raw/Logistics-2/test" 

# Log targets placed safely in your top-level workspace root
LEDGER_JSON_PATH = "compliance_time_series_ledger.json"
LEDGER_CSV_PATH = "compliance_alerts_log.csv"

# --- ON-DEMAND CUSTOM DEPLOYMENT ENDPOINT ---
# Points directly to the serverless deployment identifier you set up
# Change this line at the top of your script:

ACCOUNT_ID = os.getenv("FIREWORKS_ACCOUNT_ID", "default_placeholder")
CUSTOM_MODEL_ID = f"accounts/{ACCOUNT_ID}/deployments/wavb1x6c"
FIREWORKSAI_API_KEY = os.environ.get("FIREWORKSAI_API_KEY")

if not FIREWORKSAI_API_KEY:
    raise ValueError("❌ Missing 'FIREWORKSAI_API_KEY' environment variable. Run 'export FIREWORKSAI_API_KEY=...' first.")

# Connect to the Fireworks Gateway
client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=FIREWORKSAI_API_KEY
)

def init_ledgers():
    """
    Safely checks for existing diagnostics logs. 
    Appends new data to preserve all historical run outputs without overwriting.
    """
    # 1. Handle JSON Ledger Safety (Read and preserve if it exists)
    if os.path.exists(LEDGER_JSON_PATH):
        try:
            with open(LEDGER_JSON_PATH, "r") as f:
                content = f.read().strip()
                if not content:
                    with open(LEDGER_JSON_PATH, "w") as wf:
                        json.dump([], wf)
                    print("🔄 Empty JSON ledger found. Initialized tracking array.")
                else:
                    print(f"🗄️ Existing JSON ledger found at '{LEDGER_JSON_PATH}'. Preserving entries.")
        except Exception as e:
            print(f"⚠️ Warning verifying JSON ledger: {str(e)}")
    else:
        with open(LEDGER_JSON_PATH, "w") as f:
            json.dump([], f)
        print(f"📝 Created new JSON time-series ledger at '{LEDGER_JSON_PATH}'.")

    # 2. Handle CSV Diagnostics Safety (Append only, never overwrite headers)
    if os.path.exists(LEDGER_CSV_PATH):
        print(f"📊 Existing Diagnostic CSV log found at '{LEDGER_CSV_PATH}'. New breaches will append below.")
    else:
        with open(LEDGER_CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Asset_Name", "Alert_Type", 
                "Risk_Level", "Description", "Total_Inventory_Items"
            ])
        print(f"📝 Created new CSV diagnostic log at '{LEDGER_CSV_PATH}'.")

def _encode_local_image_to_base64(image_path):
    """Converts a local file frame to a Base64 data URI string for network delivery."""
    ext = os.path.splitext(image_path)[1].lower().replace(".", "")
    mime_type = "jpeg" if ext in ["jpg", "jpeg"] else "png"
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/{mime_type};base64,{encoded_string}"

def analyze_warehouse_frame_via_fireworks(image_path):
    """Dispatches payload to your on-demand deployment endpoint on Fireworks."""
    try:
        base64_image_uri = _encode_local_image_to_base64(image_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": (
                            "You are an AI Warehouse Operations and Safety Auditor. Analyze the provided camera feed frame "
                            "and return a strictly valid JSON document containing both structural inventory tracking and life-safety compliance analysis. "
                            "Analyze this warehouse zone for safety and inventory metrics. Output formatting requirement: Return ONLY the raw valid JSON payload structure."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": base64_image_uri}
                    }
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model=CUSTOM_MODEL_ID,
            messages=messages,
            temperature=0.1,
            max_tokens=1024
        )
        
        output_text = response.choices[0].message.content.strip()
        
        # Clean any markdown wrap if it slips through the vision layer
        if output_text.startswith("```"):
            lines = output_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned_text = "\n".join(lines).strip()
        else:
            cleaned_text = output_text

        return json.loads(cleaned_text)
    except Exception as e:
        return {"error": f"Cloud deployment response error: {str(e)}"}

def run_pure_fireworks_pipeline():
    init_ledgers()
    
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: Local directory '{DATASET_DIR}' not found. Verify your paths.")
        return

    valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
    all_image_files = sorted([f for f in os.listdir(DATASET_DIR) if f.lower().endswith(valid_extensions)])
    
    # Process downsampled slices (sampling every 15th frame)
    image_files = all_image_files[::15]

    total_files = len(image_files)
    if total_files == 0:
        print(f"⚠️ No matching images found in path: {DATASET_DIR}")
        return

    print(f"📡 Target Active Cloud Endpoint: {CUSTOM_MODEL_ID}")
    print(f"📦 Streaming {total_files} warehouse frames directly to your on-demand deployment...")
    print("=" * 80)

    base_time = datetime.now()

    for idx, filename in enumerate(image_files, start=1):
        image_path = os.path.join(DATASET_DIR, filename)
        frame_timestamp = (base_time + timedelta(minutes=idx * 5)).strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"🎥 [{idx}/{total_files}] Processing Frame: {filename}")
        
        payload = analyze_warehouse_frame_via_fireworks(image_path)
        
        if "error" in payload:
            print(f"  ❌ Generation Failure: {payload['error']}")
            print("  💡 Note: Since Min Replicas=0, the first frame triggers a cold-start. Give it ~60s to warm up.")
            continue

        inventory_list = payload.get("inventory_tracking", {}).get("structural_inventory", []) if isinstance(payload.get("inventory_tracking"), dict) else []
        violations = payload.get("life_safety_compliance", {}).get("violations", []) if isinstance(payload.get("life_safety_compliance"), dict) else []
        compliances = payload.get("life_safety_compliance", {}).get("compliances", []) if isinstance(payload.get("life_safety_compliance"), dict) else []

        total_items_counted = 0
        for item in inventory_list:
            if isinstance(item, dict):
                try:
                    total_items_counted += int(item.get("quantity", 1))
                except (TypeError, ValueError):
                    total_items_counted += 1
            else:
                total_items_counted += 1

        high_risk_count = 0
        valid_violations = []
        for v in violations:
            if isinstance(v, dict):
                valid_violations.append(v)
                if str(v.get("risk_level", "")).lower() == "high":
                    high_risk_count += 1
            elif isinstance(v, str):
                valid_violations.append({"type": "Safety Alert", "description": v, "risk_level": "Medium"})

        ledger_entry = {
            "timestamp": frame_timestamp,
            "asset_identity": filename,
            "summary_metrics": {
                "total_inventory_items": total_items_counted,
                "total_violations_detected": len(valid_violations),
                "total_compliant_indicators": len(compliances),
                "critical_safety_breach": high_risk_count > 0
            },
            "raw_inference_payload": payload
        }

        # Safe transactional append step for JSON ledger
        try:
            with open(LEDGER_JSON_PATH, "r+") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
                data.append(ledger_entry)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
        except Exception as e:
            print(f"  ⚠️ JSON write skipped: {str(e)}")

        # Safe append step for CSV alerts
        if valid_violations:
            try:
                with open(LEDGER_CSV_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    for violation in valid_violations:
                        writer.writerow([
                            frame_timestamp, filename,
                            violation.get("type", "Unknown Breach"),
                            violation.get("risk_level", "Medium"),
                            violation.get("description", ""),
                            total_items_counted
                        ])
            except Exception as e:
                print(f"  ⚠️ CSV log skipped: {str(e)}")

        print(f"  📊 Items: {total_items_counted} | Breaches: {len(valid_violations)} (🔥 {high_risk_count} CRITICAL)")
        print("-" * 80)

    print(f"\n✅ Pipeline Complete! Output logs securely updated in the workspace root.")

if __name__ == "__main__":
    run_pure_fireworks_pipeline()