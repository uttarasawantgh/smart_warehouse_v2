import torch
import os
import json
import csv
import glob
import re
from datetime import datetime, timedelta
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
from tqdm import tqdm

# --- STRICT WORKSPACE PATH ALIGNMENT ---
BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_DIR = "./models/amd_warehouse_qwen_lora"
DATASET_DIR = "data/raw/Logistics-2/test" 

LEDGER_JSON_PATH = "compliance_time_series_ledger.json"
LEDGER_CSV_PATH = "compliance_alerts_log.csv"

def init_ledgers():
    """Ensures storage layers exist before writing records, appending gracefully."""
    # 1. Initialize JSON ledger as an empty array if missing
    if os.path.exists(LEDGER_JSON_PATH):
        print(f"🗄️ Existing JSON ledger verified at '{LEDGER_JSON_PATH}'. Preserving history.")
    else:
        with open(LEDGER_JSON_PATH, "w") as f:
            json.dump([], f)
        print(f"📝 Created new JSON time-series ledger at '{LEDGER_JSON_PATH}'.")
            
    # 2. Initialize CSV alert ledger with headers if missing
    if os.path.exists(LEDGER_CSV_PATH):
        print(f"📊 Existing Diagnostic CSV log verified at '{LEDGER_CSV_PATH}'. New alerts will append.")
    else:
        with open(LEDGER_CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Asset_Name", "Alert_Type", 
                "Risk_Level", "Description", "Total_Inventory_Items"
            ])
        print(f"📝 Created new CSV diagnostic log at '{LEDGER_CSV_PATH}'.")

def main():
    init_ledgers()
    
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: Target dataset directory '{DATASET_DIR}' not found.")
        return

    # Gather and sort all image assets in the target path
    valid_extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    image_files = []
    for ext in valid_extensions:
        image_files.extend(glob.glob(os.path.join(DATASET_DIR, ext)))
    image_files.sort()

    # --- THROTTLED SPATIAL DOWNSAMPLING ---
    # Process every 50th frame to scale down to 191 total items for immediate evaluation
    sampled_image_files = image_files[::50]
    total_files = len(sampled_image_files)
    
    if total_files == 0:
        print(f"⚠️ No matching image assets found in {DATASET_DIR}")
        return

    print(f"📦 Spatial Downsampling Active: Sampling {total_files} of {len(image_files)} total frames.")
    print("📡 Loading local fine-tuned model and processor layers onto Instinct core elements...")
    
    # Load base weights natively using standard PyTorch ROCm setup
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="sdpa"
    )
    # Merge local AMD adapter layers
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model = model.to("cuda")
    
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    
    base_time = datetime.now()
    print(f"\n🔥 Commencing local VLM inference batch loops (Throttled Baseline)...")
    print("=" * 80)

    for idx, img_path in enumerate(tqdm(sampled_image_files, desc="Batch Auditing"), start=1):
        try:
            filename = os.path.basename(img_path)
            frame_timestamp = (base_time + timedelta(minutes=idx * 5)).strftime("%Y-%m-%d %H:%M:%S")
            raw_image = Image.open(img_path).convert("RGB")
            
            # Reconstruct target prompt matching your exact custom adapter training schema
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": "Perform a comprehensive safety and compliance audit on this warehouse zone. Return the results strictly as a valid JSON object matching the training schema."}
                    ]
                }
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[raw_image], padding=True, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    repetition_penalty=1.2, # Verified stabilization penalty
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            # --- RUGGED JSON CLEANING AND REPAIR LAYER ---
            cleaned_json_string = output_text.strip()
            
            # 1. Strip standard Markdown wrappers if added by inference generation
            if cleaned_json_string.startswith("```json"):
                cleaned_json_string = cleaned_json_string.split("```json")[1].split("```")[0].strip()
            elif cleaned_json_string.startswith("```"):
                cleaned_json_string = cleaned_json_string.split("```")[1].split("```")[0].strip()
            
            # 2. Strip out VLM inline commentary // that break standard JSON parsers
            cleaned_json_string = re.sub(r'//.*$', '', cleaned_json_string, flags=re.MULTILINE)
            
            # 3. Target common VLM truncation errors (missing closing braces)
            if cleaned_json_string.startswith("{") and not cleaned_json_string.endswith("}"):
                open_braces = cleaned_json_string.count("{")
                close_braces = cleaned_json_string.count("}")
                if open_braces > close_braces:
                    cleaned_json_string += "}" * (open_braces - close_braces)

            # Structural Parsing Validation Check
            try:
                payload = json.loads(cleaned_json_string)
            except json.JSONDecodeError as je:
                print(f"  ❌ Skipping Frame {filename} due to a JSON syntax decoding failure.")
                print(f"     [Raw Model Output Snippet]: {output_text[:150]}...")
                continue

            # --- ALIGN VARIANT SCHEMA KEYS NATIVELY ---
            # Map "warehouse_zone_audit" dynamically if the model outputs it instead
            if "warehouse_zone_audit" in payload and "safety_compliance_audit" not in payload:
                payload["safety_compliance_audit"] = payload["warehouse_zone_audit"]

            audit_block = payload.get("safety_compliance_audit", {})
            machinery_list = audit_block.get("machinery_active_verified", [])
            
            # Extract pedestrians presence cleanly across layout variations
            pedestrians_present = audit_block.get("pedestrians_present", False)
            if "pedestrian_traffic_verified" in audit_block and not pedestrians_present:
                # Fallback check for structural text inference variations
                if audit_block.get("pedestrian_traffic_verified") is True:
                     pedestrians_present = True

            # Handle if ppe_verified is a dictionary or boolean
            ppe_data = audit_block.get("ppe_verified", True)
            ppe_is_compliant = True
            if isinstance(ppe_data, dict):
                ppe_is_compliant = all(bool(v) for v in ppe_data.values())
            else:
                ppe_is_compliant = bool(ppe_data)
            
            # Handle inventory extraction logic dynamically based on visual elements detected
            total_items_counted = len(machinery_list) if isinstance(machinery_list, list) else 1
            if "machinery_logistics_verified" in audit_block:
                logistics = audit_block.get("machinery_logistics_verified", [])
                total_items_counted += len(logistics) if isinstance(logistics, list) else 0

            # Map high/medium risk evaluation thresholds for alert parsing
            valid_violations = []
            high_risk_count = 0
            
            if pedestrians_present:
                high_risk_count += 1
                valid_violations.append({
                    "type": "Pedestrian Hazard",
                    "risk_level": "High",
                    "description": "Active pedestrian spotted inside the forklift machinery operations sector."
                })
                
            if not ppe_is_compliant:
                valid_violations.append({
                    "type": "PPE Compliance Issue",
                    "risk_level": "Medium",
                    "description": "Visible PPE standards or protective gear deviations flagged."
                })

            # 4. Construct Ledger Entry
            ledger_entry = {
                "timestamp": frame_timestamp,
                "asset_identity": filename,
                "summary_metrics": {
                    "total_inventory_items": total_items_counted,
                    "total_violations_detected": len(valid_violations),
                    "total_compliant_indicators": 1 if ppe_is_compliant and not pedestrians_present else 0,
                    "critical_safety_breach": high_risk_count > 0
                },
                "raw_inference_payload": payload
            }

            # 5. Transactional Append into JSON ledger file
            try:
                with open(LEDGER_JSON_PATH, "r+") as f:
                    data = json.load(f)
                    data.append(ledger_entry)
                    f.seek(0)
                    json.dump(data, f, indent=2)
                    f.truncate()
            except Exception as e:
                print(f"  ⚠️ Error writing to JSON transaction volume: {str(e)}")

            # 6. Stream Active Violations directly to Flat CSV File
            if valid_violations:
                try:
                    with open(LEDGER_CSV_PATH, "a", newline="") as f:
                        writer = csv.writer(f)
                        for violation in valid_violations:
                            writer.writerow([
                                frame_timestamp,
                                filename,
                                violation.get("type", "Safety Violation"),
                                violation.get("risk_level", "Medium"),
                                violation.get("description", ""),
                                total_items_counted
                            ])
                except Exception as e:
                    print(f"  ⚠️ Error writing to CSV alert log: {str(e)}")

        except Exception as e:
            print(f"\n⚠️ Unexpected stall on frame asset processing loop: {str(e)}")
            continue

    print(f"\n✅ Throttled Batch Processing Phase Complete! Logs written out.")
    print(f" ➡️ Time-Series JSON Ledger updated: {LEDGER_JSON_PATH}")
    print(f" ➡️ Flat CSV Alert Metrics Log updated: {LEDGER_CSV_PATH}")

if __name__ == "__main__":
    main()