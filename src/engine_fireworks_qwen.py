# engine_fireworks_qwen.py
import os
import json
import base64
from openai import OpenAI

# --- CREDENTIAL CONFIGURATION ---
FIREWORKSAI_API_KEY = os.environ.get("FIREWORKSAI_API_KEY")

# --- CUSTOM FINE-TUNED MODEL ID ---
# Formatted as accounts/{account_id}/models/{fine_tuned_model_id}
ACCOUNT_ID = os.getenv("FIREWORKS_ACCOUNT_ID", "default_placeholder")
CUSTOM_MODEL_ID = f"accounts/{ACCOUNT_ID}/deployments/wavb1x6c"

if not FIREWORKSAI_API_KEY:
    print("⚠️ WARNING: 'FIREWORKSAI_API_KEY' environment variable not detected.")
else:
    print(f"📡 Serving Serverless via Fireworks AMD Clusters")
    print(f"🎯 Target Fine-Tuned Model: {CUSTOM_MODEL_ID}")

# Initialize OpenAI-compatible network client
client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=FIREWORKSAI_API_KEY
)

def _encode_local_image_to_base64(image_path):
    """
    Converts local image files into Base64 URI strings for network delivery.
    """
    ext = os.path.splitext(image_path)[1].lower().replace(".", "")
    mime_type = "jpeg" if ext in ["jpg", "jpeg"] else "png"
    
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        
    return f"data:image/{mime_type};base64,{encoded_string}"


def analyze_warehouse_frame(image_path):
    """
    Routes warehouse frames straight through your custom fine-tuned Qwen 
    weights serverless on Fireworks AI infrastructure.
    """
    if not os.path.exists(image_path):
        return {"error": f"Target image asset not found at: {image_path}"}

    try:
        # Convert local image asset path to network data stream
        base64_image_uri = _encode_local_image_to_base64(image_path)
        
        # System/User message payload structure
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": (
                            "You are an AI Warehouse Operations and Safety Auditor. Analyze the provided camera feed frame "
                            "and return a strictly valid JSON document containing both structural inventory tracking and life-safety compliance analysis. "
                            "Analyze this warehouse zone for safety and inventory metrics."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image_uri
                        }
                    }
                ]
            }
        ]
        
        # Dispatch the request straight to your dedicated LoRA checkpoint
        response = client.chat.completions.create(
            model=CUSTOM_MODEL_ID,
            messages=messages,
            temperature=0.1,
            max_tokens=1024
        )
        
        output_text = response.choices[0].message.content.strip()
        
        # Clean Markdown wrappers if present
        if output_text.startswith("```"):
            lines = output_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned_text = "\n".join(lines).strip()
        else:
            cleaned_text = output_text

        # Return the structured JSON document
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            return {
                "warning": "Fine-tuned model output deviated from structured format rules.",
                "raw_output": output_text
            }
            
    except Exception as e:
        return {"error": f"Fireworks fine-tune serverless execution failure: {str(e)}"}