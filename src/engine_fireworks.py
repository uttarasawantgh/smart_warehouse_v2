import os
import json
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# --- LOCAL HARDWARE CONFIGURATION ---
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Fall back to your default path if the environment variable isn't set
ADAPTER_DIR = os.getenv(
    "DYNAMIC_ADAPTER_DIR", 
    "/workspace/models/ft-rotjdido29sbo/tuned-model-alycv3uw/9898da/ft-rotjdido29sbo/checkpoint"
)

print("📡 Loading foundational Qwen2.5-VL-7B base weights...")
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"  # Let Hugging Face map cleanly to the GPU natively
)

print("🎛️ Injecting custom fine-tuned warehouse compliance adapters...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval() # Freeze model weights into evaluation state

print("🧩 Initializing image processing pipeline...")
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
print("✅ Local  Multi-Modal Inference Engine Ready!")

def analyze_warehouse_frame(image_path):
    """
    Processes a local image through your custom local model layers
    and extracts structured compliance/inventory metrics.
    """
    if not os.path.exists(image_path):
        return {"error": f"Target image asset not found at: {image_path}"}

    try:
        # Load local image asset directly via PIL instead of Base64 strings
        image = Image.open(image_path).convert("RGB")
        
        # Build conversational structural logic
        messages = [
            {
                "role": "system",
                "content": "You are an AI Warehouse Operations and Safety Auditor. Analyze the provided camera feed frame and return a strictly valid JSON document containing both structural inventory tracking and life-safety compliance analysis."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Analyze this warehouse zone for safety and inventory metrics."}
                ]
            }
        ]
        
        # Format the textual pipeline for Qwen VL compatibility
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Compile multi-modal visual tokens into execution tensor arrays
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to("cuda") # Run computation instantly inside your GPU core array
        
        # Generate token distribution array
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            
        # Strip generation prompts to harvest text string
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0].strip()
        
        # --- FIX: Clean Markdown formatting if the model wraps the JSON ---
        if output_text.startswith("```"):
            lines = output_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned_text = "\n".join(lines).strip()
        else:
            cleaned_text = output_text

        # Convert cleaned text block back to structured Python dictionary layout safely
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            # Fallback wrapper to catch raw output if parsing still fails
            return {
                "warning": "Model output was not perfectly structured JSON",
                "raw_output": output_text
            }
        
    except Exception as e:
        return {"error": f"Local hardware execution failure: {str(e)}"}