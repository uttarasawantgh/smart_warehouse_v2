import torch
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image

BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_DIR = "./models/amd_warehouse_qwen_lora"
TEST_IMAGE_PATH = "./data/raw/Logistics-2/test/3-Wheel20Electric1200X800_jpg.rf.33aaa03178981b6f809b8b59b096deb7.jpg"

if not os.path.exists(TEST_IMAGE_PATH):
    raise FileNotFoundError(f"Could not locate the test image at: {TEST_IMAGE_PATH}")

print("📡 Loading base multi-modal layers...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda:0", 
    attn_implementation="sdpa"
)

print("🔗 Merging custom smart warehouse LoRA adapters...")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
processor = AutoProcessor.from_pretrained(BASE_MODEL)

# Explicit target prompt matching the training format
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": TEST_IMAGE_PATH},
            {"type": "text", "text": "Perform a comprehensive safety and compliance audit on this warehouse zone. Return the results strictly as a valid JSON object matching the training schema."}
        ]
    }
]

print("🎨 Preprocessing image and prompt tokens...")
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
raw_image = Image.open(TEST_IMAGE_PATH).convert("RGB")
inputs = processor(text=[text], images=[raw_image], padding=True, return_tensors="pt").to("cuda:0")

print("🔥 Generating inference prediction on Instinct hardware...")
with torch.no_grad():
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=512,
        repetition_penalty=1.2,          # Snaps the model out of the duplicate structural loop
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id
    )
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print("\n📋 Custom Model Inspection Audit Report:")
print("=" * 60)
print(output_text[0])
print("=" * 60)