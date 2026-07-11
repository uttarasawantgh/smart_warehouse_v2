import torch
import os
import json
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def main():
    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    DATASET_PATH = "./data/processed/warehouse_train_ready.json"
    OUTPUT_DIR = "./models/amd_warehouse_qwen_lora"

    print("=" * 60)
    print("🚀 STARTING NATIVE AMD FINE-TUNING PIPELINE")
    print(f"Base Model: {MODEL_ID}")
    print("=" * 60)

    # 1. Load the native Qwen2.5-VL Processor
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # 2. Load base model layers explicitly onto your active AMD device
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0", 
        attn_implementation="sdpa"
    )

    # 3. Enable gradient checkpointing to safeguard VRAM boundaries
    model.gradient_checkpointing_enable()

    # 4. Configure LoRA targeted to Qwen's attention & multi-modal projections
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Load the locally processed dataset
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # 6. Standard Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        num_train_epochs=3,
        bf16=True,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False # Crucial for keeping multi-modal image inputs intact
    )

    # 7. Initialize the Trainer without conflicting text-packing parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )

    # 8. Execute the compute graph
    print("\n🔥 Commencing training loops on ROCm compute blocks...")
    trainer.train()

    # 9. Save the locally baked adapter arrays
    print(f"\n✅ Training complete! Saving custom adapters locally to: {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()