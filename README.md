# 🏭 AI Smart Warehouse Safety & Audit System

A Multi-Track Physical AI Compliance Dashboard utilizing vision-language models (VLMs) to automate equipment safety, inventory tracking, and regulatory compliance. This project serves as a technical benchmark demonstrating the performance and throughput advantages of running localized inference on dedicated hardware versus public serverless cloud infrastructure.

---

## 🚀 Project Links & Presentation Assets

* **🌐 Live Interactive Dashboard:** [Hugging Face Space Live App](https://huggingface.co/spaces/uttarasawant/smartwarehouse) *(Direct Fullscreen App: `https://uttarasawant-smartwarehouse.hf.space`)*
* **📊 Project Slide Deck:** [View the Presentation Slides](https://huggingface.co/spaces/uttarasawant/smartwarehouse/resolve/main/AI%20Smart%20Warehouse%20Safety%20%26%20Audit%20System.pdf)
* **🎬 Video Demonstration:** [Watch the Technical Walkthrough](https://youtu.be/QI2QIqxWNDY)
* **🎬 Core logic:** [Visit github link](https://github.com/uttarasawantgh/smart_warehouse_v2)
* **🎬 UI logic:** [Visit github link](https://github.com/uttarasawantgh/smart_warehouse_ui)

---

## ⚡ Hardware Telemetry & Benchmarking Highlights

This system processes incoming industrial facility feeds under two distinct infrastructure profiles to showcase massive optimization:

| Metric | Local AMD Instinct Compute (PyTorch ROCm) | Fireworks AI Serverless (Cloud API) |
| :--- | :--- | :--- |
| **Avg Frame Latency** | **0.047s** | 1.465s (Inc. Network Transport) |
| **Token Throughput** | **2,962.24 tk/s** (~60x Increase) | 48.56 tk/s |
| **Total VLM Tokens** | **136,122 VLM Tokens** (Raw Uncompressed) | ~22,400 Tokens (Payload Compressed) |
| **Marginal Infrastructure Cost** | **$0.00** (Fixed/Air-gapped Bare-Metal) | Metered Pay-Per-Token ($1.45) |

### Key Architectural Takeaways for Judges:
1. **60x Throughput Advantage:** Running locally via optimized ROCm bindings fully saturates local VRAM memory channels, bypassing the internet transport bottlenecks of streaming heavy base64 strings to a public API endpoint.
2. **Zero Marginal Operational Cost:** Local deployment allows the system to process massive token densities with $0.00 ongoing infrastructure costs, compared to volatile metered cloud usage.

---

## 🛠️ Repository & System Structure

* `download_data.py` - Code to download Roboflow dataset common functionality for AMD and FireworksAI platforms
* `prepare_dataset_amd.py` / `_fireworks.py` - Convert COCO JSON format to VLM format and copy output into data/processed folder
* `upload_to_fireworks.py` - Upload warehouse_train.jsonl output from prepare_dataset into fireworksAI
* `download_weights_fireworks.py` -  After fine-tuning in FireworksAI download fine-tuned weights locally
* `engine_fireworks_qwen.py` - Inference engine for fireworksAI
* `train_amd_native.py` - Use warehouse_train_ready.json output from prepare_dataset for AMD and use it to fine-tune model save output into models/amd_warehouse_qwen_lora folder
* `inference_amd.py` - Run inference on AMD using test image
* `batch_audit_amd.py` - Run through all test images through fine-tuned weights on AMD and save into compliance_alerts_log_amd.csv and compliance_time_series_ledger_amd.json
* `run_fireworks_batch.py` - Run through all test images through fine-tuned weights on FireworksAI and save into compliance_alerts_log_fireworks.csv and compliance_time_series_ledger_fireworks.json
* `run_fireworks_benchmark.py` - Calculate benchmark data for fireworksAI inferencing
* `extract_amd_telemetry.py` - Using compliance_alerts_log_amd.csv and compliance_time_series_ledger_amd.json timestamps capture tentative benchmark similar to fireworksAI
* `compliance_alerts_log_amd.csv` / `_fireworks.csv` - The parsed operational risk matrices displayed in Tab 1.
* `compliance_time_series_ledger_amd.json` / `_fireworks.json` - The multi-ledger data analytics mapping used for the interactive timeline trends in Tab 2.
* FireworksAI fine-tuning activity is done using ID accounts/<myemail>-k6jmmt/supervisedFineTuningJobs/alycv3uw and deployment job is created using ID accounts/<myemail>-k6jmmt/deployments/wavb1x6c
* Create .env file in your local directory and set following variables
* FIREWORKS_ACCOUNT_ID
* FIREWORKSAI_API_KEY
* ROBOFLOW_API_KEY
* HF_TOKEN

### 💰 Resource Management & Financial Efficiency

| Category | Initial Allocation | Remaining Balance |
| :--- | :---: | :---: |
| **AMD Developer Cloud** | $105.00 | $1.02 |
| **Fireworks AI** | $56.00 | $11.04 |
| **Total Project Credit** | **$161.00** | **$12.06** |
