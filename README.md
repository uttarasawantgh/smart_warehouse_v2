ROBOFLOW_API_KEY=your_key_here

FIREWORKS_API_KEY=your_key_here

HF_TOKEN=your_key_here

FIREWORKSAI_API_KEY=your_key_here


"Ensure your fine-tuned model checkpoint is located in models/checkpoint/ or set the DYNAMIC_ADAPTER_DIR environment variable to point to your specific path."

# 🏭 AI Smart Warehouse Safety & Audit System

A Multi-Track Physical AI Compliance Dashboard utilizing vision-language models (VLMs) to automate equipment safety, inventory tracking, and regulatory compliance. This project serves as a technical benchmark demonstrating the performance and throughput advantages of running localized inference on dedicated hardware versus public serverless cloud infrastructure.

## 🚀 Project Links & Presentation Assets

* **🌐 Live Interactive Dashboard:** [Hugging Face Space Live App](https://huggingface.co/spaces/uttarasawant/smartwarehouse) *(Direct Fullscreen App: `https://uttarasawant-smartwarehouse.hf.space`)*
* **📊 Project Slide Deck:** [View the Presentation Slides](https://huggingface.co/spaces/uttarasawant/smartwarehouse/resolve/main/AI%20Smart%20Warehouse%20Safety%20%26%20Audit%20System.pdf)
* **🎬 Video Demonstration:** [Watch the Technical Walkthrough](https://youtu.be/eTtG0kP_vwc)
* **🎬 Core logic:** [Visit github link](https://github.com/uttarasawantgh/smart_warehouse_new)
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

* `download_data.py` - Downloads roboflow dataset for warehouse equipment using ROBOFLOW_API key into local data/raw into COCO JSON format. This is common to AMD and FireworksAI platform.
* `prepare_dataset.py` - Converts a local image file into a Base64 data URI string for Fireworks VLM ingest warehouse_train_ready.jsonl and will be used to upload to fireworksAI
*  `prepare_dataset_amd.py` - Converts a local image file into a Base64 data URI string for Fireworks VLM ingest warehouse_train_ready.jsonl and will be used to upload to fireworksAI


