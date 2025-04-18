# Fine-Tuning and Inference with Llama 2 (7B) on Intel® Gaudi® 2 AI Accelerator

This repository provides a comprehensive Jupyter Notebook (`llama2_fine_tuning_inference_single.ipynb`) for fine-tuning the Llama 2 7B model using Parameter Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA) and performing inference on a single Intel® Gaudi® 2 AI Accelerator. The notebook leverages the Optimum Habana library, optimized for Gaudi accelerators, and demonstrates efficient fine-tuning and text generation tasks.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Fine-Tuning Process](#fine-tuning-process)
- [Inference Process](#inference-process)
- [Comparing PEFT and Non-PEFT Inference](#comparing-peft-and-non-peft-inference)
- [Directory Structure](#directory-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview
This notebook demonstrates:
1. **Fine-Tuning**: Fine-tuning the Llama 2 7B model using PEFT with LoRA on the `timdettmers/openassistant-guanaco` dataset. PEFT reduces computational and memory requirements by fine-tuning only a small subset of parameters (0.06% of the total 7B parameters).
2. **Inference**: Generating text using the fine-tuned model with a user-provided prompt and comparing the results with a non-fine-tuned model.
3. **Optimization**: Utilizing Intel® Gaudi® 2 AI Accelerator and the Optimum Habana library for efficient training and inference.

The fine-tuning process takes approximately 20 minutes for 3 epochs, and the inference task generates detailed text outputs, showcasing the benefits of PEFT.

## Prerequisites
- **Hardware**: Intel® Gaudi® 2 AI Accelerator.
- **Software**:
  - Intel® Gaudi® software version 1.15 (note: newer versions may not be compatible).
  - Python 3.10.
  - Jupyter Notebook.
- **Hugging Face Account**:
  - A Hugging Face account with access to the Llama 2 model (requires agreeing to the terms of use on the model card).
  - A Hugging Face read token for gated model access.
- **Dependencies**:
  - `peft==0.10.0`
  - `optimum-habana==1.11.1`
  - Other dependencies listed in the `requirements.txt` files for language modeling and text generation tasks.

## Setup Instructions
1. **Clone the Repository**:
   Ensure you are in the desired working directory and clone the Optimum Habana repository:
   ```bash
   cd ~/Gaudi-tutorials/PyTorch/Single_card_tutorials
   git clone -b v1.11.1 https://github.com/huggingface/optimum-habana.git
   ```

2. **Install Dependencies**:
   Install the required Python packages:
   ```bash
   pip install peft==0.10.0
   pip install -q optimum-habana==1.11.1
   cd ~/Gaudi-tutorials/PyTorch/Single_card_tutorials/optimum-habana/examples/language-modeling
   pip install --upgrade pip
   pip install -q -r requirements.txt
   ```

3. **Hugging Face Login**:
   Log in to Hugging Face to access the Llama 2 model:
   ```bash
   huggingface-cli login --token <your_huggingface_token>
   ```

4. **Directory Setup**:
   Ensure the working directory is set to:
   ```bash
   ~/Gaudi-tutorials/PyTorch/Single_card_tutorials/optimum-habana/examples/language-modeling
   ```

## Fine-Tuning Process
The fine-tuning process uses the `run_lora_clm.py` script from the Optimum Habana library to fine-tune the Llama 2 7B model with PEFT and LoRA. Key features include:
- **Dataset**: `timdettmers/openassistant-guanaco`.
- **Model**: `VAGOsolutions/Llama-3-SauerkrautLM-70b-Instruct` (used as a substitute in the notebook).
- **Parameters**:
  - LoRA rank: 8
  - LoRA alpha: 16
  - LoRA dropout: 0.05
  - Target modules: `q_proj`, `v_proj`
  - Batch size: 16
  - Epochs: 3
  - Learning rate: 1e-4
  - Maximum sequence length: 512
- **Output**: The fine-tuned model is saved in `~/Gaudi-tutorials/PyTorch/Single_card_tutorials/2025`.

Run the fine-tuning command:
```bash
python3 run_lora_clm.py \
    --overwrite_output_dir=True \
    --model_name_or_path VAGOsolutions/Llama-3-SauerkrautLM-70b-Instruct \
    --dataset_name timdettmers/openassistant-guanaco \
    --bf16 True \
    --output_dir ~/Gaudi-tutorials/PyTorch/Single_card_tutorials/2025 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 1e-4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant" \
    --max_grad_norm 0.3 \
    --logging_steps 1 \
    --do_train \
    --do_eval \
    --use_habana \
    --use_lazy_mode \
    --throughput_warmup_steps 3 \
    --lora_rank=8 \
    --lora_alpha=16 \
    --lora_dropout=0.05 \
    --lora_target_modules "q_proj" "v_proj" \
    --dataset_concatenation \
    --report_to none \
    --max_seq_length 512 \
    --trust_remote_code=True \
    --low_cpu_mem_usage True \
    --validation_split_percentage 4 \
    --adam_epsilon 1e-08
```

Upon completion, a folder named `model_lora_llama_single` is created, containing the `adapter_model.bin` file for the PEFT model.

## Inference Process
The inference process uses the `run_generation.py` script from the `text-generation` task to generate text based on a user-provided prompt. The fine-tuned PEFT model is loaded to enhance the output quality.

1. **Move to Text Generation Directory**:
   ```bash
   cd ~/Gaudi-tutorials/PyTorch/Single_card_tutorials/optimum-habana/examples/text-generation
   pip install -q -r requirements.txt
   ```

2. **Run Inference with PEFT**:
   Provide a prompt and run the inference command:
   ```bash
   prompt="I am a dog. Please help me plan a surprise birthday party for my human, including fun activities, games and decorations. And don't forget to order a big bone-shaped cake for me to share with my fur friends!"
   python3 run_generation.py \
       --model_name_or_path VAGOsolutions/Llama-3-SauerkrautLM-70b-Instruct \
       --batch_size 1 \
       --do_sample \
       --max_new_tokens 300 \
       --n_iterations 4 \
       --use_hpu_graphs \
       --use_kv_cache \
       --bf16 \
       --prompt "${prompt}" \
       --peft_model ~/Gaudi-tutorials/PyTorch/Single_card_tutorials/model_one
   ```

## Comparing PEFT and Non-PEFT Inference
To highlight the benefits of fine-tuning, the notebook includes a comparison by running inference without the PEFT model:
```bash
python3 run_generation.py \
    --model_name_or_path VAGOsolutions/Llama-3-SauerkrautLM-70b-Instruct \
    --batch_size 1 \
    --do_sample \
    --max_new_tokens 300 \
    --n_iterations 4 \
    --use_hpu_graphs \
    --use_kv_cache \
    --bf16 \
    --prompt "${prompt}"
```

The PEFT model typically produces more detailed and contextually relevant outputs compared to the non-fine-tuned model, which may lack specificity and fidelity.

## Directory Structure
After running the notebook, the directory structure will include:
```
~/Gaudi-tutorials/PyTorch/Single_card_tutorials/
├── optimum-habana/
│   ├── examples/
│   │   ├── language-modeling/
│   │   │   ├── run_lora_clm.py
│   │   │   ├── requirements.txt
│   │   │   └── model_lora_llama_single/
│   │   │       └── adapter_model.bin
│   │   ├── text-generation/
│   │   │   ├── run_generation.py
│   │   │   └── requirements.txt
├── 2025/
│   └── (fine-tuned model outputs)
└── llama2_fine_tuning_inference_single.ipynb
```

## Troubleshooting
- **Hugging Face Token Issues**: Ensure you have a valid Hugging Face token and have agreed to the Llama 2 model terms.
- **Dependency Errors**: Verify that all dependencies are installed correctly and match the specified versions (e.g., `peft==0.10.0`, `optimum-habana==1.11.1`).
- **Gaudi Software Version**: The notebook is optimized for Intel® Gaudi® software version 1.15. Newer versions may cause compatibility issues.
- **CUDA Errors**: If you encounter CUDA-related errors (e.g., `CUDA_HOME not set`), note that this notebook is designed for Gaudi accelerators, not GPUs. Ensure the environment is configured for Gaudi.
- **Model Access**: If the model fails to load, verify that the `VAGOsolutions/Llama-3-SauerkrautLM-70b-Instruct` repository is accessible and your token has the necessary permissions.

## License
This notebook is licensed under the Apache License, Version 2.0. See the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) file for details.

**Llama 2 Usage**: Use of the Llama 2 model is subject to the Llama 2 Community License Agreement (LLAMAV2). Review the terms at [https://ai.meta.com/llama/license/](https://ai.meta.com/llama/license/). Users are responsible for compliance with third-party licenses.

## Acknowledgments
- **Habana Labs, an Intel Company**: For providing the Gaudi 2 AI Accelerator and the Optimum Habana library.
- **Hugging Face**: For the PEFT library, Optimum Habana integration, and the `timdettmers/openassistant-guanaco` dataset.
- **Meta AI**: For the Llama 2 model.