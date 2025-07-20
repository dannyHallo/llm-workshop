from datasets import Dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model
import os

def display_gpu_info(device_id=0):
    """
    Displays detailed information about the specified CUDA-enabled GPU.

    This function checks for CUDA availability and, if present, prints a
    formatted table of the key properties of the GPU, including its name,
    memory, and compute capability.

    Args:
        device_id (int): The index of the GPU device to inspect. Defaults to 0.
    """
    print("=" * 50)
    print("PyTorch CUDA Environment Check")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("❌ FAILURE: CUDA is not available on this system.")
        print("PyTorch cannot detect any compatible NVIDIA GPU.")
        print("Please check your NVIDIA driver installation and PyTorch build.")
        print("=" * 50)
        return

    num_gpus = torch.cuda.device_count()
    if device_id >= num_gpus:
        print(f"❌ FAILURE: Invalid device_id. You requested device {device_id}, "
              f"but only {num_gpus} GPUs are available.")
        print("=" * 50)
        return

    gpu_name = torch.cuda.get_device_name(device_id)
    print(f"✅ SUCCESS: CUDA is available. Found {num_gpus} GPU(s).")
    print(f"Displaying information for device {device_id}: {gpu_name}\n")

    try:
        # Get device properties
        props = torch.cuda.get_device_properties(device_id)

        # Format information into a dictionary for clean printing
        info = {
            "Device Name": props.name,
            "Compute Capability": f"{props.major}.{props.minor}",
            "Total Memory": f"{props.total_memory / (1024**3):.2f} GiB",
            "Multiprocessors (SMs)": props.multi_processor_count,
            "Memory Clock Rate": f"{props.memory_clock_rate / 1000:.2f} GHz",
            "CUDA Core Count": get_cuda_core_count(props.name, props.multi_processor_count)
        }

        # Find the longest key for alignment
        max_key_len = max(len(key) for key in info.keys())

        # Print formatted information
        for key, value in info.items():
            print(f"{key:<{max_key_len}} : {value}")

    except Exception as e:
        print(f"An error occurred while fetching properties for device {device_id}: {e}")

    print("=" * 50)


def get_cuda_core_count(gpu_name, sm_count):
    """
    Estimates the number of CUDA cores based on the GPU architecture.
    This is an approximation as PyTorch doesn't provide a direct API.
    """
    gpu_name = gpu_name.lower()
    # Cores per Streaming Multiprocessor (SM) for different architectures
    cores_per_sm = {
        "turing": 64,      # RTX 20xx, T4
        "ampere": 128,     # RTX 30xx, A100
        "ada lovelace": 128, # RTX 40xx
        "volta": 64,       # V100
        "pascal": 128,     # P100, GTX 10xx
        "maxwell": 128,    # M40, GTX 9xx
        "kepler": 192      # K80
    }
    
    # Heuristics to detect architecture from name
    if "rtx 40" in gpu_name or "l4" in gpu_name:
        arch = "ada lovelace"
    elif "rtx 30" in gpu_name or "a100" in gpu_name or "a10g" in gpu_name:
        arch = "ampere"
    elif "rtx 20" in gpu_name or " t4" in gpu_name:
        arch = "turing"
    elif "v100" in gpu_name:
        arch = "volta"
    elif "p100" in gpu_name or "gtx 10" in gpu_name:
        arch = "pascal"
    else:
        return "Not Available (Unknown Arch)"
        
    return sm_count * cores_per_sm[arch]

def train():
    def process_func(example):
        MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
        input_ids, attention_mask, labels = [], [], []
        instruction = tokenizer(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n现在你要扮演皇帝身边的女人--甄嬛<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
        response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
        if len(input_ids) > MAX_LENGTH:  # 做一个截断
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


    model_id = "LLM-Research/Meta-Llama-3-8B-Instruct"
    model_path = os.path.join(os.path.expanduser('~'), '.cache', 'modelscope', 'hub', model_id)

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16)
    model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 将JSON文件转换为CSV文件
    df = pd.read_json('huanhuan.json')
    ds = Dataset.from_pandas(df)
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False, # 训练模式
        r=8, # Lora 秩
        lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1# Dropout 比例
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters() # 打印总训练参数

    args = TrainingArguments(
        output_dir="./output/llama3_1_instruct_lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100, # 为了快速演示，这里设置10，建议你设置成100
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train() # 开始训练 
    # 在训练参数中设置了自动保存策略此处并不需要手动保存。


if __name__ == "__main__":
    display_gpu_info()
    train()
