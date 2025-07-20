import torch


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