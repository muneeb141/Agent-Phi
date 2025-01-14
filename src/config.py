config = {
    "model_name": "microsoft/phi-1.5",
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "learning_rate": 0.0001,
    "batch_size": 2,  # Small batch for CPU
    "num_epochs": 20,
    "max_length": 256,
    "device_map": "cpu",
    "torch_dtype": "float32"  # Use float32 for CPU
}