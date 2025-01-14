import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
import torch
from tqdm import tqdm
from config import config

def load_training_data():
    """Load training data from JSON file"""
    with open('../data/training_data.json', 'r') as f:
        return json.load(f)

def prepare_model():
    """Initialize and prepare the model with LoRA"""
    print("Loading base model...")
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map='cpu'
    )
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    # Configure LoRA
    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    return model, tokenizer

def train():
    """Main training function"""
    print("Starting training process...")
    
    # Load data
    training_data = load_training_data()
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model()
    
    # Training loop setup
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    print("Beginning training loop...")
    for epoch in range(config['num_epochs']):
        total_loss = 0
        progress_bar = tqdm(training_data['conversations'])
        
        for conversation in progress_bar:
            # Prepare input
            input_text = f"Question: {conversation['question']}\nAnswer: {conversation['answer']}"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=config['max_length'], truncation=True)
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_description(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / len(training_data['conversations'])
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    # Save the trained model
    print("Saving model...")
    model.save_pretrained("../models/trained_erp_assistant")
    tokenizer.save_pretrained("../models/trained_erp_assistant")
    print("Training completed!")

if __name__ == "__main__":
    train()