from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_assistant():
    print("Loading the trained assistant...")
    model = AutoModelForCausalLM.from_pretrained(
        "../models/trained_erp_assistant",
        torch_dtype=torch.float32,
        device_map='cpu'
    )
    tokenizer = AutoTokenizer.from_pretrained("../models/trained_erp_assistant")
    return model, tokenizer

def generate_response(question, model, tokenizer):
    # Create a more structured prompt
    input_text = (
        "Below is a conversation about an ERP system. "
        "Provide clear and specific answers about ERP functionality.\n\n"
        f"User: {question}\n"
        "Assistant:"
    )
    
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # Adjust generation parameters
    outputs = model.generate(
        **inputs,
        max_length=250,  # Increase max_length for longer responses
        min_length=20,
        num_return_sequences=1,
        temperature=0.5,  # Reduced for more focused responses
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Clean up the response
    response = response.split("Assistant:")[-1].strip()
    return response

def main():
    model, tokenizer = load_assistant()
    print("\nERP Assistant is ready! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() == 'quit':
            break
        
        try:
            response = generate_response(question, model, tokenizer)
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"Error generating response: {e}")

if __name__ == "__main__":
    main()