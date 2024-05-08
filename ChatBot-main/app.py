from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

# Check if CUDA is available
# if not torch.cuda.is_available():
#     raise RuntimeError("CUDA is not available. This script requires CUDA to run.")

# # Set the device to GPU
# device = torch.device("cuda")

# Set the model path
MODEL_PATH = "Aditi25/Instruct_on_sharded"

# Initialize Flask application
app = Flask(__name__)

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Output the device being used

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device) 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "")
    return jsonify(response=generate_answer(msg))

def generate_answer(question: str) -> str:
    # Assuming 'model' and 'tokenizer' are already loaded and 'device' is set
    prompt = f"<human>: {question}\n<assistant>:"
    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    config = GenerationConfig(
        max_length=80,  # Limits the output length
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True  # Stop when all beams generate an EOS token
    )

    with torch.inference_mode():
        outputs = model.generate(
            **encoding,
            **config.__dict__  # Pass the generation config parameters to generate
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_start = "<assistant>:"
    response_start = response.find(assistant_start) + len(assistant_start)
    return response[response_start:].strip()

if __name__ == '__main__':
    app.run()  
