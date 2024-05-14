from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftConfig, PeftModel

app = Flask(__name__)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")  # Output the device being used

# Load the model and tokenizer
PEFT_MODEL = "Aditi25/results_HPE"
#Aditi25/ShardedonOurData

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
config = PeftConfig.from_pretrained(PEFT_MODEL)
peft_base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
)
peft_model = PeftModel.from_pretrained(peft_base_model, PEFT_MODEL)
peft_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
peft_tokenizer.pad_token = peft_tokenizer.eos_token

# Define the endpoint to serve the HTML file
@app.route('/')
def index():
    return render_template('chat.html')

# Define the endpoint to handle queries
@app.route('/get', methods=['POST'])
def chat():
    msg = request.form.get("msg", "")
    return jsonify(response=get_answer(msg))

def get_answer(question: str) -> str:
    final_prompt = f"<human>: {question}\n<assistant>:"
    peft_encoding = peft_tokenizer(final_prompt, return_tensors="pt").to(device)
    peft_outputs = peft_model.generate(
        input_ids=peft_encoding.input_ids,
        max_length=80,  
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True  
    )
    peft_text_output = peft_tokenizer.decode(peft_outputs[0], skip_special_tokens=True)
    start_token = "<assistant>:"
    
    # Find the response after the start token
    start_idx = peft_text_output.find(start_token)
    if start_idx != -1:
        response = peft_text_output[start_idx + len(start_token):].strip()
        
        # Remove additional text such as "<human>:" and everything after it
        end_idx = response.find("human")
        if end_idx != -1:
            response = response[:end_idx].strip()
    else:
        response = "No answer found."
    
    return response


if __name__ == '__main__':
    app.run()
