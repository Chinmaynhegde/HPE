from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftConfig, PeftModel

app = Flask(__name__)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")  # Output the device being used

# Load models and tokenizers for both models
PEFT_MODEL_1 = "Aditi25/Sharded-FullHPCMdata_latest"
PEFT_MODEL_2 = "Aditi25/Instruct_copy_results_latest"

# Load configuration and models for PEFT_MODEL_1
bnb_config_1 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
config_1 = PeftConfig.from_pretrained(PEFT_MODEL_1)
peft_base_model_1 = AutoModelForCausalLM.from_pretrained(
    config_1.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config_1,
    device_map="auto",
)
peft_model_1 = PeftModel.from_pretrained(peft_base_model_1, PEFT_MODEL_1)
peft_tokenizer_1 = AutoTokenizer.from_pretrained(config_1.base_model_name_or_path)
peft_tokenizer_1.pad_token = peft_tokenizer_1.eos_token



# Load configuration and models for PEFT_MODEL_2
bnb_config_2 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
config_2 = PeftConfig.from_pretrained(PEFT_MODEL_2)
peft_base_model_2 = AutoModelForCausalLM.from_pretrained(
    config_2.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config_2,
    device_map="auto",
    trust_remote_code=True,

)
peft_model_2 = PeftModel.from_pretrained(peft_base_model_2, PEFT_MODEL_2)
peft_tokenizer_2 = AutoTokenizer.from_pretrained(config_2.base_model_name_or_path)
peft_tokenizer_2.pad_token = peft_tokenizer_2.eos_token



@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['POST'])
def chat():
    msg = request.form.get("msg", "")
    model_choice = request.form.get("model", "model1")  
    if model_choice == "model1":
        model_name = "Sharded Model"
        response = get_answer_sharded(msg, peft_model_1, peft_tokenizer_1)
        # response =""
    else:
        model_name = "Instruct Model"
        response = get_answer_instruct(msg, peft_model_2, peft_tokenizer_2)
        # response = ""
    
    response = f"{model_name} results: {response}"
    return jsonify(response=response)


def get_answer_sharded(query: str,peft_model_1, peft_tokenizer_1)-> str:
  system_prompt = """Answer the following question truthfully.
  If you don't know the answer, respond 'Sorry, I don't know the answer to this question.'.
  If the question is too complex, respond 'Kindly, consult the documentation for further queries.'."""

  user_prompt = f"""###HUMAN: {query}
  ###ASSISTANT: """

  final_prompt = system_prompt + "\n" + user_prompt

  device = "cuda:0"
  dashline = "-".join("" for i in range(50))


  peft_encoding = peft_tokenizer_1(final_prompt, return_tensors="pt").to(device)
  peft_outputs = peft_model_1.generate(input_ids=peft_encoding.input_ids, 
                                     generation_config=GenerationConfig(max_new_tokens=200, pad_token_id = peft_tokenizer_1.eos_token_id, \
                                                                                                                     eos_token_id = peft_tokenizer_1.eos_token_id, attention_mask = peft_encoding.attention_mask, \
                                                                                                                     temperature=0.7, top_p=0.7, repetition_penalty=1.3, num_return_sequences=1,))
  peft_text_output = peft_tokenizer_1.decode(peft_outputs[0], skip_special_tokens=True)

  start_token = "###ASSISTANT:"
  end_token = "##"

  start_idx = peft_text_output.find(start_token)
  end_idx = peft_text_output.find(end_token, start_idx + len(start_token))

  if start_idx != -1 and end_idx != -1:
      return peft_text_output[start_idx + len(start_token):end_idx].strip()
  else:
    return "No answer found."



def get_answer_instruct(question: str, model, tokenizer) -> str:
    prompt = f"""
<human>: {question}
<assistant>:
""".strip()
    encoding = tokenizer(prompt, return_tensors="pt").to(device)

    generation_config = GenerationConfig()
    generation_config.max_new_tokens = 200
    generation_config.temperature = 0.7
    generation_config.top_p = 0.7
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    assistant_start = "<assistant>:"
    response_start = response.find(assistant_start)
    if response_start != -1:
        response = response[response_start + len(assistant_start):].strip()
    
    if response.endswith("User"):
        response = response[:-len("User")].strip()
    return response



if __name__ == '__main__':
    app.run()




