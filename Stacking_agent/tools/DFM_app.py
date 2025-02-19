from flask import Flask, request, jsonify
from transformers import AutoTokenizer
import os
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
app = Flask(__name__)

model_name_or_id = "./ChemDFM/ChemDFM-v1.0-13B"
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_id)
model = LlamaForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, device_map="auto")
def get_response(input_text):
    input_text = '"Please give me the answers of the following question:'+input_text
    input_text = f"[Round 0]\nHuman: {input_text}\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    generation_config = GenerationConfig(
        do_sample=True,
        top_k=20,
        top_p=0.9,
        temperature=0.9,
        max_new_tokens=1024,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id
    )

    outputs = model.generate(**inputs, generation_config=generation_config)
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(input_text):]
    return generated_text.strip()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    input_text = data['input_text']
    generated_text = get_response(input_text)
    return [generated_text]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4040)
