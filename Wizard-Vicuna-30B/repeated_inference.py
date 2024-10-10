from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("./Wizard-Vicuna-30B-Uncensored-GPTQ", device_map='cuda')
model = AutoModelForCausalLM.from_pretrained("./Wizard-Vicuna-30B-Uncensored-GPTQ", device_map='cuda')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

system = "You are a professor of psychology who eagerly answers the user's questions in depth using context given"
input_text = "SYSTEM: " + system
in_length = 0

def inference(input_text):
    global in_length
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    in_length = len(input_text)
    outputs = model.generate(**inputs, max_length = 4000)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def output(generated_text):
    global in_length
    print("ASSISTANT: " + generated_text[in_length+1:])
    print("\n")



print("\n" * 50)    

while True:
    user = input("USER: ")
    print("\n")
    input_text += " USER:" + user + " ASSISTANT: "
    generated_text = inference(input_text)
    output(generated_text)
    input_text = generated_text