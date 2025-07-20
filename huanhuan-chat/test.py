from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
import os
from utils import display_gpu_info

display_gpu_info()

model_id = "LLM-Research/Meta-Llama-3-8B-Instruct"
model_path = os.path.join(os.path.expanduser('~'), '.cache', 'modelscope', 'hub', model_id)

lora_path = './output/llama3_1_instruct_lora/checkpoint-699' # 这里改称你的 lora 输出对应 checkpoint 地址

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

def chat_with_huanhuan():
    print("\n" + "="*50)
    print("与甄嬛对话 - 输入 'exit' 或 'quit' 退出")
    print("="*50)
    
    while True:
        try:
            user_input = input("\n皇上：").strip()
            
            if user_input.lower() in ['exit', 'quit', '退出']:
                print("告辞...")
                break
                
            if not user_input:
                continue
                
            messages = [
                {"role": "system", "content": "假设你是皇帝身边的女人--甄嬛。"},
                {"role": "user", "content": user_input}
            ]
            
            input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')
            generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print('嬛嬛：', response)
            
        except KeyboardInterrupt:
            print("\n告辞...")
            break
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    chat_with_huanhuan()
