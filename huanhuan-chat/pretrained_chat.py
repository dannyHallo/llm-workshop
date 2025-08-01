from modelscope import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

model_name_or_path = "下载好的嬛嬛模型地址"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
model.eval()  

response, history = model.chat(tokenizer, '你好', meta_instruction=='现在你要扮演皇帝身边的女人--甄嬛', history=[])
print(response)
