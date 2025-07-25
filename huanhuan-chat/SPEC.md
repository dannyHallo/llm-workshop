# 项目环境准备指南

## Conda 与依赖

```shell
# 创建并激活虚拟环境
conda create -n huanhuan python=3.12 -y
conda activate huanhuan

# 安装 Python 依赖
pip install -r requirements.txt
```

## 预训练模型下载与使用

### 下载模型

```shell
# Meta-Llama-3 8B Instruct
modelscope download --model LLM-Research/Meta-Llama-3-8B-Instruct

# 嬛嬛 Chat InternLM2
modelscope download --model kmno4zx/huanhuan-chat-internlm2
```

下载的模型会保存在 `~/.cache/modelscope/hub/` 目录下。

### 推理测试

```shell
python pretrained_chat.py
```

---

## 微调（Finetuning）

```shell
python train.py  # 基于 pretrained model 与 huanhuan.json 进行训练
```
