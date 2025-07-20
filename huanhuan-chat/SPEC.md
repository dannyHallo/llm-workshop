# Specs

## Environment

```shell
conda create -n huanhuan python=3.12 -y
conda activate huanhuan
pip install -r requirements.txt
```

## install pytorch

- see cuda version

```shell
nvcc --version # see cuda version
```

<https://pytorch.org/get-started/locally/>

## Use Pretrained model

<https://www.modelscope.cn/models/kmno4zx/huanhuan-chat-internlm2>

### 准备数据集

```shell
modelscope download --model kmno4zx/huanhuan-chat-internlm2
```

### 使用模型 进行对话

```shell
python pretrained_chat.py
```

## Finetuning based on pretrained model and huanhuan.json

### 训练

```shell
python train.py
```
