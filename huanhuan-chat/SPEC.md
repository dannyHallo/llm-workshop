# 项目环境准备指南

## 1. 环境概览
- 远程 GPU 服务：[@AI Galaxy 控制台](https://gpu.ai-galaxy.cn/console/dashboard)  
- 本地编辑器：VS Code（Remote-SSH 插件）  
- 主要依赖：Conda / Node.js / PyTorch / Lazygit  
- 预训练模型：[@kmno4zx/huanhuan-chat-internlm2](https://www.modelscope.cn/models/kmno4zx/huanhuan-chat-internlm2)

---

## 2. VS Code Remote-SSH 连接
1. 安装 VS Code 扩展 **Remote-SSH**  
2. 打开命令面板 `Ctrl+Shift+P` → `Remote-SSH: Connect to Host`  
3. 在 AI Galaxy **控制台 → 连接方式 SSH** 中复制 SSH 登录命令并粘贴执行

---

## 3. 远程服务器初始化
```shell
# 更新系统
sudo apt update
sudo apt upgrade -y
```

### 3.1 Git & Lazygit
```shell
# Git 全局信息
git config --global user.name "your_name"
git config --global user.email "your_email"

# 安装 Lazygit
curl -Lo lazygit.tar.gz "https://github.com/jesseduffield/lazygit/releases/download/v0.53.0/lazygit_0.53.0_Linux_x86_64.tar.gz"
sudo tar -xf lazygit.tar.gz -C /usr/local/bin/
lazygit --version 
```

### 3.2 Conda 与依赖
```shell
# 创建并激活虚拟环境
conda create -n huanhuan python=3.12 -y
conda activate huanhuan

# 安装 Python 依赖
pip install -r requirements.txt
```

---

## 4. Node.js & Claude Code
> Node.js 官方下载页：<https://nodejs.org/en/download>

```shell
# 安装 nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
. "$HOME/.nvm/nvm.sh"   # 立即加载 nvm，无需重启

# 安装 Node.js LTS 22.17.1
nvm install 22
node -v    # => v22.17.1
npm -v     # => 10.9.2

# 安装 Claude Code CLI
npm install -g @anthropic-ai/claude-code
```

---

## 5. 安装 PyTorch

- 注意：租用云主机时，默认是安装了 pytorch 的，可以直接跳过此步骤。

1. 先确认 **CUDA** 版本：
   ```shell
   nvcc --version
   ```
2. 根据 CUDA 版本与操作系统，在官方 [PyTorch Get Started](https://pytorch.org/get-started/locally/) 页面生成安装命令  
3. 执行生成的 `pip install ...` 命令即可

---

## 6. 预训练模型下载与使用
### 6.1 下载模型
```shell
# Meta-Llama-3 8B Instruct
modelscope download --model LLM-Research/Meta-Llama-3-8B-Instruct

# 嬛嬛 Chat InternLM2
modelscope download --model kmno4zx/huanhuan-chat-internlm2
```

下载的模型会保存在 `~/.cache/modelscope/hub/` 目录下。

### 6.2 推理测试
```shell
python pretrained_chat.py
```

---

## 7. 微调（Finetuning）
```shell
python train.py  # 基于 pretrained model 与 huanhuan.json 进行训练
```

---

> 完成以上步骤后，即可在 VS Code 远程环境中愉快地进行模型推理与微调实验 🚀
