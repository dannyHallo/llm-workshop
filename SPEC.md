# 远程服务器初始化

## 根 repo

<https://github.com/datawhalechina/self-llm/>

## 环境概览

- 远程 GPU 服务：[@AI Galaxy 控制台](https://gpu.ai-galaxy.cn/console/dashboard)  
- 本地编辑器：VS Code（Remote-SSH 插件）  

```shell
# 更新系统
sudo apt update
sudo apt upgrade -y
```

### Git & Lazygit

```shell
# Git 全局信息
git config --global user.name "your_name"
git config --global user.email "your_email"

# 安装 Lazygit
curl -Lo lazygit.tar.gz "https://github.com/jesseduffield/lazygit/releases/download/v0.53.0/lazygit_0.53.0_Linux_x86_64.tar.gz"
sudo tar -xf lazygit.tar.gz -C /usr/local/bin/
lazygit --version 
```

---

## Node.js & Claude Code
>
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

## 安装 PyTorch

- 注意：租用云主机时，默认是安装了 pytorch 的，可以直接跳过此步骤。

1. 先确认 **CUDA** 版本：

   ```shell
   nvcc --version
   ```

2. 根据 CUDA 版本与操作系统，在官方 [PyTorch Get Started](https://pytorch.org/get-started/locally/) 页面生成安装命令  
3. 执行生成的 `pip install ...` 命令即可
