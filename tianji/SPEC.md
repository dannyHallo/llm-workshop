# 项目环境准备指南

---

## VS Code Remote-SSH 连接

1. 安装 VS Code 扩展 **Remote-SSH**  
2. 打开命令面板 `Ctrl+Shift+P` → `Remote-SSH: Connect to Host`  
3. 在 AI Galaxy **控制台 → 连接方式 SSH** 中复制 SSH 登录命令并粘贴执行

---

### Conda 与依赖

```shell
# 创建并激活虚拟环境
conda create -n tianji python=3.12 -y
conda activate tianji

# 安装 Python 依赖
pip install -e .
```
