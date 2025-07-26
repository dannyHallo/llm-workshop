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
pip install -r requirements.txt
```

---

### 开始训练

- 使用 OpenAI API 制造数据

```shell
# 制造数据
python tools/finetune/data_maker/get_wish_datav1.py

# 合并数据
python tools/finetune/data_maker/merge_data_json.py -f ./our_dataset -o ./our_dataset/tianji-wishes-test.json

# 清理数据
## 清理小于10个字符的数据
python tools/finetune/datajson_refiner/remove_shot_len.py
## input 后处理
python tools/finetune/datajson_refiner/rebuild_input.py
## output 后处理
python tools/finetune/datajson_refiner/rebuild_output.py
```

- 最后的输出路径 `./our_dataset/tianji-wishes-test.json`
