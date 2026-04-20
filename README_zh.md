<p align="center">
  <img src="https://img.shields.io/badge/ACL_Findings-2026-blue" alt="ACL 2026 Findings">
  <!-- <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"></a> -->
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
</p>

# ChemAmp: Amplified Chemistry Tools via Composable Agents

[English](README.md) | 中文

本工作已被 **ACL 2026 Findings** 接收。

> **TL;DR:** 通过 ChemAmp 实现工具增幅：动态协调化学工具，以 ≤10 个样本实现 SOTA 结果，同时降低 94% 的推理成本。

## 概述

大语言模型（LLMs）在化学领域展现了巨大的潜力，但由于缺乏领域专业知识以及无法有效利用专业化学工具，其性能往往受到限制。现有方法通常依赖单一工具或固定的工具流水线，无法充分利用不同化学模型之间的互补优势。

我们提出了 **ChemAmp**（Chemical Amplified Chemistry Tools），一个新颖的 Stacking 框架：

- **预热评估**：在训练子集上逐一评估每个候选化学工具，衡量其特定任务能力。
- **层次化堆叠**：将表现最优的工具组合式地构建为层次化的 Agent 结构，Agent 可以将其他 Agent 作为工具使用，实现深层工具协作。
- **多 Agent 拓扑**：支持多种多 Agent 通信模式（Chain、Star、Layered、Debate、FullConnected、Random），灵活编排 Agent 协作。

## 方法

![method](png/Intro.png)

## 项目结构

```
ChemAmp/
├── main.py                          # Stacking 框架主入口
├── Multiagent.py                    # 多Agent拓扑实验
├── ablation.py                      # 消融实验
├── requirements.txt                 # Python 依赖
├── template.env                     # API 配置模板
├── Stacking_agent/
│   ├── agent.py                     # 基于 ReAct 的 Agent 实现
│   ├── Basemodel.py                 # 聊天模型封装（Azure OpenAI、DashScope）
│   ├── Stacking.py                  # 核心 Stacking 算法
│   ├── generator.py                 # 工具组合生成器
│   ├── warmup.py                    # 预热阶段工具评估
│   ├── Tool.py                      # 工具接口
│   ├── utils.py                     # 评估指标（BLEU、ROUGE 等）
│   ├── prompt/                      # 提示模板
│   │   ├── ReAct_prompt.py
│   │   └── MultiAgent_prompt.py
│   └── tools/                       # 化学工具集
│       ├── Name2SMILES.py           # 分子名称 → SMILES（PubChem）
│       ├── SMILES2Property.py       # SMILES → 分子性质
│       ├── SMILES2Description.py    # SMILES → 文本描述
│       ├── Chemformer.py            # 反应预测模型
│       ├── ChemT5.py                # ChemT5 模型
│       ├── DFM.py                   # ChemDFM 模型
│       ├── Unimol.py                # UniMol 分子模型
│       ├── Agent_tool.py            # 嵌套 Agent 工具
│       ├── Chat_agent.py            # 对话式 Agent 工具
│       ├── FinalRefer_agent.py      # 最终参考 Agent
│       ├── llama.py                 # Llama 模型集成
│       └── deepseek.py              # DeepSeek 模型集成
├── Dataset/
│   ├── Molecule_Design/             # 分子设计（描述 → SMILES）
│   ├── Molecule_captioning/         # 分子描述生成（SMILES → 描述）
│   ├── ReactionPrediction/          # 反应预测
│   └── MolecularPropertyPrediction/ # 分子性质预测（BACE、BBBP、ClinTox、HIV、Tox21）
└── png/
    └── Intro.png
```

## 快速开始

### 安装

```bash
conda create -n ChemAmp python=3.10
conda activate ChemAmp
pip install -r requirements.txt
```

### 配置

将 `template.env` 复制为 `.env` 并填入 API 密钥：

```bash
cp template.env .env
```

| 参数 | 说明 |
|---|---|
| `AZURE_API_KEY` | Azure OpenAI API 密钥（用于 GPT-4o） |
| `AZURE_API_ENDPOINT` | Azure OpenAI 端点 URL |
| `AZURE_API_VERSION` | Azure OpenAI API 版本 |
| `ALI_API_KEY` | 阿里 DashScope API 密钥（用于 Llama3 / DeepSeek-R1） |

如使用其他 API 服务商，请相应修改 `Stacking_agent/Basemodel.py`。

### 添加自定义工具

在 `Stacking_agent/tools/` 下创建新的 `.py` 文件，并在 `Stacking_agent/generator.py` 中注册：

```python
self.tool_mapping = {
    ...
    'YourNewTool': 'YourNewTool("param")'
    ...
}
```

### 默认超参数

框架默认使用 GPT-4o (May 2024) 作为 ReAct 控制器。关键超参数如下：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `δ` | 0.05 | 新增堆叠层的性能提升阈值 |
| `top-k` | 5 | Stage 2 的分支因子（候选组合数量） |
| `train_data_number` | 10 | 每个任务的验证样本数（≤10 即可） |

### 运行 Stacking 框架

```bash
python main.py --Task "ReactionPrediction" --tools "[SMILES2Property(),Chemformer()]" --topN 5 --tool_number 2 --train_data_number 20
```

也可以直接指定 Stacking 结构（跳过训练阶段）：

```bash
python main.py --Task "Molecule_Design" --no_train --Stacking "['ChemDFM_1','Name2SMILES_0']" --topN 5 --tool_number 2 --train_data_number 10
```

### 消融实验与多 Agent 实验

```bash
# Stacking 参数消融实验
python ablation.py --Task "Task" --tools "tools" --topN topN --tool_number tool_number

# 多Agent拓扑实验（Chain / Star / Layered / Debate / FullConnected / Random）
python Multiagent.py --mode Chain --agents 0 --no_tool True
```

## 支持的任务

| 任务 | 输入 | 输出 | 评估指标 |
|---|---|---|---|
| 分子设计 | 文本描述 | SMILES | Exact Match、Validity、BLEU-2、MACCS/RDK/Morgan FTS |
| 分子描述生成 | SMILES | 文本描述 | BLEU-2/4、ROUGE-2/4/L、METEOR |
| 反应预测 | 反应方程式 | 产物 SMILES | BLEU-2 |
| 分子性质预测 | SMILES | Yes/No | AUC-ROC |

## 引用

如果您觉得本工作有帮助，请引用：

```bibtex
@misc{li2026chemampamplifiedchemistrytools,
      title={ChemAmp: Amplified Chemistry Tools via Composable Agents}, 
      author={Zhucong Li and Powei Chang and Jin Xiao and Zhijian Zhou and Qianyu He and Jiaqing Liang and Fenglei Cao and Xu Yinghui and Yuan Qi},
      year={2026},
      eprint={2505.21569},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.21569}, 
}
```


## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE)。
