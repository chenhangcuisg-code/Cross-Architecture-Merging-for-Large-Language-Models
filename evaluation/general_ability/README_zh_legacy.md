# General Ability 评估脚本使用说明

## 脚本功能

评估模型在通用能力任务上的表现，包括：
- **arc_easy**: ARC Easy 任务
- **commonsense_qa**: CommonsenseQA 任务
- **piqa**: Physical Interaction QA 任务
- **social_iqa**: Social Interaction QA 任务
- **winogrande**: Winogrande 任务

评估4个模型变体：
- **hot**: 使用HOT训练的模型
- **nohot**: 不使用HOT训练的模型
- **frozen_base**: 冻结基座的消融实验模型
- **baseline**: 基线模型

## 使用方法

### 1. 基本使用

```bash
cd /home/chenhang/optimal_trans/vis_res
bash eval_general_ability.sh
```

### 2. 自定义模型路径

如果默认模型路径不存在，可以通过环境变量覆盖：

```bash
# 设置自定义路径
export MODEL_HOT_OVERRIDE="/path/to/hot/model"
export MODEL_NOHOT_OVERRIDE="/path/to/nohot/model"
export MODEL_FROZEN_BASE_OVERRIDE="/path/to/frozen_base/model"
export MODEL_BASELINE_OVERRIDE="/path/to/baseline/model"

# 运行脚本
bash eval_general_ability.sh
```

### 3. 修改脚本中的路径

直接编辑 `eval_general_ability.sh`，修改以下变量：

```bash
# 基础路径
BASE_RUN_PATH="/your/base/path"

# 各变体的模型路径
MODEL_HOT="${BASE_RUN_PATH}/hot"
MODEL_NOHOT="${BASE_RUN_PATH}/nohot"
MODEL_FROZEN_BASE="${BASE_RUN_PATH}/hot/ablation_untrained_hot_fused"
MODEL_BASELINE="/path/to/baseline/model"
```

## 输出结果

### 1. 评估日志

结果保存在：`eval_general_ability.log`

包含每个变体的详细评估结果表格。

### 2. JSON结果文件

结果保存在：`/data/chenhang/codes/lm-evaluation-harness/output_general_ability/`

每个变体有独立的目录：
- `output_general_ability/hot/results*.json`
- `output_general_ability/nohot/results*.json`
- `output_general_ability/frozen_base/results*.json`
- `output_general_ability/baseline/results*.json`

### 3. 提取结果表格

评估完成后，可以运行以下Python脚本提取结果并生成表格：

```python
# 从JSON文件中提取结果并生成表格
import json
import glob

variants = ['hot', 'nohot', 'frozen_base', 'baseline']
tasks = ['arc_easy', 'commonsense_qa', 'piqa', 'social_iqa', 'winogrande']

results = {}
for variant in variants:
    json_file = f'/data/chenhang/codes/lm-evaluation-harness/output_general_ability/{variant}/results*.json'
    files = glob.glob(json_file)
    if files:
        with open(files[0], 'r') as f:
            data = json.load(f)
            results[variant] = {}
            for task in tasks:
                if task in data['results']:
                    # 提取acc指标
                    if 'acc' in data['results'][task]:
                        results[variant][task] = data['results'][task]['acc']
                    elif 'acc_norm' in data['results'][task]:
                        results[variant][task] = data['results'][task]['acc_norm']

# 生成表格
print("model\tarc_easy\tcommonsense_qa\tpiqa\tsocial_iqa\twinogrande\taverage")
for variant in variants:
    if variant in results:
        row = [variant]
        scores = []
        for task in tasks:
            score = results[variant].get(task, 0)
            row.append(f"{score:.4f}")
            scores.append(score)
        avg = sum(scores) / len(scores) if scores else 0
        row.append(f"{avg:.4f}")
        print("\t".join(row))
```

## 注意事项

1. **模型路径检查**: 脚本会检查模型路径是否存在，如果不存在会报错并提示设置环境变量。

2. **跳过已完成的评估**: 如果输出目录中已有 `results*.json` 文件，脚本会跳过该变体的评估。

3. **Conda环境**: 脚本会自动切换到 `lm-eval` conda环境进行评估，评估完成后切换回 `trans_opt` 环境。

4. **GPU使用**: 默认使用 `cuda:0`，如需使用其他GPU，可以修改脚本中的 `--device` 参数。

5. **批处理大小**: 默认批处理大小为8，如果GPU内存不足，可以修改 `EVAL_BATCH_SIZE` 变量。

## 故障排除

### 模型路径不存在

如果遇到模型路径不存在的错误：

1. 检查实际模型路径：
```bash
find /data/chenhang/optimal_trans_new -name "alpha0.005_lr6e-7" -type d
```

2. 使用环境变量覆盖路径（见上面的使用方法）

3. 直接修改脚本中的路径变量

### Conda环境问题

如果conda环境激活失败：

1. 检查conda是否已安装：
```bash
which conda
source ~/miniconda3/etc/profile.d/conda.sh
conda env list
```

2. 确保 `lm-eval` 环境存在：
```bash
conda activate lm-eval
```

### 评估任务失败

如果某个任务评估失败：

1. 检查任务名称是否正确
2. 检查lm-evaluation-harness是否已安装并更新
3. 查看详细错误日志：`eval_general_ability.log`
