"""
小老曾 LoRA 微调脚本
基座: Qwen2.5-1.5B-Instruct
数据: 4688 QA 对 (ShareGPT -> MLX chat format)
框架: mlx-lm (Apple M3 Metal 加速)
"""
import json
import time
from pathlib import Path
from mlx_lm import lora, load
from mlx_lm.tuner import TrainingArgs

BASE = Path('/Users/rickicui/WorkBuddy/产品老曾')
MODEL_PATH = BASE / 'models/qwen2.5-1.5b-instruct-mlx'
TRAIN_DATA = BASE / 'rag_xiaozeng/training_data/train_split.jsonl'
VAL_DATA = BASE / 'rag_xiaozeng/training_data/val_split.jsonl'
ADAPTER_PATH = BASE / 'models/xiaozeng-lora-adapter'

# 加载数据集
def load_dataset(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

print('=' * 50)
print('  小老曾 LoRA 微调')
print('=' * 50)
print(f'  Model: {MODEL_PATH}')
print(f'  Train: {TRAIN_DATA}')
print(f'  Val: {VAL_DATA}')
print()

train_dataset = load_dataset(TRAIN_DATA)
val_dataset = load_dataset(VAL_DATA)
print(f'  Train samples: {len(train_dataset)}')
print(f'  Val samples: {len(val_dataset)}')

# 计算 step 数：3 epoch
steps_per_epoch = len(train_dataset) // 4
total_iters = steps_per_epoch * 3
print(f'  Steps per epoch: {steps_per_epoch}')
print(f'  Total iters (3 epochs): {total_iters}')
print()

# 训练参数
args = TrainingArgs(
    batch_size=4,
    iters=total_iters,
    val_batches=25,
    steps_per_report=20,
    steps_per_eval=200,
    steps_per_save=500,
    max_seq_length=2048,
    adapter_file=str(ADAPTER_PATH / 'adapters.safetensors'),
)

print('Starting training...')
print(f'  Adapter output: {ADAPTER_PATH}')
print(f'  Report every {args.steps_per_report} steps')
print()

start = time.time()

lora.train(
    model=str(MODEL_PATH),
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    args=args,
)

elapsed = time.time() - start
print(f'\nTraining complete! Elapsed: {elapsed/60:.1f} min')
print(f'Adapter saved to: {ADAPTER_PATH}')
