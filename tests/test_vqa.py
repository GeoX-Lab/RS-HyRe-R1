from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
import csv
import random
import string
import torch.distributed as dist
from typing import List, Union, Dict

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# ================= 分布式设置 =================
def setup_distributed():
    os.environ["NCCL_DEBUG"] = "INFO"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return local_rank, world_size, rank

local_rank, world_size, rank = setup_distributed()
device = f"cuda:{local_rank}"
if rank == 0:
    print(f"Master process {rank} initialized. World size: {world_size}")

# ================= 配置区 =================
# 添加全局变量用于记录准确率
# ACCURACY_LOG_PATH = "./checknoveltynew/accuracy_vqa.csv"
ACCURACY_LOG_PATH = "./checkmulti/accuracy_vqa.csv"
# ACCURACY_LOG_PATH = "./checkpointsnone/accuracy_vqa.csv"

# 与训练配置保持一致，防止大图崩溃
MAX_PIXELS = 4194304  # 防止 Token 爆炸
DEFAULT_COORD_MAX = 5000

STEPS = int(os.environ.get("TEST_STEP", "10"))  # 默认值10
steps_val = STEPS  # 用于输出文件名

RUN_NAME = "VQA"

main_rank = 0
if rank == main_rank:
    print("Testing step: ", STEPS)  # 更新打印信息

# 请根据实际路径修改
MODEL_ROOT = "/g0001sr/zgz/model/multi_models"  # 模型根目录
# MODEL_PATH = f"{MODEL_ROOT}/checknovelty_step_{STEPS}"  # 动态路径
# OUTPUT_DIR = "./checknoveltynew"  # 确保目录存在
MODEL_PATH = f"{MODEL_ROOT}/checkmulti_step_{STEPS}"  # 动态路径
OUTPUT_DIR = "./checkmulti"  # 确保目录存在
# MODEL_PATH = f"{MODEL_ROOT}/checkpointsnone_step_{STEPS}"  # 动态路径
# OUTPUT_DIR = "./checkpointsnone"  # 确保目录存在
OUTPUT_PATH = f"{OUTPUT_DIR}/vqa_result_{{STEPS}}.json"  # 保留{STEPS}用于后续format
# 数据集
DATA_ROOT = "/g0001sr/zgz/datasets/test/VQA"  # 假设这是你的VQA数据根目录
TEST_FILENAME = "RSVQA.jsonl"            # json文件名（不带.json）
IMAGE_ROOT_PREFIX = "/g0001sr/zgz/datasets/test/VQA/images/" # 如果json里的路径是绝对路径，这里留空；如果是相对路径，填前缀

BSZ = 256  # 根据显存调整 Batch Size

# ================= 加载模型 =================

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": local_rank},
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

def normalize_text(s: str) -> str:
    """VQA 答案标准化：转小写、去标点、去首尾空格"""
    if s is None:
        return ""
    s = str(s).lower().strip()
    translator = str.maketrans('', '', string.punctuation)
    s = s.translate(translator)
    return s.strip()

def _extract_answer_content(response: str) -> str:
    """从模型输出中提取答案"""
    # 优先提取 <answer> 标签
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 兜底：去除 <think>，清理常见前缀
    content = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    content = re.sub(r'^The answer is\s*', '', content, flags=re.IGNORECASE)
    content = re.sub(r'^Answer:\s*', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\.$', '', content)
    return content.strip()

# ================= Prompt 模板 =================

QUESTION_TEMPLATE = (
    "{Question}\n\n"
    "Reasoning:\n"
    "Analyze the image content carefully to answer the question. Enclose your thinking process within <think> and </think> tags.\n\n"
    "Final Answer:\n"
    "Provide a short and concise answer (e.g., 'yes', 'no', 'urban', '42') enclosed within <answer> and </answer> tags.\n"
    "Example: <answer>urban</answer>"
)

# ================= 主逻辑 =================

if rank == 0:
    print(f"Processing {TEST_FILENAME}...")

ds_path = os.path.join(DATA_ROOT, TEST_FILENAME)

# 读取 .jsonl 数据
data = []
if os.path.exists(ds_path):
    with open(ds_path, "r", encoding="utf-8") as f:
        # 逐行读取 JSONL
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    if rank == 0:
                        print(f"Skipping invalid line: {e}")
else:
    if rank == 0:
        print(f"Error: Dataset not found at {ds_path}")
    exit(1)

if rank == 0:
    print(f"Loaded {len(data)} samples from {TEST_FILENAME}")

# 分布式切分
per_rank_data = len(data) // world_size
start_idx = rank * per_rank_data
end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
rank_data = data[start_idx:end_idx]

messages = []

for x in rank_data:
    # 样本格式适配: {"problem": "...", "answer": {"text": "..."}, "images": ["/abs/path/..."]}
    
    # 获取图片路径
    if isinstance(x.get('images'), list):
        img_rel_path = x['images'][0]
    else:
        img_rel_path = x.get('images', "")

    # 处理绝对路径 vs 相对路径
    # 如果 img_rel_path 以前斜杠开头，os.path.join 会忽略前面的 IMAGE_ROOT_PREFIX
    image_path = os.path.join(IMAGE_ROOT_PREFIX, img_rel_path)
    
    # 构造 Chat 消息
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": f"file://{image_path}"
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=x['problem'])
                }
            ]
        }
    ]
    messages.append(message)

rank_outputs = []

# 批量推理
for i in tqdm(range(0, len(messages), BSZ), disable=rank != 0):
    batch_messages = messages[i:i + BSZ]
    
    # 应用 Chat Template
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    
    image_inputs, video_inputs = process_vision_info(batch_messages)
    
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    )
    
    inputs = inputs.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            use_cache=True, 
            max_new_tokens=768, 
            do_sample=False 
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    rank_outputs.extend(batch_output_text)

# 收集结果
rank_results = []
for idx, output in enumerate(rank_outputs):
    global_idx = start_idx + idx
    rank_results.append((global_idx, output))

gathered_results = [None] * world_size
dist.all_gather_object(gathered_results, rank_results)

# 主进程统计准确率
if rank == 0:
    all_outputs_map = {}
    for res_list in gathered_results:
        for idx, out in res_list:
            all_outputs_map[idx] = out
    
    all_outputs = [all_outputs_map[i] for i in range(len(data))]
    
    final_output = []
    correct_number = 0

    for i, (input_example, model_output) in enumerate(zip(data, all_outputs)):
        # 提取 Ground Truth (根据你的样本结构: answer -> text)
        ground_truth_raw = input_example.get('answer', {}).get('text', "")
        
        # 提取模型预测
        extracted_content = _extract_answer_content(model_output)
        
        # 标准化对比 (VQA Accuracy)
        pred_norm = normalize_text(extracted_content)
        gt_norm = normalize_text(ground_truth_raw)
        
        is_correct = 1 if pred_norm == gt_norm else 0
        correct_number += is_correct
        
        result = {
            'image': input_example.get('images', [""])[0],
            'question': input_example.get('problem', ""),
            'ground_truth': ground_truth_raw,
            'model_output_raw': model_output,
            'extracted_answer': extracted_content,
            'pred_norm': pred_norm,
            'gt_norm': gt_norm,
            'correct': is_correct
        }
        final_output.append(result)

    accuracy = correct_number / len(data) * 100
    print(f"\nAccuracy of {TEST_FILENAME}: {accuracy:.2f}%")

    log_exists = os.path.exists(ACCURACY_LOG_PATH)
    with open(ACCURACY_LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(['step', 'accuracy', 'dataset'])
        writer.writerow([STEPS, f"{accuracy:.2f}", TEST_FILENAME])

    output_path = OUTPUT_PATH.format(STEPS=steps_val)
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            'config': {'max_pixels': MAX_PIXELS, 'dataset': TEST_FILENAME},
            'accuracy': accuracy,
            'total_samples': len(data),
            'results': final_output
        }, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_path}")
    print("-" * 100)

dist.barrier()