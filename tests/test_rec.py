from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
import csv
import random
import torch.distributed as dist
from typing import List, Union, Dict

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# ================= 分布式设置 =================

def setup_distributed():
    # 抑制 socket 警告 (Optional, 视环境而定)
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
# 与训练配置保持一致，防止大图崩溃
MAX_PIXELS = 4194304  # 防止 Token 爆炸
DEFAULT_COORD_MAX = 5000

# 添加全局变量用于记录准确率
# ACCURACY_LOG_PATH = "./checkfianl/accuracy_res.csv"
ACCURACY_LOG_PATH = "./checkmulti/accuracy_res.csv"

STEPS = int(os.environ.get("TEST_STEP", "10"))  # 默认值10
steps_val = STEPS  # 用于输出文件名

RUN_NAME = "REC"

main_rank = 0
if rank == main_rank:
    print("Testing step: ", STEPS)  # 更新打印信息

# 请根据实际路径修改
MODEL_ROOT = "/g0001sr/zgz/model/multi_models"  # 模型根目录
MODEL_PATH = f"{MODEL_ROOT}/checkmulti_step_{STEPS}"  # 动态路径
OUTPUT_DIR = "./checkmulti"  # 确保目录存在
# MODEL_PATH = f"{MODEL_ROOT}/checkpointsnone_step_{STEPS}"  # 动态路径
# OUTPUT_DIR = "./checkpointsnone"  # 确保目录存在
OUTPUT_PATH = f"{OUTPUT_DIR}/res_result_{{STEPS}}.json"  # 保留{STEPS}用于后续format
# 数据集
DATA_ROOT = "/g0001sr/zgz/datasets/VRSBench"
TEST_DATASETS = ['VRSBench_EVAL_referring_detailed_with_unique']
IMAGE_ROOT = "/g0001sr/zgz/datasets/VRSBench/Images/val"


BSZ = 256  # 根据显存调整

# 加载模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": local_rank}, 
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# ================= 解析工具函数 (源自 optimized_reward.py) =================
def parse_res_bbox(pred_content: str) -> List[float]:
    try:
        # 去掉 markdown
        clean = re.sub(r'```json\s*', '', pred_content, flags=re.IGNORECASE)
        clean = re.sub(r'```', '', clean).strip()
        
        # 尝试JSON解析
        data = json.loads(clean)
        
        # Case 1: 直接 [x,y,x,y]
        if isinstance(data, list) and len(data) == 4 and all(isinstance(x, (int, float)) for x in data):
            return [max(0.0, min(512.0, float(x))) for x in data]
        
        # Case 2: [{"bbox_2d": [x,y,x,y], ...}] or [{"bbox": ...}]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            item = data[0]
            if 'bbox_2d' in item and isinstance(item['bbox_2d'], list) and len(item['bbox_2d']) == 4:
                return [max(0.0, min(512.0, float(x))) for x in item['bbox_2d']]
            elif 'bbox' in item and isinstance(item['bbox'], list) and len(item['bbox']) == 4:
                return [max(0.0, min(512.0, float(x))) for x in item['bbox']]
        
        # Case 3: {"bbox_2d": [x,y,x,y]} or {"bbox": ...}
        if isinstance(data, dict):
            if 'bbox_2d' in data and isinstance(data['bbox_2d'], list) and len(data['bbox_2d']) == 4:
                return [max(0.0, min(512.0, float(x))) for x in data['bbox_2d']]
            elif 'bbox' in data and isinstance(data['bbox'], list) and len(data['bbox']) == 4:
                return [max(0.0, min(512.0, float(x))) for x in data['bbox']]
        
    except json.JSONDecodeError:
        pass  # Fallback to regex
    
    # Regex fallback: 找第一个 [x,y,x,y]
    bbox_match = re.search(r'\[([\d\.\s,]+)\]', pred_content)
    if bbox_match:
        nums = [float(x) for x in re.split(r'[,\s]+', bbox_match.group(1)) if x.strip()]
        if len(nums) == 4:
            return [max(0.0, min(512.0, x)) for x in nums]
    
    return []  # 失败返回空

def _extract_answer_content(response: str) -> str:
    """鲁棒的答案提取，优先提取 <answer> 标签"""
    # 1. Priority: <answer>...</answer>
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 2. Fallback: Answer: ...
    match = re.search(r'Answer:\s*(.*)', response, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
    else:
        # 3. Final Fallback: Remove <think> and take remainder
        content = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    
    # 清理逻辑：如果看起来像 JSON，保留原样；否则清理前缀
    is_json_like = content.strip().startswith(('{', '[', '```'))
    
    if not is_json_like:
        lines = content.split('\n')
        if lines:
            content = lines[0].strip()
        content = re.sub(r'^The answer is\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'\.$', '', content)
        
    return content.strip()

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """标准的 IoU 计算"""
    if not box1 or not box2: return 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

# Prompt 模板：必须与训练时的 Format Prompt 对齐
# ⚠️ 关键修复：JSON 示例中的花括号必须使用 {{ }} 进行转义，否则 .format() 会报错
QUESTION_TEMPLATE = (
    "{Question}\n\n"
    "Reasoning:\n"
    "Analyze the visual cues and spatial localization. Enclose your step-by-step thinking process within <think> and </think> tags.\n\n"
    "Final Answer:\n"
    "Provide the final bounding box in JSON format enclosed within <answer> and </answer> tags.\n"
    "Example: <answer>{{\"bbox\": [x1, y1, x2, y2]}}</answer>"
)

num_samples = 20000

for ds in TEST_DATASETS:
    if rank == 0:
        print(f"Processing {ds}...")
    ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
    
    with open(ds_path, "r") as f:
        data = json.load(f)
        
    # 为了复现一致性，固定种子
    random.seed(42)
    random.shuffle(data)
    data = data[:num_samples]

    if rank == 0:
        print(f"Loaded {len(data)} samples from {TEST_DATASETS}")

    # 分布式数据切分
    per_rank_data = len(data) // world_size
    start_idx = rank * per_rank_data
    end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
    rank_data = data[start_idx:end_idx]

    messages = []
    # 预存原始尺寸，用于后续坐标解析
    original_sizes = [] 

    for x in rank_data:
        image_path = os.path.join(IMAGE_ROOT, x['image'])
        
        # 尝试获取图片尺寸信息，如果数据集里没有，后续 clip 会使用默认上限
        w = x.get('width', DEFAULT_COORD_MAX)
        h = x.get('height', DEFAULT_COORD_MAX)
        original_sizes.append([w, h])

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
        
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        
        # ⚠️ 关键修正：传入 max_pixels 以匹配训练设置，防止大图报错
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
                max_new_tokens=768,  # 增加 token 数以容纳 CoT
                do_sample=False      # 验证时通常使用贪婪搜索
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
    # rank_outputs 对应的是 rank_data 的顺序
    for idx, output in enumerate(rank_outputs):
        global_idx = start_idx + idx
        rank_results.append((global_idx, output))

    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, rank_results)

    # 主进程汇总与计算指标
    if rank == 0:
        all_outputs_map = {}
        for res_list in gathered_results:
            for idx, out in res_list:
                all_outputs_map[idx] = out
        
        # 还原顺序
        all_outputs = [all_outputs_map[i] for i in range(len(data))]
        
        final_output = []
        correct_number_05 = 0
        correct_number_07 = 0

        for i, (input_example, model_output) in enumerate(zip(data, all_outputs)):
            ground_truth = input_example['solution']
            
            # 解析预测内容
            extracted_content = _extract_answer_content(model_output)
            
            # 获取图片尺寸用于 Clip (假设全集数据和原始数据顺序一致)
            # 如果 input_example 中没有宽高，尝试使用默认值
            img_w = input_example.get('width', DEFAULT_COORD_MAX)
            img_h = input_example.get('height', DEFAULT_COORD_MAX)
            img_size = [img_w, img_h]
            
            # 解析 BBox
            model_bbox = parse_res_bbox(extracted_content)
            
            # 处理 Ground Truth (兼容 list 或 string 格式)
            if isinstance(ground_truth, str):
                try:
                    gt_bbox = json.loads(ground_truth)
                except:
                    gt_bbox = []
            else:
                gt_bbox = ground_truth
            
            # 确保 GT 格式正确
            if isinstance(gt_bbox, dict):
                # 如果 GT 是 {"bbox": ...} 格式
                gt_bbox = gt_bbox.get('bbox', [])

            # 计算 IoU
            score = calculate_iou(model_bbox, gt_bbox)
            is_correct_05 = 1 if score > 0.5 else 0
            is_correct_07 = 1 if score > 0.7 else 0  # 新增
            correct_number_05 += is_correct_05
            correct_number_07 += is_correct_07

            result = {
                'image': input_example['image'],
                'question': input_example['problem'],
                'ground_truth': gt_bbox,
                'model_output': model_output,
                'extracted_content': extracted_content,
                'parsed_bbox': model_bbox,
                'iou': score,
                'correct05': is_correct_05,
                'correct07': is_correct_07
            }
            final_output.append(result)

        accuracy_05 = correct_number_05 / len(data) * 100  # 原accuracy
        accuracy_07 = correct_number_07 / len(data) * 100  # 新增
        print(f"\nACC@0.5 of {ds}: {accuracy_05:.2f}%")
        print(f"\nACC@0.7 of {ds}: {accuracy_07:.2f}%")

        log_exists = os.path.exists(ACCURACY_LOG_PATH)
        with open(ACCURACY_LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            if not log_exists:
                writer.writerow(['step', 'ACC@0.5', 'ACC@0.7', 'dataset'])
            writer.writerow([STEPS, f"{accuracy_05:.2f}", f"{accuracy_07:.2f}", ds])

        output_path = OUTPUT_PATH.format(DATASET=ds, RUN_NAME=RUN_NAME, STEPS=steps_val)
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_path, "w") as f:
            json.dump({
                'config': {'max_pixels': MAX_PIXELS},
                'ACC@0.5': accuracy_05,
                'ACC@0.7': accuracy_07,
                'total_samples': len(data),
                'results': final_output
            }, f, indent=2)

        print(f"Results saved to {output_path}")
        print("-" * 100)

    dist.barrier()