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
from typing import List, Dict, Any, Tuple
import numpy as np

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
MAX_PIXELS = 4194304  # 防止 Token 爆炸
DEFAULT_COORD_MAX = 5000

# 输出日志路径
ACCURACY_LOG_PATH = "./checkmulti/accuracy_ovd.csv"
# ACCURACY_LOG_PATH = "./checknoveltyovd/accuracy_ovd.csv"

STEPS = int(os.environ.get("TEST_STEP", "10"))
steps_val = STEPS 

RUN_NAME = "OVD" # 修改任务名称

if rank == 0:
    print("Testing step: ", STEPS)

# 请根据实际路径修改
MODEL_ROOT = "/g0001sr/zgz/model/multi_models"
# MODEL_PATH = f"{MODEL_ROOT}/checknovelty_step_{STEPS}"  # 动态路径
# OUTPUT_DIR = "./checknoveltyovd"  # 确保目录存在
MODEL_PATH = f"{MODEL_ROOT}/checkmulti_step_{STEPS}"
OUTPUT_DIR = "./checkmulti" # 修改输出目录
OUTPUT_PATH = f"{OUTPUT_DIR}/ovd_result_{{STEPS}}.json"

# 数据集配置
DATA_ROOT = "/g0001sr/zgz/datasets/test/RSOD512" 
# 假设您的OVD数据集名字，请修改
TEST_DATASETS = ['RSOD_val'] 
IMAGE_ROOT = "/g0001sr/zgz/datasets/test/RSOD512/images" # 请修改为实际图片路径

BSZ = 128

# 加载模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": local_rank}, 
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, fix_mistral_regex=True)

# ================= 核心工具函数 =================

def normalize_text(text: str) -> str:
    """标准化标签文本：小写、去空、去标点"""
    if not isinstance(text, str): return str(text)
    text = text.lower().strip()
    text = text.replace('.', '').replace('_', ' ').replace('-', ' ')
    return text

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """计算 IoU"""
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

# ================= 标准 mAP 计算实现 =================
def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """计算 PR 曲线下面积（11点插值法）"""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0
    return ap

def compute_class_ap(gt_boxes: List[List[float]], 
                     pred_boxes: List[List[float]], 
                     pred_scores: List[float],
                     iou_threshold: float) -> float:
    """
    计算单类别、单IoU阈值下的AP
    """
    # 按置信度降序排序
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = [pred_boxes[i] for i in sorted_indices]
    pred_scores = [pred_scores[i] for i in sorted_indices]
    
    num_gts = len(gt_boxes)
    detected = [False] * num_gts
    tp = []
    fp = []
    
    # 遍历每个预测
    for pred_box, score in zip(pred_boxes, pred_scores):
        best_iou = 0.0
        best_gt_idx = -1
        
        # 找最佳匹配GT
        for i, gt_box in enumerate(gt_boxes):
            if detected[i]: continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        # 判定TP/FP
        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            detected[best_gt_idx] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    
    # 累计TP/FP
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    
    # 计算Precision/Recall
    recalls = tp / num_gts if num_gts > 0 else np.zeros(len(tp))
    precisions = tp / (tp + fp + 1e-8)
    
    # 11点插值计算AP
    ap = compute_ap(recalls, precisions)
    return ap

def _score_ovd_standard(pred_list: List[Dict], gt_list: List[Dict]) -> Dict[str, float]:
    """
    标准 mAP@[.5:.95] 计算 (COCO 风格)
    返回: {
        'mAP': float,            # 所有类别平均
        'mAP@.5': float,         # IoU=0.5 时的 mAP
        'mAP@.75': float,        # IoU=0.75 时的 mAP
        'category_AP': Dict[str, float]  # 每个类别的 AP@[.5:.95]
    }
    """
    # 1. 收集类别
    gt_classes = set(normalize_text(x['label']) for x in gt_list)
    pred_classes = set(normalize_text(x['label']) for x in pred_list)
    all_classes = gt_classes.union(pred_classes)
    
    if not all_classes:
        return {'mAP': 0.0, 'mAP@.5': 0.0, 'mAP@.75': 0.0, 'category_AP': {}}
    
    # 2. 按类别分组
    class_gts = {cls: [] for cls in all_classes}
    class_preds = {cls: {'boxes': [], 'scores': []} for cls in all_classes}
    
    for gt in gt_list:
        cls = normalize_text(gt['label'])
        if cls in class_gts:
            class_gts[cls].append(gt['bbox'])
    
    for pred in pred_list:
        cls = normalize_text(pred['label'])
        # 为预测添加默认置信度（按原逻辑，我们假设所有预测有效）
        score = 1.0  # 注意：真实场景应从模型获取置信度
        if cls in class_preds:
            class_preds[cls]['boxes'].append(pred['bbox'])
            class_preds[cls]['scores'].append(score)
    
    # 3. 计算每个类别的 AP@[.5:.95]
    category_ap = {}
    iou_thresholds = np.linspace(0.5, 0.95, 10)  # 0.5, 0.55, ..., 0.95
    
    for cls in all_classes:
        ap_sum = 0.0
        ap_at_50 = 0.0
        ap_at_75 = 0.0
        
        for i, iou_thresh in enumerate(iou_thresholds):
            ap = compute_class_ap(
                class_gts[cls],
                class_preds[cls]['boxes'],
                class_preds[cls]['scores'],
                iou_thresh
            )
            ap_sum += ap
            
            if iou_thresh == 0.5:
                ap_at_50 = ap
            if iou_thresh == 0.75:
                ap_at_75 = ap
        
        # 平均10个IoU阈值得到该类别的 AP@[.5:.95]
        class_ap = ap_sum / len(iou_thresholds)
        category_ap[cls] = class_ap
    
    # 4. 全局平均
    mAP = sum(category_ap.values()) / len(category_ap) if category_ap else 0.0
    
    # 5. 提取特定阈值的 mAP
    # (简化版：重用计算结果)
    mAP_50 = ap_at_50
    mAP_75 = ap_at_75
    
    return {
        'mAP': mAP,
        'mAP@.5': mAP_50,
        'mAP@.75': mAP_75,
        'category_AP': category_ap
    }

def parse_ovd_pred(pred_content: str) -> List[Dict]:
    """
    解析模型输出的 JSON 列表
    """
    clean_content = re.sub(r'```json\s*', '', pred_content, flags=re.IGNORECASE)
    clean_content = re.sub(r'```', '', clean_content)
    
    match = re.search(r'(\[.*\])', clean_content, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = clean_content

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        try:
            json_str_fix = json_str.replace("'", '"')
            data = json.loads(json_str_fix)
        except:
            return []

    valid_preds = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                bbox = item.get('bbox', item.get('bbox_2d', []))
                label = item.get('label', item.get('category', ''))
                
                if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
                    bbox = [max(0.0, min(512.0, float(x))) for x in bbox]
                    valid_preds.append({
                        "bbox": bbox,
                        "label": str(label)
                    })
    
    return valid_preds

def _extract_answer_content(response: str) -> str:
    """提取 <answer> 标签内的内容"""
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    content = re.sub(r'\<think\>.*?\<\/think\>', '', response, flags=re.DOTALL).strip()
    return content

QUESTION_TEMPLATE = (
    "{Question}\n\n"
    "Reasoning:\n"
    "Analyze the image to detect objects. Enclose your step-by-step thinking process within <think> and </think> tags.\n\n"
    "Final Answer:\n"
    "Provide the detected objects as a JSON list, where each object has a 'bbox' [x1, y1, x2, y2] and a 'label'. Enclose the JSON within <answer> and </answer> tags.\n"
    "Example: <answer>[{{\"bbox\": [10, 10, 50, 50], \"label\": \"aircraft\"}}, {{\"bbox\": [60, 60, 100, 100], \"label\": \"truck\"}}]</answer>"
)

# ----------------- 主循环 -----------------

for ds in TEST_DATASETS:
    if rank == 0:
        print(f"Processing dataset: {ds}")
    ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
    
    if not os.path.exists(ds_path):
        if rank == 0: print(f"Dataset {ds_path} not found, skipping.")
        continue

    with open(ds_path, "r") as f:
        data = json.load(f)

    # [新增] 打印数据集实际大小，确保您加载了所有数据
    if rank == 0:
        print(f"Total samples loaded from {ds_path}: {len(data)}")
        
    random.seed(42)
    random.shuffle(data)
    
    # [修改] 彻底移除了切片代码，确保测试全量数据
    # data = data[:num_samples]  <-- 已删除

    # 分布式切分
    # 使用余数切分法防止尾部数据丢失，并确保更均匀
    # (原代码的切片法在 len(data) < world_size 时会有问题，这里保持原逻辑但请注意)
    per_rank_data = len(data) // world_size
    start_idx = rank * per_rank_data
    # 如果是最后一张卡，取剩下所有数据
    end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
    rank_data = data[start_idx:end_idx]
    
    if rank == 0:
        print(f"Rank 0 processing {len(rank_data)} samples (Batch Size: {BSZ})")

    messages = []
    
    for x in rank_data:
        image_path = os.path.join(IMAGE_ROOT, x['image'])
        
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

    # 推理循环
    # [注意] disable=rank!=0 只会让主进程显示进度条
    # 进度条的总长度 = len(messages) / BSZ。如果 BSZ 很大，进度条会很短。
    for i in tqdm(range(0, len(messages), BSZ), disable=rank != 0, desc=f"Inferencing {ds}"):
        batch_messages = messages[i:i + BSZ]
        
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
                max_new_tokens=2048, 
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

    if rank == 0:
        all_outputs_map = {}
        for res_list in gathered_results:
            for idx, out in res_list:
                all_outputs_map[idx] = out
        
        # 重组完整列表
        # 注意：如果分布式切分有余数丢失，这里可能会报错，所以加个 .get
        all_outputs = [all_outputs_map.get(i, "") for i in range(len(data))]
        
        final_output = []
        total_mAP = 0.0
        total_mAP_50 = 0.0
        total_mAP_75 = 0.0
        valid_count = 0

        print("Evaluating with standard COCO-style mAP...")
        for i, (input_example, model_output) in enumerate(zip(data, all_outputs)):
            if not model_output: continue # 跳过分布式可能丢失的极少数样本
            
            extracted_content = _extract_answer_content(model_output)
            pred_list = parse_ovd_pred(extracted_content)
            
            gt_data = input_example['solution']
            if isinstance(gt_data, str):
                try:
                    gt_data = json.loads(gt_data)
                except:
                    gt_data = {}
            
            gt_list = []
            if isinstance(gt_data, dict):
                gt_list = gt_data.get('boxes', [])
            elif isinstance(gt_data, list):
                gt_list = gt_data
            
            # 使用标准 mAP 评估
            metrics = _score_ovd_standard(pred_list, gt_list)
            mAP_score = metrics['mAP']
            
            total_mAP += mAP_score
            total_mAP_50 += metrics['mAP@.5']
            total_mAP_75 += metrics['mAP@.75']
            valid_count += 1
            
            result = {
                'image': input_example['image'],
                'question': input_example['problem'],
                'ground_truth': gt_list,
                'model_output': model_output,
                'extracted_content': extracted_content,
                'parsed_predictions': pred_list,
                'mAP': mAP_score,
                'mAP@.5': metrics['mAP@.5'],
                'mAP@.75': metrics['mAP@.75'],
                'category_AP': metrics['category_AP']
            }
            final_output.append(result)

        avg_mAP = total_mAP / valid_count if valid_count > 0 else 0
        avg_mAP_50 = total_mAP_50 / valid_count if valid_count > 0 else 0
        avg_mAP_75 = total_mAP_75 / valid_count if valid_count > 0 else 0
        print(f"\nStandard COCO mAP@[.5:.95] of {ds}: {avg_mAP:.4f} (Computed on {valid_count} samples)")
        print(f"mAP@.5: {avg_mAP_50:.4f}, mAP@.75: {avg_mAP_75:.4f}")

        # 保存日志
        log_exists = os.path.exists(ACCURACY_LOG_PATH)
        output_dir = os.path.dirname(ACCURACY_LOG_PATH)
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        with open(ACCURACY_LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            if not log_exists:
                writer.writerow(['step', 'mAP@[.5:.95]', 'mAP@.5', 'mAP@.75', 'dataset'])
            writer.writerow([STEPS, f"{avg_mAP:.4f}", f"{avg_mAP_50:.4f}", f"{avg_mAP_75:.4f}", ds])

        # 保存详细 JSON
        output_path = OUTPUT_PATH.format(DATASET=ds, RUN_NAME=RUN_NAME, STEPS=steps_val)
        output_dir_json = os.path.dirname(output_path)
        if not os.path.exists(output_dir_json):
            os.makedirs(output_dir_json)
            
        with open(output_path, "w") as f:
            json.dump({
                'config': {'max_pixels': MAX_PIXELS},
                'average_mAP': avg_mAP,
                'average_mAP@.5': avg_mAP_50,
                'average_mAP@.75': avg_mAP_75,
                'total_samples': valid_count,
                'results': final_output
            }, f, indent=2)

        print(f"Results saved to {output_path}")
        print("-" * 100)

    dist.barrier()