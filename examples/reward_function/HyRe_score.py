import re
import json
import numpy as np
from typing import List, Dict, Any, Union
from examples.reward_function.local_qwen_embedding import LocalQwen3Embedding

# ================= Configuration =================
EMBEDDING_MODEL = None
SERVER_URL = "http://localhost:8003"

def get_embedding_model():
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        try:
            EMBEDDING_MODEL = LocalQwen3Embedding(server_url=SERVER_URL)
            EMBEDDING_MODEL.encode(["warmup"])
        except Exception as e:
            print(f"⚠️ Embedding server error: {e}")
    return EMBEDDING_MODEL

# ================= Analysis and Utility Functions =================
def parse_ovd_prediction(json_str: str) -> List[Dict]:
    try:
        clean = re.sub(r'```json\s*', '', json_str, flags=re.IGNORECASE)
        clean = re.sub(r'```', '', clean).strip()
        
        data = json.loads(clean)
        if not isinstance(data, list):
            return []
        valid = []
        for item in data:
            if (isinstance(item, dict) and
                isinstance(item.get('label'), str) and
                isinstance(item.get('bbox'), list) and
                len(item.get('bbox', [])) == 4 and
                all(isinstance(x, (int, float)) for x in item.get('bbox', []))):
                
                item['bbox'] = [float(x) for x in item['bbox']]
                # x1 < x2, y1 < y2
                if item['bbox'][0] > item['bbox'][2]:
                    item['bbox'][0], item['bbox'][2] = item['bbox'][2], item['bbox'][0]
                if item['bbox'][1] > item['bbox'][3]:
                    item['bbox'][1], item['bbox'][3] = item['bbox'][3], item['bbox'][1]
                valid.append(item)
        return valid
        
    except Exception as e:
        return []

def parse_res_bbox(pred_content: str) -> List[float]:
    try:
        clean = re.sub(r'```json\s*', '', pred_content, flags=re.IGNORECASE)
        clean = re.sub(r'```', '', clean).strip()
        
        data = json.loads(clean)
        
        if isinstance(data, list) and len(data) == 4 and all(isinstance(x, (int, float)) for x in data):
            return [float(x) for x in data]
             
        # Case 2: [{"bbox_2d": [x,y,x,y], ...}] or [{"bbox": ...}]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            item = data[0]
            if 'bbox_2d' in item and isinstance(item['bbox_2d'], list) and len(item['bbox_2d']) == 4:
                return [float(x) for x in item['bbox_2d']]
            elif 'bbox' in item and isinstance(item['bbox'], list) and len(item['bbox']) == 4:
                return [float(x) for x in item['bbox']]
        
        # Case 3: {"bbox_2d": [x,y,x,y]} or {"bbox": ...}
        if isinstance(data, dict):
            if 'bbox_2d' in data and isinstance(data['bbox_2d'], list) and len(data['bbox_2d']) == 4:
                return [float(x) for x in data['bbox_2d']]
            elif 'bbox' in data and isinstance(data['bbox'], list) and len(data['bbox']) == 4:
                return [float(x) for x in data['bbox']]
        
    except json.JSONDecodeError:
        pass  # Fallback to regex
    
    bbox_match = re.search(r'\[([\d\.\s,]+)\]', pred_content)
    if bbox_match:
        nums = [float(x) for x in re.split(r'[,\s]+', bbox_match.group(1)) if x.strip()]
        if len(nums) == 4:
            return [float(x) for x in nums]
    
    return []

def parse_number(text: str):
    try:
        clean_text = re.sub(r'[^\d\.\-]', '', str(text))
        return float(clean_text)
    except ValueError:
        return None

def normalize_text(text: str) -> str:
    """Standardised text"""
    text = str(text).lower().strip()
    text = text.replace(',', '')
    return re.sub(r'\s+', ' ', text)

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """calculate IoU"""
    if not box1 or not box2: return 0.0

    box1 = [float(b) for b in box1]
    box2 = [float(b) for b in box2]
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    if union <= 0: return 0.0
    return intersection / union

def _extract_think_content(response: str) -> str:
    match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""

def _extract_answer_content(response: str) -> str:

    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    
    if match:
        content = match.group(1).strip()
    else:
        content = ""

    if content:
        is_json_like = content.strip().startswith(('{', '['))
        
        if not is_json_like:
            content = re.sub(r'^The answer is\s*', '', content, flags=re.IGNORECASE).strip()
            
            content = re.sub(r'\.$', '', content).strip()
    
    return content.strip()

# ================= Scoring Logic (OVD/REC/VQA) =================
def _score_res(pred_bbox: List[float], gt_bbox: List[float]) -> float:
    iou = calculate_iou(pred_bbox, gt_bbox)
    if iou >= 0.5:
        return iou
    elif iou >= 0.3:
        return iou * 0.8
    else:
        return 0.0

def _score_ovd(pred_list: List[Dict], gt_list: List[Dict]) -> float:
    if not gt_list:
        return 1.0 if not pred_list else 0.0
    if not pred_list:
        return 0.0
    gt_pool = [item.copy() for item in gt_list]
    true_positives = 0
    partial_matches = 0
    
    for pred in pred_list:
        best_iou = 0.0
        best_gt_idx = -1
        label_match = False
        
        pred_label = normalize_text(pred.get('label', ''))
        pred_box = pred.get('bbox', [])
        for i, gt in enumerate(gt_pool):
            gt_label = normalize_text(gt.get('label', ''))
            
            if pred_label == gt_label:
                label_match = True
                iou = calculate_iou(pred_box, gt.get('bbox', []))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
        
        if best_iou >= 0.5 and best_gt_idx != -1:
            true_positives += 1
            gt_pool.pop(best_gt_idx)
        elif label_match and best_iou >= 0.3:
            partial_matches += 1
    
    tp_with_partial = true_positives + (partial_matches * 0.5)
    precision = tp_with_partial / len(pred_list) if pred_list else 0.0
    recall = tp_with_partial / len(gt_list) if gt_list else 0.0
    
    if precision + recall == 0: return 0.0
    return 2 * (precision * recall) / (precision + recall)

def _score_vqa(pred_text: str, gt_text: str) -> float:
    val_gt = parse_number(gt_text)
    if val_gt is not None:
        val_pred = parse_number(pred_text)
        if val_pred is not None:
            if abs(val_gt - val_pred) < 1e-6:
                return 1.0
            elif val_gt > 0 and abs(val_gt - val_pred) / val_gt < 0.05:
                return 0.5
        return 0.0
    
    norm_gt = normalize_text(gt_text)
    norm_pred = normalize_text(pred_text)
    
    if norm_gt == norm_pred:
        return 1.0
    
    pred_words = norm_pred.split()
    gt_words = norm_gt.split()
    if len(norm_gt.split()) == 1 and norm_gt in pred_words and len(pred_words) <= len(gt_words) * 2:
        return 1.0 if len(pred_words) == 1 else 0.8
    
    return 0.0

def determine_task_type(gt_data: Dict, problem: str = "") -> str:
    task_type = "unknown"
    
    if 'boxes' in gt_data and isinstance(gt_data['boxes'], list) and len(gt_data['boxes']) > 0 and isinstance(gt_data['boxes'][0], dict) and 'label' in gt_data['boxes'][0] and gt_data['boxes'][0]['label'] is not None:
        task_type = "ovd"
    elif 'bbox' in gt_data and isinstance(gt_data['bbox'], list) and len(gt_data['bbox']) == 4 and all(isinstance(x, (int, float)) for x in gt_data['bbox']) and 'label' in gt_data and gt_data['label'] is None:
        task_type = "res"
    elif 'text' in gt_data:
        task_type = "vqa"
    
    if task_type == "unknown" and problem:
        problem_lower = problem.lower()
        if "bounding box coordinate" in problem_lower and "region this sentence describes" in problem_lower:
            task_type = "res"
        elif "detect the following objects" in problem_lower:
            task_type = "ovd"
        else:
            task_type = "vqa"
    
    return task_type

# ================= Primary Computing Portal =================
def compute_score(reward_inputs: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    n = len(reward_inputs)
    if n == 0: return []
    
    temp_results = []
    embedding_model = get_embedding_model()
    
    for i, item in enumerate(reward_inputs):
        response = item.get("response", "")
        gt_data = item.get("ground_truth", {})
        problem = item.get("problem", "")

        think_content = _extract_think_content(response)
        pred_content = _extract_answer_content(response)

        score = 0.0
        task_type = determine_task_type(gt_data, problem)
        
        if task_type == "ovd":
            gt_list = gt_data.get('boxes', [])  # List[Dict]
            pred_list = parse_ovd_prediction(pred_content)
            score = _score_ovd(pred_list, gt_list)
        
        elif task_type == "res":
            gt_bbox = [float(x) for x in gt_data.get('bbox', []) if isinstance(gt_data.get('bbox'), list) and len(gt_data['bbox']) == 4]
            pred_bbox = parse_res_bbox(pred_content)
            if pred_bbox and gt_bbox:
                score = _score_res(pred_bbox, gt_bbox)
            else:
                score = 0.0
        
        elif task_type == "vqa":
            gt_text = gt_data.get('text', '')
            score = _score_vqa(pred_content, gt_text)
        
        # print(f"Index {i}: Task type = {task_type}, Score = {score}")  # debuggings
        
        temp_results.append({
            "index": i,
            "accuracy": score,
            "format": 1.0 if (think_content and ("<answer>" in response and "</answer>" in response)) else 0.0,  # 优化: 严格要求<answer>标签
            "think": think_content,
            "task_type": task_type,
        })
    
    novelty_scores = np.zeros(n)
    valid_indices = [
        r["index"] for r in temp_results
        if r["format"] >= 0.5 and r["accuracy"] >= 0.5 and len(r["think"]) > 10
    ]
    if len(valid_indices) > 2 and embedding_model:
        try:
            valid_thoughts = [temp_results[idx]["think"] for idx in valid_indices]
            embeddings = embedding_model.encode(valid_thoughts)
            
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            norm_emb = embeddings / norms
            sim_matrix = norm_emb @ norm_emb.T
            
            np.fill_diagonal(sim_matrix, 0)
            avg_sim = np.sum(sim_matrix, axis=1) / (len(valid_indices) - 1)
            
            # max_sim = np.max(sim_matrix, axis=1)

            alpha = 0.5
            raw_novelty = np.clip(1.0 - (alpha * avg_sim ), 0.0, 1.0)

            for k, idx in enumerate(valid_indices):
                novelty_scores[idx] = raw_novelty[k]

        except Exception as e:
            print(f"⚠️ Novelty calc error: {e}")
            for idx in valid_indices: novelty_scores[idx] = 0
    else:
        for idx in valid_indices:
            novelty_scores[idx] = 0
    
    final_results = []
    for i in range(n):
        res = temp_results[i]
        nov = novelty_scores[i]
        
        overall = (0.7 * res["accuracy"]) + (0.1 * res["format"]) + (0.2 * nov)
        
        final_results.append({
            "overall": float(overall),
            "accuracy": res["accuracy"],
            "format": res["format"],
            "novelty": float(nov)
        })
    
    return final_results