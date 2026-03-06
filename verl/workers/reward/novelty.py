import re
import torch
import numpy as np
from typing import List, Tuple, Any
from collections import defaultdict
from transformers import PreTrainedTokenizer
from ...protocol import DataProto
from ...utils.reward_score import compute_universal_reward
from .local_qwen_embedding import LocalQwen3Embedding

class UniversalRewardManager:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        use_local_embedding: bool = True,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.use_local = use_local_embedding
        assert use_local_embedding, "Only local embedding supported"
        
        # Qwen3嵌入服务 (固定地址)
        server_url = "http://172.30.217.209:8001"
        self.embedding_model = LocalQwen3Embedding(server_url=server_url)
        self.local_embedding_fn = self.embedding_model.encode

    def _decode_response(self, data_item) -> str:
        """解码单个响应"""
        prompt_len = data_item.batch["prompts"].shape[-1]
        attn_mask = data_item.batch["attention_mask"]
        resp_len = attn_mask[prompt_len:].sum().item()
        resp_ids = data_item.batch["responses"][:resp_len]
        return self.tokenizer.decode(resp_ids, skip_special_tokens=True)
    
    def _default_gt_converter(self, gt: Any) -> str:
        """默认ground truth转换器 (任务无关)"""
        if isinstance(gt, (list, tuple, np.ndarray)):
            if all(isinstance(x, (int, float)) for x in gt):
                # 数值型结构 (bbox/坐标等)
                return f"[{','.join(f'{x:.1f}' for x in gt)}]"
            return ", ".join(str(x) for x in gt)  # 列表/元组
        return str(gt)  # 其他类型直接转字符串
    
    def _default_answer_extractor(self, response: str) -> str:
        """默认答案提取器 (任务无关)"""
        # 优先提取\boxed{}内容
        boxed_match = re.search(r'\\boxed\{(.*?)\}', response, re.DOTALL)
        if boxed_match:
            return boxed_match.group(1).strip()
        
        # 备用：提取Answer is/Answer:/Final answer:模式
        patterns = [
            r'Answer is[:\s]*([^\n]+)',
            r'Answer[:\s]*([^\n]+)',
            r'Final answer[:\s]*([^\n]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # 最后手段：返回整个响应 (会获得低分)
        return response.strip()

    def __call__(self, data: DataProto, **kwargs) -> torch.Tensor:
        """主入口：计算奖励张量"""
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        uids = data.non_tensor_batch.get("uid", None)
        ground_truths = data.non_tensor_batch["ground_truth"]
        
        # 分组处理 (GRPO标准模式)
        groups = defaultdict(list)
        if uids is None:  # 无uid时每条数据独立成组
            for i in range(len(data)):
                groups[i].append(i)
        else:
            for i, uid in enumerate(uids):
                groups[uid].append(i)
        
        # 处理每组数据
        for indices in groups.values():
            group_responses = [self._decode_response(data[i]) for i in indices]
            gt = ground_truths[indices[0]]  # 同组共享ground truth
            scores = self._compute_group_reward(group_responses, gt)
            
            for i, score in zip(indices, scores):
                prompt_len = data[i].batch["prompts"].shape[-1]
                resp_len = data[i].batch["attention_mask"][prompt_len:].sum().item()
                if resp_len > 0:
                    reward_tensor[i, resp_len - 1] = score
        
        return reward_tensor

    def _compute_group_reward(self, responses: List[str], ground_truth: Any) -> List[float]:
        """计算单组奖励 (核心逻辑)"""
        # 1. 预处理: 转换ground truth & 提取预测答案
        gt_str = self._default_gt_converter(ground_truth)
        pred_answers = [self._default_answer_extractor(resp) for resp in responses]
        
        # 2. 计算Qwen3嵌入
        try:
            # (a) 预测答案嵌入
            pred_answer_embeddings = self.local_embedding_fn(pred_answers)
            
            # (b) 真实答案嵌入 (单次计算)
            gt_embedding = self.local_embedding_fn([gt_str])[0]
            gt_embeddings = [gt_embedding] * len(responses)
            
            # (c) 完整响应嵌入
            full_response_embeddings = self.local_embedding_fn(responses)
        except Exception as e:
            print(f"❌ Embedding failed: {str(e)}")
            return [-0.5] * len(responses)  # 安全回退
        
        
        # 3. 调用通用奖励函数 (固定权重)
        return compute_universal_reward(
            responses=responses,
            ground_truths=[gt_str] * len(responses),
            pred_answer_embeddings=pred_answer_embeddings,
            gt_embeddings=gt_embeddings,
            full_response_embeddings=full_response_embeddings,
            exact_threshold=0.9,
            partial_threshold=0.7,
            weights=(0.1, 0.5, 0.4)  # 固定权重比例
        )