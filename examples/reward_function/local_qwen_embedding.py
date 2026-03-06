import requests
import numpy as np
from typing import List

class LocalQwen3Embedding:
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')
        try:
            resp = requests.get(f"{self.server_url}/v1/models", timeout=5)
            resp.raise_for_status()  # 确保 200 OK
            data = resp.json()
            expected_model = "qwen3-embedding-4b"
            if "data" in data and any(m["id"] == expected_model for m in data["data"]):
                print(f"✅ Embedding service ready at {self.server_url}. Model: {expected_model}")
                self._available = True
            else:
                raise ValueError(f"Health check failed: Model '{expected_model}' not found in {data}")
        except Exception as e:
            print(f"⚠️ Embedding service down ({e}), falling back to random")
            self._available = False

    def encode(self, texts: List[str]) -> List[np.ndarray]:
        if not self._available:
            return [np.random.rand(2560).astype(np.float32) for _ in texts]
        try:
            resp = requests.post(
                f"{self.server_url}/v1/embeddings",
                json={
                    "model": "qwen3-embedding-4b",
                    "input": texts
                },
                timeout=30
            )
            print(f"DEBUG: Status: {resp.status_code}, Body preview: {resp.text[:200]}")
            resp.raise_for_status()
            data = resp.json()
            if "data" not in data:
                raise ValueError("Invalid response format")
            embeddings = [np.array(item["embedding"], dtype=np.float32) for item in data["data"]]
            return embeddings
        except Exception as e:
            print(f"❌ Embedding failed: {e}, fallback to random")
            return [np.random.rand(2560).astype(np.float32) for _ in texts]