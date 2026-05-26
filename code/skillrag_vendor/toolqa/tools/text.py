"""Text retrieval tools for ToolQA."""

import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

_retriever_cache: dict[tuple[str, str], "TextRetriever"] = {}
_retriever_cache_lock = threading.Lock()


def get_shared_retriever(
    corpus_path: Path,
    text_field: str,
    model_name: str = EMBED_MODEL_NAME,
    top_k: int = 3,
) -> "TextRetriever":
    """Return a shared TextRetriever instance."""
    key = (str(corpus_path), text_field)
    if key in _retriever_cache:
        return _retriever_cache[key]

    with _retriever_cache_lock:
        if key not in _retriever_cache:
            retriever = TextRetriever(corpus_path, text_field, model_name, top_k)
            retriever._ensure_index()
            _retriever_cache[key] = retriever
    return _retriever_cache[key]


def _compute_cache_path(corpus_path: Path, model_name: str) -> Path:
    """Compute the embeddings cache file path."""
    corpus_hash = hashlib.md5(str(corpus_path).encode()).hexdigest()[:8]
    model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
    cache_dir = Path.home() / ".cache" / "toolqa_embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"embed_{corpus_hash}_{model_hash}.npz"


class TextRetriever:
    """Semantic text retriever using sentence-transformers cosine similarity."""

    def __init__(
        self,
        corpus_path: Path,
        text_field: str,
        model_name: str = EMBED_MODEL_NAME,
        top_k: int = 3,
    ):
        self.corpus_path = Path(corpus_path)
        self.text_field = text_field
        self.model_name = model_name
        self.top_k = top_k

        self._model = None
        self._texts: list[str] | None = None
        self._embeddings_tensor: torch.Tensor | None = None
        self._init_lock = threading.Lock()

    def _get_target_device(self) -> str:
        """Return the CUDA device assigned to the current process."""
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            return f"cuda:{current_device}"
        return "cpu"

    def _ensure_index(self) -> None:
        if self._embeddings_tensor is not None:
            return

        with self._init_lock:
            if self._embeddings_tensor is not None:
                return

            cache_path = _compute_cache_path(self.corpus_path, self.model_name)
            texts_path = cache_path.with_suffix(".texts.json")

            lock_path = cache_path.with_suffix(".lock")

            if cache_path.exists() and texts_path.exists():
                try:
                    embeddings_np = np.load(cache_path)["embeddings"]
                    with open(texts_path, "r", encoding="utf-8") as f:
                        self._texts = json.load(f)
                except (EOFError, OSError, ValueError, json.JSONDecodeError):
                    cache_path.unlink(missing_ok=True)
                    texts_path.unlink(missing_ok=True)
                    lock_path.unlink(missing_ok=True)
                    # fall through to compute embeddings below
                else:
                    target_device = self._get_target_device()
                    import sentence_transformers
                    self._model = sentence_transformers.SentenceTransformer(self.model_name, device=target_device)
                    self._embeddings_tensor = torch.from_numpy(embeddings_np).to(target_device)
                    return

            while lock_path.exists():
                import time
                time.sleep(0.5)
                if cache_path.exists() and texts_path.exists():
                    break

            if cache_path.exists() and texts_path.exists():
                try:
                    embeddings_np = np.load(cache_path)["embeddings"]
                    with open(texts_path, "r", encoding="utf-8") as f:
                        self._texts = json.load(f)
                except (EOFError, OSError, ValueError, json.JSONDecodeError):
                    cache_path.unlink(missing_ok=True)
                    texts_path.unlink(missing_ok=True)
                    lock_path.unlink(missing_ok=True)
                else:
                    target_device = self._get_target_device()
                    import sentence_transformers
                    self._model = sentence_transformers.SentenceTransformer(self.model_name, device=target_device)
                    self._embeddings_tensor = torch.from_numpy(embeddings_np).to(target_device)
                    return

            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_path, "w") as lock_file:
                lock_file.write("computing")

            try:
                texts: list[str] = []
                with open(self.corpus_path, "r", encoding="utf-8") as file_obj:
                    for line in file_obj:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        texts.append(item[self.text_field])
                self._texts = texts

                import sentence_transformers

                target_device = self._get_target_device()
                self._model = sentence_transformers.SentenceTransformer(self.model_name, device=target_device)

                embeddings = self._model.encode(texts, show_progress_bar=True, convert_to_tensor=True)
                if not isinstance(embeddings, torch.Tensor):
                    embeddings = torch.tensor(embeddings, device=target_device)
                norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
                norms = torch.where(norms == 0, torch.ones_like(norms), norms)
                embeddings_normalized = embeddings / norms
                self._embeddings_tensor = embeddings_normalized

                embeddings_np = embeddings_normalized.cpu().numpy()
                np.savez(cache_path, embeddings=embeddings_np)
                with open(texts_path, "w", encoding="utf-8") as f:
                    json.dump(self._texts, f)
            finally:
                if lock_path.exists():
                    lock_path.unlink()

    def query(self, query_text: str, top_k: int | None = None) -> str:
        query_start = time.time()
        self._ensure_index()
        k = top_k or self.top_k

        encode_start = time.time()
        query_emb = self._model.encode([query_text], convert_to_tensor=True)
        encode_time = time.time() - encode_start
        print(f"[TextRetriever] model.encode took {encode_time:.3f}s")

        if not isinstance(query_emb, torch.Tensor):
            query_emb = torch.tensor(query_emb, device=self._embeddings_tensor.device)

        norm_start = time.time()
        query_norm = torch.norm(query_emb, p=2, dim=1, keepdim=True)
        query_norm = torch.where(query_norm == 0, torch.ones_like(query_norm), query_norm)
        query_emb_normalized = query_emb / query_norm
        norm_time = time.time() - norm_start

        matmul_start = time.time()
        scores = (self._embeddings_tensor @ query_emb_normalized.T).squeeze()
        matmul_time = time.time() - matmul_start

        argsort_start = time.time()
        top_indices = torch.argsort(scores, descending=True)[:k].tolist()
        argsort_time = time.time() - argsort_start

        results = [self._texts[i] for i in top_indices]
        total_time = time.time() - query_start
        print(f"[TextRetriever] query total: {total_time:.3f}s (encode={encode_time:.3f}s, norm={norm_time:.3f}s, matmul={matmul_time:.3f}s, argsort={argsort_time:.3f}s)")
        return "\n".join(results)
