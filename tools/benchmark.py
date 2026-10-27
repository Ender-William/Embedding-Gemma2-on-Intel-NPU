"""用测试文档列表对比 PyTorch / ONNX Runtime / OpenVINO NPU。"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from embedding_service.core import runtime_env  # noqa: E402,F401
from tests.data.documents import DOCUMENTS, QUERIES

QUERY_PROMPT = "task: search result | query: "
DOC_PROMPT = "title: none | text: "

query_text = QUERY_PROMPT + QUERIES[0]
doc_texts = [DOC_PROMPT + d for d in DOCUMENTS]


def report(name: str, query_emb: np.ndarray, doc_embs: np.ndarray, elapsed: float):
    scores = doc_embs @ query_emb
    top = np.argsort(scores)[::-1][:10]
    print(f"\n[{name}] {len(doc_embs)} docs encoded in {elapsed:.2f}s")
    for i in top:
        print(f"  {scores[i]:.8f}  {DOCUMENTS[i][:36]}")


# 后端 1：PyTorch (sentence-transformers, CUDA)
def run_pytorch():
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        str(ROOT / "embedding_service" / "model" / "google--embeddinggemma-2"),
        device="cuda",
    )
    t0 = time.perf_counter()
    q = model.encode(query_text, prompt_name="query")
    d = model.encode(doc_texts, prompt_name="document")
    elapsed = time.perf_counter() - t0
    report("PyTorch/CUDA", np.asarray(q), np.asarray(d), elapsed)


# 后端 2：ONNX Runtime (CUDA EP)
def run_onnx():
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / "embedding_service" / "model" / "google--embeddinggemma-2")
    )
    sess = ort.InferenceSession(
        str(ROOT / "embedding_service" / "model" / "onnx" / "embeddinggemma2_text.onnx"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    out_name = sess.get_outputs()[0].name

    def encode(texts):
        enc = tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="np"
        )
        embs = sess.run(
            [out_name],
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            },
        )[0]
        return embs / np.linalg.norm(embs, axis=1, keepdims=True)

    t0 = time.perf_counter()
    q = encode([query_text])[0]
    d = encode(doc_texts)
    elapsed = time.perf_counter() - t0
    report("ONNX Runtime/CUDA", q, d, elapsed)


# 后端 3/4：OpenVINO (NPU)，model_path 指定 FP16 或 W8 量化的 IR
# 注意：该模型的图在此 NPU 驱动上 batch>1 无法编译，只能 [1, seq_len]；
# 吞吐靠 AsyncInferQueue 并行多个推理请求来提升
def run_openvino(model_path: str, name: str, seq_len: int = 512, jobs: int = 1):
    import openvino as ov
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / "embedding_service" / "model" / "google--embeddinggemma-2")
    )
    core = ov.Core()
    model = core.read_model(model_path)
    model.reshape({"input_ids": [1, seq_len], "attention_mask": [1, seq_len]})
    try:
        compiled = core.compile_model(
            model, "NPU", {"INFERENCE_PRECISION_HINT": "f16", "CACHE_DIR": "./ov_cache"}
        )
    except Exception as e:
        msg = str(e).strip().splitlines()[-1] if str(e).strip() else repr(e)
        print(f"\n[{name} [s={seq_len}]] compile FAILED: {msg}")
        return

    def tokenize_one(text):
        enc = tokenizer(
            [text],
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="np",
        )
        return {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }

    def encode_sync(texts):
        embs = []
        for text in texts:
            result = compiled(tokenize_one(text))
            embs.append(np.asarray(result[compiled.output(0)], dtype=np.float32)[0])
        return np.stack(embs)

    def encode_async(texts):
        results = [None] * len(texts)

        def on_done(request, idx):
            results[idx] = np.asarray(
                request.get_output_tensor(0).data, dtype=np.float32
            )[0].copy()

        queue = ov.AsyncInferQueue(compiled, jobs=jobs)
        queue.set_callback(on_done)
        for i, text in enumerate(texts):
            queue.start_async(tokenize_one(text), userdata=i)
        queue.wait_all()
        return np.stack(results)

    def encode(texts):
        embs = encode_sync(texts) if jobs <= 1 else encode_async(texts)
        return embs / np.linalg.norm(embs, axis=1, keepdims=True)

    t0 = time.perf_counter()
    q = encode([query_text])[0]
    d = encode(doc_texts)
    elapsed = time.perf_counter() - t0
    report(f"{name} [s={seq_len}, jobs={jobs}]", q, d, elapsed)


if __name__ == "__main__":
    print(f"query: {QUERIES[0]}  |  documents: {len(DOCUMENTS)}")
    run_pytorch()
    run_onnx()
    # NPU：同步基线 vs AsyncInferQueue 并行请求（batch>1 的图在该驱动上无法编译）
    fp16_ir = str(ROOT / "embedding_service" / "model" / "openvino" / "embeddinggemma2_text_openvino.xml")
    w8_ir = str(
        ROOT
        / "embedding_service"
        / "model"
        / "quantized"
        / "embeddinggemma-2__int8__intel_npu"
        / "embeddinggemma2_text_w8.xml"
    )
    run_openvino(fp16_ir, "OpenVINO/NPU (fp16)", seq_len=512, jobs=1)
    run_openvino(fp16_ir, "OpenVINO/NPU (fp16)", seq_len=256, jobs=4)
    run_openvino(w8_ir, "OpenVINO/NPU (int8 w8)", seq_len=256, jobs=4)
