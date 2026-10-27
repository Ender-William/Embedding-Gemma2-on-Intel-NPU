# EmbeddingGemma 2 Service

本地多模态 EmbeddingGemma 2 服务，支持 ONNX Runtime、OpenVINO CPU/GPU/NPU，
并通过 `EmbeddingService` 统一管理下载、量化、加载、推理和卸载。

## 目录

```text
embedding_service/       # 服务、类型、下载、量化、设备能力
  core/                  # ONNX/OpenVINO 推理核心与多模态流水线
examples/                # 使用示例
tests/                   # 单元测试和测试素材
tools/                   # 导出、量化、模型校验、基准脚本
embedding_service/model/
├── google--embeddinggemma-2/                   # 下载的 HuggingFace checkpoint
├── onnx/                                        # FP32 ONNX
├── openvino/                                    # FP16 OpenVINO
├── quantized/embeddinggemma-2__int8__intel_npu/ # INT8_W8
└── manifest.json
cache/                   # 编译临时缓存，使用后自动删除
```

## 安装

```powershell
uv sync
```

基础环境使用 CUDA 版 torch/torchvision：图片处理器需要 torchvision，CUDA torch
同时提供 ONNX Runtime CUDA EP 所需的 cuBLAS/cuDNN DLL。OpenVINO NPU 不受影响。
视频使用 PyAV 解码，不依赖系统 FFmpeg DLL。

## EmbeddingService

```python
from embedding_service import EmbeddingService

service = EmbeddingService(
    models_root="./model",
    cache_root="./cache",
    default_device="cuda",
)

service.load("./embedding_service/model/onnx", device="cuda", warmup=True)

query = service.run(["太阳系中最大的行星是哪一颗？"], prompt="query")
docs = service.run([
    "木星是太阳系中体积最大的行星。",
    "Python 是一种编程语言。",
], prompt="document")

scores = service.similarity(docs.embeddings, query.embeddings)
service.unload()
```

下载模型：

```python
downloaded = service._download_model(
    repo_name="google/embeddinggemma-2",
    source="hf-mirror",
    revision="main",
)
```

不传 `model_dir` 时默认下载到 `embedding_service/model/`。
镜像下载默认关闭 Xet 并使用单线程；如遇 429，请等待限流窗口或设置 `HF_TOKEN`。
下载过程使用 `tqdm` 显示当前已下载字节数。

量化模型：

```python
artifact = service._quantization(
    model_path=downloaded.local_path,
    multi_modal={
        "vision": ".../vision_encoder.onnx",
        "audio": ".../audio_encoder.onnx",
        "lm": ".../lm_embeds.onnx",
        "token_embeddings": ".../token_embeddings.npy",
    },
    quant="INT8",
    device="intel_npu",
    backend="openvino",
)
service._load(artifact, device="intel_npu")
```

`multi_modal=None` 表示文本模型。全模态量化所需的 ONNX 组件缺失时，
Service 会从 HuggingFace 下载目录自动导出到 `embedding_service/model/onnx/`。
量化前会一次性校验全部组件，缺文件不会再等到文本压缩完成后才报错。
FP32/FP16/INT8_W8 的设备组合在加载前会校验，不支持的组合会抛出明确异常。

## 生命周期

```text
UNLOADED -> LOADING -> READY -> UNLOADING -> UNLOADED
                   \-> FAILED
```

- 加载新模型会自动卸载旧模型。
- `unload()` 释放 ONNX/OpenVINO/CUDA 资源并清空 cache。
- 卸载后调用 `run()` 会抛出 `ServiceStateError`。
- 同一时间只保留一个活动模型，避免显存和 NPU 内存叠加。

## 临时编译缓存

OpenVINO NPU/GPU 编译时，`cache_root` 下会创建唯一的临时子目录；编译完成、
加载完成或卸载时自动删除。模型运行时不再使用持久化的 `ov_cache` 或预编译 blob。
Windows 下的只读 compile blob 会先清除只读属性再删除。

## 运行示例

```powershell
uv run python examples/demo.py list
uv run python examples/demo.py demo --mode all
uv run python examples/demo.py demo --device intel_npu --mode all
uv run python examples/infer.py
uv run python examples/pytorch_text.py
uv run python examples/pytorch_multimodal.py
```

## 验证

```powershell
uv run python -m unittest discover -s tests -v
uv run python tools/verify_pipeline.py
uv run python tools/validate_models.py --hash
uv run python tools/benchmark.py
```

`verify_pipeline.py` 覆盖文本、图像、音频、视频、混合输入和长序列回退。
模型 manifest 位于 `embedding_service/model/manifest.json`。
