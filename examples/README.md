# EmbeddingService Demo

查看已注册模型：

```powershell
uv run python examples/demo.py list
```

运行全部文本/图像/音频/视频/混合检索：

```powershell
uv run python examples/demo.py demo --mode all
```

使用 INT8 NPU 模型：

```powershell
uv run python examples/demo.py demo `
  --device intel_npu `
  --model embeddinggemma-2__int8__intel_npu `
  --mode all
```

下载模型：

```powershell
uv run python examples/demo.py download `
  --repo google/embeddinggemma-2 `
  --source hf-mirror `
  --max-workers 1
```

量化为 INT8：

```powershell
uv run python examples/demo.py quantize `
  --model-path .\embedding_service\model\google--embeddinggemma-2 `
  --quant INT8 `
  --device intel_npu `
  --overwrite
```

第一次运行全模态量化时，如果 `embedding_service/model/onnx/` 还不存在，
Demo 会先从下载目录自动导出所需 ONNX 组件，再开始 NPU 量化。
只量化文本模型时加 `--text-only`。

Demo 会自动在 `examples/assets/` 生成图片、音频和视频测试素材。
