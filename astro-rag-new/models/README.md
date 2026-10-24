# Local model artifacts

The OpenVINO NPU model directories are intentionally empty in this source package. Model exports and LoRA/base-model weights are large, environment-specific artifacts and are not redistributed in this ZIP.

To prepare a local NPU export, install the model-preparation requirements and follow the instructions in the main README:

```powershell
pip install -r requirements-local-model.txt
python scripts/prepare_npu_model.py
```

The preparation script downloads the Qwen3 base model and the configured LoRA adapter, merges the adapter, and exports an int4 OpenVINO model. It may require several GB of free RAM/disk and Hugging Face access for the adapter. If the output is stored outside `models/astro-qwen-npu/`, set `OPENVINO_NPU_MODEL_DIR` to the export directory. Do not expect the included source ZIP alone to contain runnable model weights.
