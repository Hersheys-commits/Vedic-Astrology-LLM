"""Merge the Astro Qwen LoRA and export an int4 OpenVINO model for Intel NPU."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import settings  # noqa: E402
from app.generation.qwen_client import (  # noqa: E402
    ADAPTER_MODEL,
    _load_adapter_tokenizer,
)

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def prepare_model(output_dir: Path) -> None:
    try:
        import torch
        from huggingface_hub import get_token, snapshot_download
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Model preparation needs PyTorch, Hugging Face Hub, PEFT, and "
            "Transformers. Install requirements-local-model.txt first."
        ) from exc

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )

    token = get_token()
    local_candidate = Path(ADAPTER_MODEL).expanduser()
    if local_candidate.is_dir():
        adapter_dir = local_candidate.resolve()
    elif not token:
        raise SystemExit(
            f"{ADAPTER_MODEL} is private. Run `hf auth login` with an account "
            "that has read access to the adapter."
        )
    else:
        adapter_dir = Path(snapshot_download(ADAPTER_MODEL, token=token))
    chat_template_path = adapter_dir / "chat_template.jinja"

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="astro-qwen-npu-export-",
        dir=output_dir.parent,
    ) as temporary:
        temporary_dir = Path(temporary)
        merged_dir = temporary_dir / "merged"
        export_dir = temporary_dir / "openvino"

        print(
            "Loading the unquantized Qwen3 base to merge the LoRA. "
            "This needs several GB of free RAM and disk space.",
            flush=True,
        )
        tokenizer = _load_adapter_tokenizer(
            AutoTokenizer,
            adapter_dir,
            token,
            base_model=BASE_MODEL,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            token=token,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        adapted = PeftModel.from_pretrained(
            base,
            adapter_dir,
            low_cpu_mem_usage=True,
        )
        merged = adapted.merge_and_unload(safe_merge=True)
        merged.save_pretrained(
            merged_dir,
            safe_serialization=True,
            max_shard_size="2GB",
        )
        tokenizer.save_pretrained(merged_dir)

        cli = shutil.which("optimum-cli")
        if cli is None:
            raise SystemExit(
                "optimum-cli was not found. Install requirements-openvino-npu.txt "
                "in the active Astro RAG environment."
            )
        subprocess.run(
            [
                cli,
                "export",
                "openvino",
                "--model",
                str(merged_dir),
                "--task",
                "text-generation-with-past",
                "--weight-format",
                "int4",
                "--sym",
                "--ratio",
                "1.0",
                "--group-size",
                "128",
                str(export_dir),
            ],
            check=True,
        )
        tokenizer.save_pretrained(export_dir)
        chat_template = (
            chat_template_path.read_text(encoding="utf-8")
            if chat_template_path.is_file()
            else tokenizer.chat_template
        )
        (export_dir / "chat_template.jinja").write_text(
            chat_template,
            encoding="utf-8",
        )

        if not (export_dir / "config.json").is_file() or not list(
            export_dir.glob("*.xml")
        ):
            raise RuntimeError(
                "OpenVINO export completed without the expected config.json and "
                "IR XML files."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        for item in export_dir.iterdir():
            destination = output_dir / item.name
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)

    print(f"OpenVINO NPU model ready at: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.openvino_npu_model_dir,
        help="Output artifact directory (default: OPENVINO_NPU_MODEL_DIR).",
    )
    args = parser.parse_args()
    prepare_model(args.output_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
