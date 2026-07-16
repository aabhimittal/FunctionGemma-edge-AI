"""Edge export  —  quantize the trained model so it fits on-device.

A bf16 2B model is ~5 GB: too big for a phone. Quantization is what makes the
"edge" in the title real. The industrial options, all supported here behind
lazy imports:

  * **int4 (GGUF)** for llama.cpp / Ollama — the usual choice for phones &
    laptops. ~0.9 GB, CPU-friendly.
  * **int8 (ONNX Runtime)** for cross-platform mobile/embedded deployment.

``--dry-run`` (deps-free) prints the footprint/latency budget table so you can
reason about the trade-off before committing a GPU box to the conversion. The
resulting artifact path is what ``pipelines/promote.py`` records in the registry.

Usage:
    python -m pipelines.quantize --dry-run
    python -m pipelines.quantize --format gguf --adapter artifacts/adapter
"""

import argparse

# Rough, illustrative footprints for a ~2B base — the point is the trade-off,
# not exact numbers (which depend on the specific checkpoint).
FOOTPRINTS = {
    "bf16": {"size_gb": 5.0, "cpu_tok_s": 4, "notes": "training / server only"},
    "int8": {"size_gb": 2.5, "cpu_tok_s": 12, "notes": "ONNX Runtime, mobile"},
    "int4": {"size_gb": 0.9, "cpu_tok_s": 25, "notes": "GGUF, phone / laptop"},
}


def print_budget():
    print(f"{'precision':<8} {'size(GB)':>9} {'cpu tok/s':>10}  notes")
    for prec, f in FOOTPRINTS.items():
        print(f"{prec:<8} {f['size_gb']:>9} {f['cpu_tok_s']:>10}  {f['notes']}")


def export(args):  # pragma: no cover - requires heavy deps + a real model
    """Convert + quantize a trained model to an edge format."""
    if args.format == "onnx":
        from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        model = ORTModelForCausalLM.from_pretrained(args.adapter, export=True)
        quantizer = ORTQuantizer.from_pretrained(model)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
        quantizer.quantize(save_dir=args.output, quantization_config=qconfig)
    else:  # gguf via llama.cpp conversion tooling
        import subprocess

        subprocess.run(
            ["python", "convert_hf_to_gguf.py", args.adapter,
             "--outfile", f"{args.output}/model-q4.gguf", "--outtype", "q4_k_m"],
            check=True,
        )
    print(f"exported {args.format} artifact -> {args.output}")


def main():
    ap = argparse.ArgumentParser(description="Quantize FunctionGemma for the edge.")
    ap.add_argument("--format", choices=["gguf", "onnx"], default="gguf")
    ap.add_argument("--adapter", default="artifacts/adapter")
    ap.add_argument("--output", default="artifacts/edge")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=== Edge footprint budget ===")
    print_budget()
    if args.dry_run:
        print("\nDry run: no conversion performed. "
              "Install requirements-ml.txt and drop --dry-run to export.")
    else:
        export(args)


if __name__ == "__main__":
    main()
