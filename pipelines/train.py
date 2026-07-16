"""LoRA fine-tuning of Gemma into a function-calling specialist.

This is the one *heavy* step. It is written the way you'd actually do it in
industry — Hugging Face Transformers + PEFT (LoRA) on a small Gemma base — but
kept behind lazy imports so the rest of the repo needs none of it. Run
``--dry-run`` (the default in CI and docs) to validate the data and print the
exact training plan without downloading a model; drop the flag on a GPU box with
``pip install -r requirements-ml.txt`` to train for real.

Why LoRA: we only adapt a few million low-rank parameters, so a specialist is
cheap to train, tiny to ship (a few MB of adapter weights), and safe to iterate
on. Why a small Gemma base: the whole premise is *edge* — the trained model must
fit and run on-device after quantization (see ``pipelines/quantize.py``).

Distillation seam: to push accuracy without hand labels, generate the
``completion`` field of the training data (``pipelines/generate_data.py``) with a
large teacher model instead of templates, then run this exact script on it. The
training code does not change — only where the labels come from.

Usage:
    python -m pipelines.train --dry-run                      # no deps needed
    python -m pipelines.train --data data/train.jsonl --epochs 3
"""

import argparse
import json
import os


def load_dataset(path):
    """Read the JSONL produced by generate_data.py and format prompt+target."""
    from functiongemma.prompt import build_prompt
    from functiongemma.tools import TOOLS

    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            ex = json.loads(line)
            prompt = build_prompt(TOOLS, ex["request"])
            rows.append({"text": prompt + " " + ex["completion"]})
    return rows


def dry_run(args, rows):
    """Validate inputs and print the plan — the deps-free path."""
    print("=== FunctionGemma LoRA training plan (dry run) ===")
    print(f"base model     : {args.base_model}")
    print(f"train examples : {len(rows)}")
    print(f"epochs         : {args.epochs}")
    print(f"lora rank/alpha: {args.lora_r}/{args.lora_alpha}")
    print(f"output dir     : {args.output_dir}")
    print("sample formatted example:")
    if rows:
        print("  " + rows[0]["text"].replace("\n", "\n  ")[:400])
    print("\nAdd `pip install -r requirements-ml.txt` and drop --dry-run to train.")


def train(args, rows):  # pragma: no cover - requires GPU + heavy deps
    """Real LoRA fine-tune. Imported lazily so CI never needs these packages."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16
    )
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=Dataset.from_list(rows),
        peft_config=lora,
        args=SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            bf16=True,
            logging_steps=10,
        ),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"saved LoRA adapter -> {args.output_dir}")


def main():
    ap = argparse.ArgumentParser(description="LoRA fine-tune Gemma for tool calling.")
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--base-model", default="google/gemma-2-2b")
    ap.add_argument("--output-dir", default="artifacts/adapter")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate + print plan without downloading a model")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(
            f"{args.data} not found — run `python -m pipelines.generate_data` first."
        )
    rows = load_dataset(args.data)
    if args.dry_run:
        dry_run(args, rows)
    else:
        train(args, rows)


if __name__ == "__main__":
    main()
