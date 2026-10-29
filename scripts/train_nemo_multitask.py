"""Quick training entry for a Canary-style prompted AED multitask model.

Trains a small FastConformer-Transformer `EncDecMultiTaskModel` from scratch on
the manifests produced by build_datasets.py, reusing the existing
CanaryMultilingualDataModule (lhotse batching + prompted minibatches).

Supports single-task ASR, single-task AST, or joint multitask via --task.

Example (Nahuatl ASR smoke run):
    python scripts/train_nemo_multitask.py --task asr --language-mode azz --max-epochs 20
"""

import json
import math
import os
import sys
from pathlib import Path

# Allow running as `python scripts/train_nemo_multitask.py` from the repo root
# (data.* / utils.* imports resolve against the repo root, not scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Cap PyTorch's CUDA caching allocator so DDP can't hoard system RAM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import click

# Manifest directory names (build_datasets.py --language-mode) mapped to the
# language IDs used in Canary prompts and sub-tokenizer names.
# shortcut: this is the prompting registry inline; it lives in models/prompting.py
# in the full plan layout.
LANG_CODES = {"azz": "azz", "map": "map", "que": "que", "an4": "en"}
AST_TARGET_LANG = "es"
SPECIAL_TASK_TOKENS = ["transcribe", "translate"]


def langs_for_task(task: str, lang: str) -> list[str]:
    """Sub-tokenizers needed for a task (prompt control tokens live in spl_tokens)."""
    if task == "asr":
        return [lang]
    if task == "ast":
        return [AST_TARGET_LANG]
    return [lang, AST_TARGET_LANG]


def normalize_entry(entry: dict, lang: str, task: str) -> dict:
    """Rewrite the placeholder language fields build_datasets.py writes.

    The prompt/metrics machinery (CanaryPromptFormatter, multitask metric
    constraints) reads source_lang/target_lang, so a Nahuatl ASR row must say
    source==target==<lang>, and an AST row source=<lang>, target=es.
    shortcut: task is trusted from the manifest dir / "task" field as built by
    build_datasets.py; upgrade when manifests carry reliable per-entry langs.
    """
    t = entry.get("task", task)
    entry["task"] = t
    entry["source_lang"] = lang
    entry["target_lang"] = lang if t == "asr" else AST_TARGET_LANG
    entry.setdefault("pnc", "no")
    return entry


def normalize_manifests(
    manifests_dir: str, language_mode: str, task: str, out_root: str
):
    """Copy manifests/<lang>/<task>/ to a normalized JSONL tree (never touch the originals)."""
    lang = LANG_CODES[language_mode]
    base = os.path.join(manifests_dir, language_mode, task)
    out_base = os.path.join(out_root, language_mode, task)
    os.makedirs(out_base, exist_ok=True)

    counts = {}
    durations = {}
    for split, fname in [
        ("train", "train_manifest.json"),
        ("validation", "val_manifest.json"),
        ("test", "test_manifest.json"),
    ]:
        src = os.path.join(base, fname)
        if not os.path.exists(src):
            sys.exit(f"Manifest not found: {src}. Run build_datasets.py first.")
        n = 0
        dur = 0.0
        with (
            open(src, encoding="utf-8") as fin,
            open(os.path.join(out_base, fname), "w", encoding="utf-8") as fout,
        ):
            for line in fin:
                if not line.strip():
                    continue
                entry = normalize_entry(json.loads(line), lang, task)
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n += 1
                dur += float(entry.get("duration", 0.0))
        counts[split] = n
        durations[split] = dur
    return out_base, counts, durations


def ensure_tokenizers(
    tokenizer_root: str, langs: list[str], train_manifest: str, vocab_size: int
):
    """Build the spl_tokens tokenizer and, if missing, per-language SentencePiece models.

    Tokenizer is trained only on the TRAIN manifest (never val/test), per NeMo
    fine-tuning guidance.
    """
    from nemo.collections.common.tokenizers.canary_tokenizer import CanaryTokenizer
    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import (
        create_spt_model,
    )

    CanaryTokenizer.build_special_tokenizer(
        SPECIAL_TASK_TOKENS + langs, os.path.join(tokenizer_root, "spl_tokens")
    )
    for lang in langs:
        model_dir = os.path.join(tokenizer_root, lang)
        if os.path.exists(os.path.join(model_dir, "tokenizer.model")):
            continue
        texts = [
            json.loads(line)["text"]
            for line in open(train_manifest, encoding="utf-8")
            if line.strip()
            and json.loads(line).get("target_lang") == lang
            and json.loads(line).get("text")
        ]
        if not texts:
            sys.exit(
                f"No training text found for language '{lang}' in {train_manifest}; "
                "check --task and the manifests."
            )
        os.makedirs(model_dir, exist_ok=True)
        text_path = os.path.join(model_dir, "train_text.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write("\n".join(texts) + "\n")
        create_spt_model(
            text_path,
            vocab_size=vocab_size,
            sample_size=-1,
            do_lower_case=False,
            output_dir=model_dir,
        )


@click.command()
@click.option("--task", type=click.Choice(["asr", "ast", "multitask"]), default="asr")
@click.option(
    "--language-mode", default="azz", help="Manifest dir name from build_datasets.py."
)
@click.option("--config", default="configs/nemo/multitask_nah_es.yaml")
@click.option("--max-epochs", type=int, default=50)
@click.option("--lr", type=float, default=3e-4)
@click.option("--vocab-size", type=int, default=2000)
@click.option(
    "--batch-size", type=int, default=8, help="Static batch size for val/test."
)
@click.option(
    "--batch-duration",
    type=float,
    default=60.0,
    help="Seconds of audio per GPU train batch.",
)
@click.option("--val-check-interval", type=int, default=100)
@click.option("--num-data-workers", type=int, default=4)
@click.option("--devices", type=int, default=1)
@click.option("--models-dir", default="models")
@click.option("--manifests-dir", default="manifests")
@click.option(
    "--tokenizer-root", default=None, help="Defaults to tokenizers/<language-mode>."
)
@click.option("--no-resume", is_flag=True, help="Ignore an existing last.ckpt.")
def main(
    task,
    language_mode,
    config,
    max_epochs,
    lr,
    vocab_size,
    batch_size,
    batch_duration,
    val_check_interval,
    num_data_workers,
    devices,
    models_dir,
    manifests_dir,
    tokenizer_root,
    no_resume,
):
    os.makedirs("logs", exist_ok=True)
    from utils.logging_utils import get_logger, setup_logging

    setup_logging(
        log_file=f"logs/train_{language_mode}_{task}.log",
        level=os.environ.get("LOG_LEVEL", "INFO"),
    )
    logger = get_logger(__name__)

    import lightning.pytorch as L
    import torch
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from lightning.pytorch.loggers import TensorBoardLogger
    from nemo.collections.asr.models import EncDecMultiTaskModel
    from omegaconf import OmegaConf

    from data.datamodule import CanaryMultilingualDataModule

    logger.info("Setting up torch float32 matmul precision to medium")
    torch.set_float32_matmul_precision("medium")

    lang = LANG_CODES[language_mode]
    langs = langs_for_task(task, lang)

    norm_root = os.path.join("outputs", "manifests_norm")
    norm_base, counts, durations = normalize_manifests(
        manifests_dir, language_mode, task, norm_root
    )
    logger.info(
        f"Normalized manifests -> {norm_base} "
        f"(train: {counts['train']} utts, {durations['train']:.0f}s)"
    )

    tokenizer_root = tokenizer_root or os.path.join("tokenizers", language_mode)
    ensure_tokenizers(
        tokenizer_root,
        langs,
        os.path.join(norm_base, "train_manifest.json"),
        vocab_size,
    )
    logger.info(
        f"Tokenizers ready under {tokenizer_root} for langs: spl_tokens + {langs}"
    )

    cfg = OmegaConf.load(config)
    mdl = cfg.model
    mdl.tokenizer.langs = OmegaConf.create(
        {
            "spl_tokens": {
                "dir": os.path.join(tokenizer_root, "spl_tokens"),
                "type": "bpe",
            },
            **{
                l: {"dir": os.path.join(tokenizer_root, l), "type": "bpe"}
                for l in langs
            },
        }
    )
    mdl.optim.lr = lr
    # warmup over 5% of the run; steps/epoch estimated from the duration-budget
    # batching (shortcut: approximate, good enough for warmup).
    est_steps_per_epoch = max(1, math.ceil(durations["train"] / batch_duration))
    mdl.optim.sched.warmup_steps = int(0.05 * max_epochs * est_steps_per_epoch)
    logger.info(
        f"Est. {est_steps_per_epoch} steps/epoch -> {max_epochs * est_steps_per_epoch} "
        f"total, warmup={mdl.optim.sched.warmup_steps}"
    )

    os.makedirs(models_dir, exist_ok=True)
    exp_name = f"fastconformer_aed_{language_mode}_{task}"
    use_gpu = torch.cuda.is_available()
    strategy = "ddp" if devices > 1 else "auto"
    logger.info(
        f"Trainer: devices={devices}, strategy={strategy}, "
        f"precision={'bf16-mixed' if use_gpu else '32-true'}, accumulate_grad_batches=4"
    )
    trainer = L.Trainer(
        devices=devices,
        accelerator="gpu" if use_gpu else "cpu",
        strategy=strategy,
        max_epochs=max_epochs,
        val_check_interval=val_check_interval,
        precision="bf16-mixed" if use_gpu else "32-true",
        accumulate_grad_batches=4,
        gradient_clip_val=1.0,
        logger=TensorBoardLogger(save_dir="logs", name=exp_name),
        callbacks=[
            ModelCheckpoint(
                dirpath=models_dir,
                filename=f"{exp_name}_{{step:06d}}",
                every_n_train_steps=val_check_interval,
                save_top_k=3,
                monitor="val_wer",
                mode="min",
                save_last=True,
            ),
            EarlyStopping(monitor="val_wer", patience=3, mode="min"),
            LearningRateMonitor(logging_interval="step"),
        ],
        use_distributed_sampler=False,
    )

    logger.info(f"Building model from {config}")
    model = EncDecMultiTaskModel(cfg=mdl, trainer=trainer)
    logger.info(model.summarize())

    data_module = CanaryMultilingualDataModule(
        tokenizer=model.tokenizer,
        prompt_formatter=model.prompt,
        language_mode=language_mode,
        task=task,
        manifests_dir=norm_root,
        num_workers=num_data_workers,
        batch_size=batch_size,
        batch_duration=batch_duration,
    )

    last_ckpt = os.path.join(models_dir, "last.ckpt")
    ckpt_path = (
        None if no_resume else (last_ckpt if os.path.exists(last_ckpt) else None)
    )
    if ckpt_path:
        logger.info(f"Resuming from {ckpt_path}")
    trainer.fit(model, data_module, ckpt_path=ckpt_path)

    final_model_path = os.path.join(models_dir, f"{language_mode}_{task}_final.nemo")
    logger.info(f"FINISHED TRAINING. Saving final model at {final_model_path}")
    model.save_to(final_model_path)


if __name__ == "__main__":
    main()
