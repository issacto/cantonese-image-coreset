%%writefile train_sft.py
"""
Fine-tune IBM Granite Vision 4.1-4B on Docmatix (first 100K samples)
using TRL's SFTTrainer — the standard HuggingFace SFT pipeline.

Usage:
  python train_sft.py                    # LoRA, streaming 100K samples
  python train_sft.py --full_finetune    # full FT (needs A100-80GB)
  python train_sft.py --load_in_4bit     # QLoRA
"""

import io
import logging
import argparse

import torch
import PIL.Image
from datasets import load_dataset, IterableDataset
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_ID = "HuggingFaceM4/Docmatix"
MAX_SAMPLES = 100_000
MAX_IMAGE_SIDE = 1120  # Granite Vision native tile size; larger = wasted VRAM


# ─── Dataset ────────────────────────────────────────────────────────────────

def ensure_pil(img, max_side: int = MAX_IMAGE_SIDE) -> PIL.Image.Image | None:
    try:
        if isinstance(img, PIL.Image.Image):
            pil = img.convert("RGB")
        elif isinstance(img, (bytes, bytearray)):
            pil = PIL.Image.open(io.BytesIO(img)).convert("RGB")
        elif isinstance(img, dict) and "bytes" in img:
            pil = PIL.Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        else:
            return None

        # Downscale if either side exceeds the cap — preserves aspect ratio.
        # This runs BEFORE the processor ever sees the image, so large PDF pages
        # never allocate an oversized pixel tensor.
        w, h = pil.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            pil = pil.resize((int(w * scale), int(h * scale)), PIL.Image.LANCZOS)
        return pil
    except Exception:
        return None


def build_streaming_dataset(
    max_samples: int = MAX_SAMPLES,
    max_qa_per_sample: int = 4,
    max_images_per_doc: int = 8,
) -> IterableDataset:
    """
    Each Docmatix document has:
      - images: list of page images (all pages of the PDF)
      - texts:  list of QA dicts, each QA may reference any/all pages

    We chunk QA pairs into groups of max_qa_per_sample and emit ONE row per
    chunk. Every chunk re-passes all page images so the model always has full
    visual context. This keeps sequence length bounded regardless of how many
    QA pairs a document has, which was the root cause of the OOM.

    Row format:
      - messages : list[dict]  — multi-turn chat with one <image> per page
      - images   : list[PIL.Image]  — all pages (capped at max_images_per_doc)
    """
    raw = load_dataset(
        DATASET_ID, "images", split="train", streaming=True, trust_remote_code=True
    )

    def _generate():
        count = 0
        for sample in raw:
            if count >= max_samples:
                break

            # ── collect valid page images, capped at max_images_per_doc ────
            images = []
            for img_raw in (sample.get("images") or [])[:max_images_per_doc]:
                pil = ensure_pil(img_raw)
                if pil is not None:
                    images.append(pil)

            if not images:
                continue

            # ── collect all valid QA pairs ──────────────────────────────────
            qa_pairs = []
            for qa in (sample.get("texts") or []):
                question = (qa.get("user") or "").strip()
                answer   = (qa.get("assistant") or "").strip()
                if question and answer:
                    qa_pairs.append((question, answer))

            if not qa_pairs:
                continue

            # ── chunk QA pairs, emit one row per chunk ──────────────────────
            # Each chunk is at most max_qa_per_sample turns, so sequence length
            # stays bounded regardless of how many QAs the document has.
            # Every chunk re-passes the full images list so visual context is
            # always present. The first turn of each chunk carries the image
            # tokens; subsequent turns in the chunk are text-only.
            for chunk_start in range(0, len(qa_pairs), max_qa_per_sample):
                if count >= max_samples:
                    break

                chunk = qa_pairs[chunk_start : chunk_start + max_qa_per_sample]
                messages = []

                for i, (question, answer) in enumerate(chunk):
                    if i == 0:
                        # First turn of every chunk: attach all page images
                        first_content = [{"type": "image"} for _ in images]
                        first_content.append({"type": "text", "text": question})
                        messages.append({"role": "user", "content": first_content})
                    else:
                        messages.append({
                            "role": "user",
                            "content": [{"type": "text", "text": question}],
                        })
                    messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": answer}],
                    })

                yield {"messages": messages, "images": images}
                count += 1

                if count % 5_000 == 0:
                    logger.info(f"Streamed {count:,} / {max_samples:,} samples")

    return IterableDataset.from_generator(_generate)


# ─── Collator ───────────────────────────────────────────────────────────────

class MultimodalCollator:
    """
    TRL's default collator expects pre-tokenized 'input_ids' in each sample,
    but our dataset yields raw 'messages' + 'images'. This collator calls the
    processor itself to tokenize + encode images, then builds the batch with
    proper label masking (loss only on assistant tokens).
    """

    def __init__(self, processor, max_length: int = 4096):
        self.processor = processor
        self.max_length = max_length
        # Token id that marks the start of the assistant turn — used for
        # completion-only label masking so we don't train on prompt tokens.
        self.assistant_token_ids = processor.tokenizer.encode(
            "<|assistant|>", add_special_tokens=False
        )

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        input_ids_list      = []
        attention_mask_list = []
        labels_list         = []
        pixel_values_list   = []
        image_sizes_list    = []

        for ex in examples:
            messages = ex["messages"]
            images   = ex.get("images") or []

            # Apply chat template → formatted string
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            # Encode with all images. We do NOT resize or truncate here.
            # image_sizes in the encoding reflects the true image dimensions
            # which pack_image_features uses to unpad features — it MUST match
            # the actual images passed or the token/feature count will diverge.
            encoding = self.processor(
                text=text,
                images=images if images else None,
                return_tensors="pt",
                padding=False,
            )

            input_ids      = encoding["input_ids"][0]
            attention_mask = encoding["attention_mask"][0]

            # If over max_length: drop whole images from the END one at a time
            # and re-encode. We never slice input_ids because the <image> tokens
            # and image_sizes must always stay consistent with pixel_values.
            # NOTE: ensure_pil already resizes images before they reach here,
            # so this loop is a last-resort safety net, not the primary defence.
            while input_ids.shape[0] > self.max_length and images:
                images = images[:-1]
                seen, total = 0, len(images)
                clean = []
                for msg in messages:
                    content = msg["content"] if isinstance(msg["content"], list) else [{"type": "text", "text": msg["content"]}]
                    nc = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image":
                            if seen < total:
                                nc.append(item)
                                seen += 1
                        else:
                            nc.append(item)
                    clean.append({**msg, "content": nc})
                messages = clean
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                encoding = self.processor(
                    text=text,
                    images=images if images else None,
                    return_tensors="pt",
                    padding=False,
                )
                input_ids      = encoding["input_ids"][0]
                attention_mask = encoding["attention_mask"][0]

            # ── Label masking: -100 on prompt, real token ids on assistant ──
            labels = input_ids.clone()
            ids    = input_ids.tolist()
            labels[:] = -100
            i = 0
            while i < len(ids):
                if ids[i : i + len(self.assistant_token_ids)] == self.assistant_token_ids:
                    j = i + len(self.assistant_token_ids)
                    while j < len(ids):
                        labels[j] = input_ids[j]
                        j += 1
                        if input_ids[j - 1] == self.processor.tokenizer.eos_token_id:
                            break
                    i = j
                else:
                    i += 1

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

            if "pixel_values" in encoding:
                pixel_values_list.append(encoding["pixel_values"][0])
            # FIX: LlavaNext model forward() iterates image_sizes to slice
            # image features per image. Without it you get:
            #   TypeError: 'NoneType' object is not iterable
            if "image_sizes" in encoding:
                image_sizes_list.append(encoding["image_sizes"][0])

        # Pad all sequences to the longest in the batch
        pad_id  = self.processor.tokenizer.pad_token_id
        max_len = max(x.shape[0] for x in input_ids_list)

        def pad(tensor, pad_value):
            diff = max_len - tensor.shape[0]
            if diff == 0:
                return tensor
            return torch.cat([tensor, tensor.new_full((diff,), pad_value)])

        batch = {
            "input_ids":      torch.stack([pad(x, pad_id) for x in input_ids_list]),
            "attention_mask": torch.stack([pad(x, 0)      for x in attention_mask_list]),
            "labels":         torch.stack([pad(x, -100)   for x in labels_list]),
        }
        if pixel_values_list:
            try:
                batch["pixel_values"] = torch.stack(pixel_values_list)
            except RuntimeError:
                batch["pixel_values"] = pixel_values_list
        if image_sizes_list:
            # image_sizes is a 2-D tensor: (num_images, 2) per sample
            try:
                batch["image_sizes"] = torch.stack(image_sizes_list)
            except RuntimeError:
                batch["image_sizes"] = image_sizes_list

        return batch


# ─── Model ───────────────────────────────────────────────────────────────────

def load_model_and_processor(args):
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ── FIX 1: Delegate ALL missing tokenizer attrs/methods to processor.tokenizer
    # TRL's SFTTrainer probes processing_class for tokenizer attributes like
    # .pad_token, .pad_token_id, .convert_tokens_to_ids(), .vocab_size, etc.
    # Multimodal processors (LlavaNextProcessor, GraniteProcessor) don't expose
    # these directly. Subclassing with __getattr__ delegation catches every
    # missing attr in one place, so we never hit AttributeError again regardless
    # of which TRL version probes which method next.
    tokenizer = processor.tokenizer

    class _PatchedProcessor(processor.__class__):
        def __getattr__(self, name):
            # Called only when normal lookup fails — try the tokenizer first.
            try:
                return getattr(tokenizer, name)
            except AttributeError:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'"
                )

    processor.__class__ = _PatchedProcessor

    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Auto-detect flash attention
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        logger.info("Flash Attention 2 detected.")
    except ImportError:
        attn_impl = "sdpa"
        logger.info("Flash Attention 2 not found, using sdpa.")

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    return model, processor


# ─── LoRA config ─────────────────────────────────────────────────────────────

def make_lora_config(args) -> LoraConfig | None:
    if args.full_finetune:
        return None
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_finetune",      action="store_true")
    parser.add_argument("--load_in_4bit",       action="store_true")
    parser.add_argument("--lora_r",             type=int,   default=16)
    parser.add_argument("--lora_alpha",         type=int,   default=32)
    parser.add_argument("--lora_dropout",       type=float, default=0.05)
    parser.add_argument("--max_samples",        type=int,   default=MAX_SAMPLES)
    parser.add_argument("--max_length",         type=int,   default=4096)
    parser.add_argument("--batch_size",         type=int,   default=1)
    parser.add_argument("--grad_accum",         type=int,   default=16)
    parser.add_argument("--lr",                 type=float, default=2e-4)
    parser.add_argument("--epochs",             type=int,   default=1)
    parser.add_argument("--output_dir",         type=str,   default="./checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--model_id",           type=str,   default="ibm-granite/granite-vision-3.3-2b")
    parser.add_argument("--max_images_per_doc", type=int,   default=8,
                        help="Cap pages per document to avoid OOM on long PDFs")
    parser.add_argument("--max_qa_per_sample",  type=int,   default=4,
                        help="Max QA pairs per training row; controls sequence length")
    parser.add_argument("--max_steps",          type=int,   default=-1,
                        help="Override computed max_steps (useful for quick smoke-tests)")
    args = parser.parse_args()

    logger.info("Building streaming Docmatix dataset …")
    train_dataset = build_streaming_dataset(
        max_samples=args.max_samples,
        max_qa_per_sample=args.max_qa_per_sample,
        max_images_per_doc=args.max_images_per_doc,
    )

    # ── FIX 3: Streaming datasets have no __len__, so the LR scheduler cannot
    # derive the number of steps from epochs. We compute max_steps explicitly:
    #   max_steps = ceil(max_samples / (batch_size * grad_accum))
    # This is equivalent to 1 epoch over the streamed data.
    # Pass --max_steps N to override (e.g. for a quick smoke-test).
    if args.max_steps > 0:
        max_steps = args.max_steps
    else:
        effective_batch = args.batch_size * args.grad_accum
        max_steps = (args.max_samples + effective_batch - 1) // effective_batch
    logger.info(
        f"max_steps={max_steps:,}  "
        f"(max_samples={args.max_samples:,}, effective_batch={args.batch_size * args.grad_accum})"
    )

    model, processor = load_model_and_processor(args)
    lora_config = make_lora_config(args)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        # num_train_epochs is NOT used — streaming datasets require max_steps.
        max_steps=max_steps,

        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        optim="adamw_torch_fused",
        learning_rate=args.lr,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,

        bf16=True,
        tf32=True,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        max_seq_length=args.max_length,
        dataset_text_field=None,
        remove_unused_columns=False,

        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        report_to="tensorboard",
        logging_dir="./logs",

        dataloader_drop_last=True,
        seed=42,

        # ── FIX 2: Must be False when world_size == 1 (single GPU) ───────────
        # TRL sets this True by default but it's only valid for multi-GPU runs.
        average_tokens_across_devices=False,
    )

    # FIX 4: Pass our custom collator so raw messages+images get processed.
    # TRL's default collator expects pre-tokenized 'input_ids' — ours calls
    # the processor on the fly to tokenize text and encode images.
    collator = MultimodalCollator(processor, max_length=args.max_length)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=processor,
        data_collator=collator,
        peft_config=lora_config,
    )

    logger.info("Starting SFT training …")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = f"{args.output_dir}/final"
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    logger.info(f"Saved to {final_dir}")


if __name__ == "__main__":
    main()