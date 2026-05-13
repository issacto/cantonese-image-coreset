%%writefile push_to_hub.py
"""
Merge LoRA adapter weights into the base model and push to HuggingFace Hub.
"""

import argparse
import logging

import torch
from transformers import LlavaNextProcessor, AutoModelForVision2Seq
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", type=str, default="ibm-granite/granite-vision-3.2-2b")
    parser.add_argument("--adapter_dir", type=str, default="./checkpoints/final")
    parser.add_argument("--hub_repo", type=str, required=True)
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--adapter_only", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def get_dtype(dtype_str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_str]


def push_adapter_only(args):
    logger.info(f"Loading adapter from {args.adapter_dir} …")
    from peft import PeftConfig
    PeftConfig.from_pretrained(args.adapter_dir)

    logger.info(f"Pushing adapter to {args.hub_repo} …")
    from huggingface_hub import HfApi
    api = HfApi(token=args.hub_token)
    api.create_repo(repo_id=args.hub_repo, private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=args.adapter_dir,
        repo_id=args.hub_repo,
        commit_message="Upload LoRA adapter",
    )
    logger.info("Done — adapter pushed.")


def push_merged(args):
    import os
    import shutil
    from huggingface_hub import HfApi

    dtype = get_dtype(args.dtype)
    local_dir = "./merged_model_upload"

    logger.info(f"Loading base model {args.base_model_id} …")
    base_model = AutoModelForVision2Seq.from_pretrained(
        args.base_model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info(f"Loading processor from {args.base_model_id} …")
    processor = LlavaNextProcessor.from_pretrained(args.base_model_id, trust_remote_code=True)

    logger.info(f"Loading LoRA adapter from {args.adapter_dir} …")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    logger.info("Merging LoRA weights …")
    model = model.merge_and_unload()
    logger.info("Merge complete.")

    # Save everything locally first so you can verify before pushing
    logger.info(f"Saving merged model + processor to {local_dir} …")
    os.makedirs(local_dir, exist_ok=True)
    model.save_pretrained(local_dir)
    processor.save_pretrained(local_dir)

    # Log exactly what files will be uploaded so you can verify
    saved_files = os.listdir(local_dir)
    logger.info(f"Files in {local_dir}:")
    for f in sorted(saved_files):
        logger.info(f"  {f}")

    expected = {"preprocessor_config.json", "processor_config.json", "tokenizer_config.json"}
    missing = expected - set(saved_files)
    if missing:
        # Fall back: snapshot the base model processor files directly
        logger.warning(f"Missing processor files: {missing} — copying from base model snapshot")
        from huggingface_hub import snapshot_download
        base_snapshot = snapshot_download(args.base_model_id, ignore_patterns=["*.bin", "*.safetensors"])
        for fname in missing:
            src = os.path.join(base_snapshot, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(local_dir, fname))
                logger.info(f"  Copied {fname} from base snapshot")

    logger.info(f"Uploading to {args.hub_repo} …")
    api = HfApi(token=args.hub_token)
    api.create_repo(repo_id=args.hub_repo, private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=local_dir,
        repo_id=args.hub_repo,
        commit_message="Upload merged LoRA model + full processor",
    )
    logger.info(f"Done — https://huggingface.co/{args.hub_repo}")
    

def main():
    args = parse_args()

    if args.hub_token:
        from huggingface_hub import login
        login(token=args.hub_token)

    if args.adapter_only:
        push_adapter_only(args)
    else:
        push_merged(args)


if __name__ == "__main__":
    main()



# !python push_to_hub.py \
#   --base_model_id ibm-granite/granite-vision-3.3-2b \
#   --adapter_dir ./checkpoints/final \
#   --hub_repo Issactoto/granite-vision-3.3-2b-enhanced-coreset \
#   --hub_token XX