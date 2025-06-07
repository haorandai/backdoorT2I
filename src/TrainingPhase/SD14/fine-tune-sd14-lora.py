import argparse
import gc
import os
import torch
from huggingface_hub import login

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=5e-05)
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    return parser.parse_args()

def main():
    args = parse_args()
    main_save_path = "xxx"
    os.makedirs(main_save_path, exist_ok=True)
    save_model_path = os.path.join(main_save_path, args.dataset_name)
    os.makedirs(save_model_path, exist_ok=True)

    login(token=args.hf_token)
    gc.collect()
    torch.cuda.empty_cache()
    
    train_cmd = [
        "accelerate", "launch",
        "train_text_to_image_lora.py",
        f"--pretrained_model_name_or_path={args.model_name}",
        f"--dataset_name={args.dataset_name}",
        f"--output_dir={save_model_path}",
        f"--resolution={args.resolution}",
        "--center_crop",
        "--random_flip",
        f"--train_batch_size={args.train_batch_size}",
        f"--gradient_accumulation_steps={args.gradient_accumulation_steps}",
        "--gradient_checkpointing",
        f"--mixed_precision={args.mixed_precision}",
        f"--learning_rate={args.learning_rate}",
        f"--lr_scheduler={args.lr_scheduler}",
        f"--lr_warmup_steps={args.lr_warmup_steps}",
        f"--rank={args.rank}"
    ]
    
    train_cmd.append(f"--max_train_steps={args.max_train_steps}" if args.max_train_steps else f"--num_train_epochs={args.num_train_epochs}")
    
    try:
        os.system(" ".join(train_cmd))
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()