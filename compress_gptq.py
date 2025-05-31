import argparse
import torch
from pathlib import Path
from lite_llama.utils.gptq_quantization import save_quantized


def main():
    parser = argparse.ArgumentParser(description="Compress a lite-llama model using GPTQ-style quantization")
    parser.add_argument("checkpoint_dir", help="Directory containing fp16 weights (.pth)")
    parser.add_argument("output", help="Path to save quantized checkpoint")
    parser.add_argument("--bits", type=int, default=4, help="Number of bits for quantization")
    args = parser.parse_args()

    ckpt = sorted(Path(args.checkpoint_dir).glob("*.pth"))
    if not ckpt:
        parser.error("No checkpoint file found in directory")
    state_dict = torch.load(str(ckpt[0]), map_location="cpu")
    save_quantized(state_dict, args.output, bits=args.bits)
    print(f"Quantized checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()
