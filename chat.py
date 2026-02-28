from transformers import (
    TextStreamer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from argparse import ArgumentParser
import torch

# Monkey-patch for PyTorch MPS bug with torch.histc on integer tensors (used in MoE routing)
_original_histc = torch.histc
def _mps_histc(input, bins=100, min=0, max=0, *, out=None):
    if input.device.type == "mps" and input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        return _original_histc(input.float(), bins=bins, min=min, max=max, out=out)
    return _original_histc(input, bins=bins, min=min, max=max, out=out)
torch.histc = _mps_histc

parser = ArgumentParser()
parser.add_argument(
    "--model", "-m", type=str, required=True, help="Path to model directory"
)
parser.add_argument(
    "--precision",
    "-p",
    type=str,
    default="fp16",
    choices=["fp16", "bf16", "fp32"],
    help="Precision of model",
)
parser.add_argument(
    "--device",
    "-d",
    type=str,
    choices=["auto", "cuda", "cpu"],
    default="auto",
    help="Target device to process abliteration. Warning, bitsandbytes quantization DOES NOT support CPU",
)
parser.add_argument(
    "--max-new-tokens", "-n", type=int, default=256, help="Max new tokens to generate"
)
parser.add_argument(
    "--system-prompt", "-s", type=str, default=None, help="System prompt"
)
quant = parser.add_mutually_exclusive_group()
quant.add_argument(
    "--load-in-4bit",
    action="store_true",
    default=False,
    help="Load model in 4-bit precision using bitsandbytes",
)
quant.add_argument(
    "--load-in-8bit",
    action="store_true",
    default=False,
    help="Load model in 8-bit precision using bitsandbytes",
)
parser.add_argument(
    "--flash-attn", action="store_true", default=False, help="Use flash attention 2"
)
args = parser.parse_args()


if __name__ == "__main__":
    if args.precision == "fp16":
        precision = torch.float16
    elif args.precision == "bf16":
        precision = torch.bfloat16
    elif args.precision == "fp32":
        precision = torch.float32

    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=precision,
            bnb_4bit_use_double_quant=True,
        )
    elif args.load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=True,
        )
    else:
        quant_config = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=precision,
        low_cpu_mem_usage=True,
        device_map=args.device,
        quantization_config=quant_config,
        attn_implementation="flash_attention_2" if args.flash_attn else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, device_map=args.device
    )

    conversation = []
    if args.system_prompt is not None:
        print(f"System Prompt: {args.system_prompt}")
        conversation.append({"role": "system", "content": args.system_prompt})
    streamer = TextStreamer(tokenizer)
    print("Type /clear to clear history, /exit to quit.")
    while True:
        prompt = input("User> ")
        if prompt == "/clear":
            conversation = []
            print("! History cleared.")
            continue
        elif prompt == "/exit":
            break
        elif prompt == "":
            print("! Please type a message.")
            continue
        conversation.append({"role": "user", "content": prompt})
        toks = tokenizer.apply_chat_template(
            conversation=conversation, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        toks = toks.to(model.device)
        
        gen = model.generate(
            **toks, streamer=streamer, max_new_tokens=args.max_new_tokens
        )
        
        if hasattr(toks, "input_ids"):
            input_len = toks["input_ids"].shape[1]
        else:
            input_len = toks.shape[1]
            
        decoded = tokenizer.batch_decode(
            gen[:, input_len:], skip_special_tokens=True
        )
        conversation.append({"role": "assistant", "content": "".join(decoded)})
