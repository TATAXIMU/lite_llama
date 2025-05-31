import torch
from typing import Tuple, Dict


def quantize_tensor(t: torch.Tensor, bits: int = 4) -> Tuple[torch.Tensor, float]:
    """Quantize a tensor to int representation.

    This implements a simple symmetric per-tensor quantization used as a
    lightweight approximation of GPTQ.
    """
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    max_val = t.abs().max()
    if max_val == 0:
        scale = 1.0
    else:
        scale = qmax / max_val
    qt = torch.clamp((t * scale).round(), qmin, qmax).to(torch.int8)
    return qt, float(scale)


def dequantize_tensor(qt: torch.Tensor, scale: float) -> torch.Tensor:
    return qt.float() / scale


def compress_state_dict(state_dict: Dict[str, torch.Tensor], bits: int = 4) -> Dict:
    q_state: Dict[str, torch.Tensor] = {}
    scales: Dict[str, float] = {}
    for name, param in state_dict.items():
        if param.dtype in (torch.float16, torch.float32):
            qt, scale = quantize_tensor(param.float(), bits)
            q_state[name] = qt
            scales[name] = scale
        else:
            q_state[name] = param
    return {"state_dict": q_state, "scales": scales, "bits": bits}


def save_quantized(state_dict: Dict[str, torch.Tensor], path: str, bits: int = 4) -> None:
    data = compress_state_dict(state_dict, bits)
    torch.save(data, path)


def load_quantized(model: torch.nn.Module, path: str) -> None:
    data = torch.load(path, map_location=model.device)
    q_state = data.get("state_dict", {})
    scales = data.get("scales", {})
    for name, param in model.named_parameters():
        if name in q_state:
            qval = q_state[name]
            if isinstance(qval, torch.Tensor) and qval.dtype == torch.int8:
                scale = scales.get(name, 1.0)
                deq = dequantize_tensor(qval, scale).to(param.dtype)
                param.data.copy_(deq)
            else:
                param.data.copy_(qval)
    model.eval()
