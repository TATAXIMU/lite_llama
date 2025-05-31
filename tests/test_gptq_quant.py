import torch
from lite_llama.utils.gptq_quantization import quantize_tensor, dequantize_tensor

def test_quant_dequant_close():
    x = torch.randn(100)
    q, s = quantize_tensor(x, bits=4)
    x2 = dequantize_tensor(q, s)
    assert torch.allclose(x, x2, atol=0.1)
