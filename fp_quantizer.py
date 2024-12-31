import torch
import torch.nn.functional as F
from torch.autograd import Function

class RoundStraightThrough(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad

round_ste_func = RoundStraightThrough.apply

def quantize_to_fpx(x_float: torch.Tensor, 
                    e_bits: torch.Tensor,
                    m_bits: torch.Tensor,
                    maxval: torch.Tensor,
                    minval: torch.Tensor,
                    sign_bits: int=1,) -> torch.Tensor:
    # According to the paper(Qualcomm AI Research: FP8 Quantization: The Power of the Exponent),
    # The bias/maxval is calculated as follows:
    paper_bias = 2**(e_bits - 1) 
    paper_maxval = (2 - 2**(-m_bits.float())) * 2**(2**e_bits - paper_bias - 1)

    # When m_bits.dtype = torch.int64, 2**(-m_bits) is error ! m_bits need to convert to float dtype
    bias = 2**e_bits - torch.log2(maxval) + torch.log2(2-2**(-m_bits.float())) - 1

    x_clip = torch.clamp(x_float, min=minval, max=maxval)
    log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(x_clip)) + bias)).detach(), 1.0)
    scales = 2.0 ** (log_scales - m_bits - bias)
    fpx = round_ste_func(x_clip / scales) * scales
    return fpx

if __name__ == "__main__":
    x_float = torch.rand([3, 3], dtype=torch.float32)
    x_e4m3 = x_float.to(torch.float8_e4m3fn)

    fp8 = quantize_to_fpx(x_float, 
                          e_bits=torch.tensor(4), 
                          m_bits=torch.tensor(3), 
                          maxval=torch.tensor(448),
                          minval=torch.tensor(-448))


    print("Simulate Quantized FP8 Tensor:", fp8)
    print("Torch Cast FP8 Tensor:", x_e4m3)