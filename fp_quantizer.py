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

def get_max_value(num_exponent_bits: int = 4, bias: int = 7):
    num_fraction_bits = 7 - num_exponent_bits
    scale = 2**-num_fraction_bits
    max_frac = 1 - scale
    max_value = 2 ** (2**num_exponent_bits - 1 - bias) * (1 + max_frac)
    return max_value

def quantize_to_fpx(x_float: torch.Tensor, 
                    e_bits: torch.Tensor,
                    m_bits: torch.Tensor,
                    maxval: torch.Tensor,
                    minval: torch.Tensor,
                    sign_bits: int=1,) -> torch.Tensor:

    b = 2**(e_bits - 1)
    expect_maxval = (2 - 2**(-m_bits)) * 2**(2**e_bits - b - 1)
    
    bias = 2**e_bits - torch.log2(maxval) + torch.log2(2-2**(-m_bits)) - 1
    # bias = 7
    x_clip = torch.clamp(x_float, min=minval, max=maxval)
    log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(x_clip)) + bias)).detach(), 1.0)
    scales = 2.0 ** (log_scales - m_bits - bias)
    fpx = round_ste_func(x_clip / scales) * scales
    import ipdb; ipdb.set_trace()
    return fpx

if __name__ == "__main__":
    x_float = torch.rand([3, 3], dtype=torch.float32)
    x_e4m3 = x_float.to(torch.float8_e4m3fn)

    fp8 = quantize_to_fpx(x_float, 
                          e_bits=torch.tensor(4), 
                          m_bits=torch.tensor(3), 
                          maxval=torch.tensor(448),
                          minval=torch.tensor(-448))

    import ipdb; ipdb.set_trace()
    print (fp8 - x_e4m3.float())