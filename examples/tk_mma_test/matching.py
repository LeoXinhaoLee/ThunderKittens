import copy
import torch

import tk_mma_test

def pt_matmul_add_bf16(W, b, X, **kwargs):
    Z = torch.matmul(X.float(), W.float()) + b.float()
    Z = Z.bfloat16()
    return Z

def tk_matmul_add_bf16(W, b, X, **kwargs):
    output_tensor = torch.empty_like(X)
    tk_mma_test.matmul_add_bf16(W, b, X, output_tensor)
    return output_tensor


if __name__ == "__main__":

    torch.manual_seed(0)
    input_dtype = torch.bfloat16
    BS, CS, HF, = 512, 16, 64

    original_state_dict = {
        'W': torch.randn(BS, HF, HF, device='cuda', dtype=input_dtype) * 0.2,
        'b': torch.randn(BS, 1, HF, device='cuda', dtype=input_dtype).expand(-1, CS, -1).contiguous() * 0.2,  # @xinhao: replicate to be loaded by reg tile
    }
    original_input_dict = {
        'X': torch.randn(BS, CS, HF, device='cuda', dtype=input_dtype) * 0.2,
    }

    ########## PyTorch  ##########
    pt_state_dict = copy.deepcopy(original_state_dict)
    pt_input_dict = copy.deepcopy(original_input_dict)
    pt_output = pt_matmul_add_bf16(**pt_input_dict, **pt_state_dict)
    ##############################


    ########## TK ##########
    tk_state_dict = copy.deepcopy(original_state_dict)
    tk_input_dict = copy.deepcopy(original_input_dict)
    tk_output = tk_matmul_add_bf16(**tk_input_dict, **tk_state_dict)
    ##############################

    print(f'Pytorch v.s TK, dtype: {input_dtype}')
    diff = torch.abs(pt_output - tk_output)
    print(f'Output diff: max={diff.max()}, median={diff.median()}\n')
