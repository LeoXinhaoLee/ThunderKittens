import copy
import torch

import tk_m1_decode

def pt_load_store_vector(X, **kwargs):
    output = X
    return output

def tk_load_store_vector(X, **kwargs):
    output_tensor = torch.empty_like(X)
    tk_m1_decode.load_store_vector_fp16(X, output_tensor)
    return output_tensor

def pt_outer_product(X, Y, **kwargs):
    output = Y @ Z  # [B,nh,HF,1] @ [B,nh,1,HF] -> [B,nh,HF,HF]
    return output

def tk_outer_product(X, Y, **kwargs):
    BS, NH, CS, HF = X.shape
    output = torch.empty((BS, NH, HF, HF), device=X.device, dtype=X.dtype)
    tk_m1_decode.outer_product_vector_fp16(X, Y, output_tensor)
    return output


if __name__ == "__main__":

    torch.manual_seed(0)
    input_dtype = torch.float16
    BS, NH, CS, HF, = 512, 32, 1, 64


    original_input_dict = {
        'X': torch.randn(BS, NH, CS, HF, device='cuda', dtype=input_dtype) * 0.2,
        'Y': torch.randn(BS, NH, HF, CS, device='cuda', dtype=input_dtype) * 0.2,
    }

    # ########## PyTorch  ##########
    # pt_input_dict = copy.deepcopy(original_input_dict)
    # pt_output = pt_load_store_vector(**pt_input_dict)
    # ##############################
    #
    # ########## TK ##########
    # tk_input_dict = copy.deepcopy(original_input_dict)
    # tk_output = tk_load_store_vector(**tk_input_dict)
    # ##############################

    ########## PyTorch  ##########
    pt_input_dict = copy.deepcopy(original_input_dict)
    pt_output = pt_outer_product(**pt_input_dict)
    ##############################

    ########## TK ##########
    tk_input_dict = copy.deepcopy(original_input_dict)
    tk_output = tk_outer_product(**tk_input_dict)
    ##############################

    print(f'Pytorch v.s TK, dtype: {input_dtype}')
    diff = torch.abs(pt_output - tk_output)
    print(f'Output diff: max={diff.max()}, median={diff.median()}\n')
