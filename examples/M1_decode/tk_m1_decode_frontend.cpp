#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>


extern void  decode_coeff_LN_bias_fp16(
        torch::Tensor W1, torch::Tensor b1,
        torch::Tensor W1_grad, torch::Tensor b1_grad,
        torch::Tensor ln_weight, torch::Tensor ln_bias,
        torch::Tensor XA, torch::Tensor XB, torch::Tensor XC,
        torch::Tensor token_idx, torch::Tensor ilr_gated,
        torch::Tensor Output, cudaStream_t stream
);

extern void  decode_coeff_LN_bias_fp16_ref(
        torch::Tensor W1, torch::Tensor b1,
        torch::Tensor W1_grad, torch::Tensor b1_grad,
        torch::Tensor ln_weight, torch::Tensor ln_bias,
        torch::Tensor XA, torch::Tensor XB, torch::Tensor XC,
        torch::Tensor token_idx, torch::Tensor ilr_gated,
        torch::Tensor Output
)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    decode_coeff_LN_bias_fp16(
            W1, b1,
            W1_grad, b1_grad,
            ln_weight, ln_bias,
            XA, XB, XC,
            token_idx, ilr_gated,
            Output, stream);
}

extern void load_store_vector_fp16(torch::Tensor XA, torch::Tensor Output, cudaStream_t stream);

extern void load_store_vector_fp16_ref(torch::Tensor XA, torch::Tensor Output)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    load_store_vector_fp16(XA, Output, stream);
}

/*
extern void outer_product_vector_fp16(torch::Tensor X, torch::Tensor Y, torch::Tensor Output, cudaStream_t stream);

extern void outer_product_vector_fp16_ref(torch::Tensor X, torch::Tensor Y, torch::Tensor Output)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    outer_product_vector_fp16(X, Y, Output, stream);
}
 */

extern void  decode_coeff_LN_bias_simplified_fp16(
        torch::Tensor W1, torch::Tensor b1,
        torch::Tensor W1_grad, torch::Tensor b1_grad,
        torch::Tensor ln_weight, torch::Tensor ln_bias,
        torch::Tensor XA, torch::Tensor XB, torch::Tensor XC,
        torch::Tensor token_idx, torch::Tensor ilr_gated,
        torch::Tensor Output, cudaStream_t stream
);

extern void  decode_coeff_LN_bias_simplified_fp16_ref(
        torch::Tensor W1, torch::Tensor b1,
        torch::Tensor W1_grad, torch::Tensor b1_grad,
        torch::Tensor ln_weight, torch::Tensor ln_bias,
        torch::Tensor XA, torch::Tensor XB, torch::Tensor XC,
        torch::Tensor token_idx, torch::Tensor ilr_gated,
        torch::Tensor Output
)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    decode_coeff_LN_bias_simplified_fp16(
            W1, b1,
            W1_grad, b1_grad,
            ln_weight, ln_bias,
            XA, XB, XC,
            token_idx, ilr_gated,
            Output, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for warp test"; // optional module docstring
    m.def("decode_coeff_LN_bias_fp16", &decode_coeff_LN_bias_fp16_ref);
    m.def("load_store_vector_fp16", &load_store_vector_fp16_ref);
//    m.def("outer_product_vector_fp16", &outer_product_vector_fp16_ref);
    m.def("decode_coeff_LN_bias_simplified_fp16", &decode_coeff_LN_bias_simplified_fp16_ref);
}