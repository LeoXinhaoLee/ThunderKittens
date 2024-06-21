#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>


extern void  prefill_loop_body(torch::Tensor W1,
                     torch::Tensor XA, torch::Tensor XB, torch::Tensor XC,
                     torch::Tensor Out,
                     cudaStream_t stream);

extern void  prefill_loop_body_ref(torch::Tensor W1,
                         torch::Tensor XA, torch::Tensor XB, torch::Tensor XC,
                         torch::Tensor Out)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    prefill_loop_body(W1, XA, XB, XC, Out, stream);
}

extern void  prefill_whole_loop(torch::Tensor W1,
                               torch::Tensor XA, torch::Tensor XB, torch::Tensor XC,
                               torch::Tensor Out,
                               cudaStream_t stream);

extern void  prefill_whole_loop_ref(torch::Tensor W1,
                                   torch::Tensor XA, torch::Tensor XB, torch::Tensor XC,
                                   torch::Tensor Out)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    prefill_whole_loop(W1, XA, XB, XC, Out, stream);
}

extern void  prefill_whole_loop_LN_bias(torch::Tensor W1, torch::Tensor b1,
                                        torch::Tensor ln_weight, torch::Tensor ln_bias,
                                        torch::Tensor cumsum_matrix, torch::Tensor make_last_b_matrix,
                                        torch::Tensor make_last_coeff_1_matrix,
                                        torch::Tensor XA, torch::Tensor XB, torch::Tensor XC, torch::Tensor Coeff,
                                        torch::Tensor Out,
                                        cudaStream_t stream);

extern void  prefill_whole_loop_LN_bias_ref(torch::Tensor W1, torch::Tensor b1,
                                            torch::Tensor ln_weight, torch::Tensor ln_bias,
                                            torch::Tensor cumsum_matrix, torch::Tensor make_last_b_matrix,
                                            torch::Tensor make_last_coeff_1_matrix,
                                            torch::Tensor XA, torch::Tensor XB, torch::Tensor XC, torch::Tensor Coeff,
                                            torch::Tensor Out)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    prefill_whole_loop_LN_bias(W1, b1, ln_weight, ln_bias,
                               cumsum_matrix, make_last_b_matrix,
                               make_last_coeff_1_matrix,
                               XA, XB, XC, Coeff, Out, stream);
}

extern void  prefill_whole_loop_LN_bias_fp16(torch::Tensor W1, torch::Tensor b1,
                                             torch::Tensor ln_weight, torch::Tensor ln_bias,
                                             torch::Tensor cumsum_matrix, torch::Tensor make_last_b_matrix,
                                             torch::Tensor make_last_coeff_1_matrix,
                                             torch::Tensor XA, torch::Tensor XB, torch::Tensor XC, torch::Tensor Coeff,
                                             torch::Tensor Out,
                                             cudaStream_t stream);

extern void  prefill_whole_loop_LN_bias_fp16_ref(torch::Tensor W1, torch::Tensor b1,
                                                 torch::Tensor ln_weight, torch::Tensor ln_bias,
                                                 torch::Tensor cumsum_matrix, torch::Tensor make_last_b_matrix,
                                                 torch::Tensor make_last_coeff_1_matrix,
                                                 torch::Tensor XA, torch::Tensor XB, torch::Tensor XC, torch::Tensor Coeff,
                                                 torch::Tensor Out)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    prefill_whole_loop_LN_bias_fp16(W1, b1, ln_weight, ln_bias,
                                    cumsum_matrix, make_last_b_matrix,
                                    make_last_coeff_1_matrix,
                                    XA, XB, XC, Coeff, Out, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for warp test"; // optional module docstring
    m.def("prefill_loop_body", &prefill_loop_body_ref);
    m.def("prefill_whole_loop", &prefill_whole_loop_ref);
    m.def("prefill_whole_loop_LN_bias", &prefill_whole_loop_LN_bias_ref);
    m.def("prefill_whole_loop_LN_bias_fp16", &prefill_whole_loop_LN_bias_fp16_ref);
}