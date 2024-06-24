#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>


extern void  prefill_whole_loop(torch::Tensor W1, torch::Tensor W2,
                                torch::Tensor XA, torch::Tensor XB, torch::Tensor XC,
                                torch::Tensor Out,
                                cudaStream_t stream);

extern void  prefill_whole_loop_ref(torch::Tensor W1, torch::Tensor W2,
                                    torch::Tensor XA, torch::Tensor XB, torch::Tensor XC,
                                    torch::Tensor Out)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    prefill_whole_loop(W1, W2,
                       XA, XB, XC,
                       Out,
                       stream);
}

extern void  prefill_whole_loop_gelu(torch::Tensor W1, torch::Tensor W2,
                                     torch::Tensor XA, torch::Tensor XB, torch::Tensor XC,
                                     torch::Tensor Out,
                                     cudaStream_t stream);

extern void  prefill_whole_loop_gelu_ref(torch::Tensor W1, torch::Tensor W2,
                                         torch::Tensor XA, torch::Tensor XB, torch::Tensor XC,
                                         torch::Tensor Out)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    prefill_whole_loop_gelu(W1, W2,
                            XA, XB, XC,
                            Out,
                            stream);
}

extern void  prefill_whole_loop_fp16(torch::Tensor W1, torch::Tensor W2,
                                     torch::Tensor XA, torch::Tensor XB, torch::Tensor XC,
                                     torch::Tensor Out,
                                     cudaStream_t stream);

extern void  prefill_whole_loop_ref_fp16(torch::Tensor W1, torch::Tensor W2,
                                         torch::Tensor XA, torch::Tensor XB, torch::Tensor XC,
                                         torch::Tensor Out)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    prefill_whole_loop_fp16(W1, W2,
                       XA, XB, XC,
                       Out,
                       stream);
}


extern void  prefill_whole_loop_gelu_coeff_bias_fp16(torch::Tensor W1,
                                                torch::Tensor W2,
                                                torch::Tensor b1,
                                                torch::Tensor b2,
                                                torch::Tensor ln_weight,
                                                torch::Tensor ln_bias,
                                                torch::Tensor cumsum_matrix,
                                                torch::Tensor make_last_b_matrix,
                                                torch::Tensor make_last_coeff_1_matrix,
                                                torch::Tensor make_last_coeff_2_matrix,
                                                torch::Tensor XA,
                                                torch::Tensor XB,
                                                torch::Tensor XC,
                                                torch::Tensor Coeff,
                                                torch::Tensor Output,
                                                cudaStream_t stream);

extern void  prefill_whole_loop_gelu_coeff_bias_fp16_ref(torch::Tensor W1,
                                                    torch::Tensor W2,
                                                    torch::Tensor b1,
                                                    torch::Tensor b2,
                                                    torch::Tensor ln_weight,
                                                    torch::Tensor ln_bias,
                                                    torch::Tensor cumsum_matrix,
                                                    torch::Tensor make_last_b_matrix,
                                                    torch::Tensor make_last_coeff_1_matrix,
                                                    torch::Tensor make_last_coeff_2_matrix,
                                                    torch::Tensor XA,
                                                    torch::Tensor XB,
                                                    torch::Tensor XC,
                                                    torch::Tensor Coeff,
                                                    torch::Tensor Output)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    prefill_whole_loop_gelu_coeff_bias_fp16(W1,
                                            W2,
                                            b1,
                                            b2,
                                            ln_weight,
                                            ln_bias,
                                            cumsum_matrix,
                                            make_last_b_matrix,
                                            make_last_coeff_1_matrix,
                                            make_last_coeff_2_matrix,
                                            XA,
                                            XB,
                                            XC,
                                            Coeff,
                                            Output,
                                            stream);
}

extern void  prefill_whole_loop_gelu_coeff_bias_LN_fp16(torch::Tensor W1,
                                                        torch::Tensor W2,
                                                        torch::Tensor b1,
                                                        torch::Tensor b2,
                                                        torch::Tensor ln_weight,
                                                        torch::Tensor ln_bias,
                                                        torch::Tensor cumsum_matrix,
                                                        torch::Tensor make_last_b_matrix,
                                                        torch::Tensor make_last_coeff_1_matrix,
                                                        torch::Tensor make_last_coeff_2_matrix,
                                                        torch::Tensor XA,
                                                        torch::Tensor XB,
                                                        torch::Tensor XC,
                                                        torch::Tensor Coeff,
                                                        torch::Tensor Output,
                                                        cudaStream_t stream);

extern void  prefill_whole_loop_gelu_coeff_bias_LN_fp16_ref(torch::Tensor W1,
                                                            torch::Tensor W2,
                                                            torch::Tensor b1,
                                                            torch::Tensor b2,
                                                            torch::Tensor ln_weight,
                                                            torch::Tensor ln_bias,
                                                            torch::Tensor cumsum_matrix,
                                                            torch::Tensor make_last_b_matrix,
                                                            torch::Tensor make_last_coeff_1_matrix,
                                                            torch::Tensor make_last_coeff_2_matrix,
                                                            torch::Tensor XA,
                                                            torch::Tensor XB,
                                                            torch::Tensor XC,
                                                            torch::Tensor Coeff,
                                                            torch::Tensor Output)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    prefill_whole_loop_gelu_coeff_bias_LN_fp16(W1,
                                               W2,
                                               b1,
                                               b2,
                                               ln_weight,
                                               ln_bias,
                                               cumsum_matrix,
                                               make_last_b_matrix,
                                               make_last_coeff_1_matrix,
                                               make_last_coeff_2_matrix,
                                               XA,
                                               XB,
                                               XC,
                                               Coeff,
                                               Output,
                                               stream);
}


extern void  prefill_whole_loop_gelu_coeff_bias_LN_res_PLN_fp16(torch::Tensor W1,
                                                                torch::Tensor W2,
                                                                torch::Tensor b1,
                                                                torch::Tensor b2,
                                                                torch::Tensor ln_weight,
                                                                torch::Tensor ln_bias,
                                                                torch::Tensor cumsum_matrix,
                                                                torch::Tensor make_last_b_matrix,
                                                                torch::Tensor make_last_coeff_1_matrix,
                                                                torch::Tensor make_last_coeff_2_matrix,
                                                                torch::Tensor XA,
                                                                torch::Tensor XB,
                                                                torch::Tensor XC,
                                                                torch::Tensor Coeff,
                                                                torch::Tensor Output,
                                                                cudaStream_t stream);

extern void  prefill_whole_loop_gelu_coeff_bias_LN_res_PLN_fp16_ref(torch::Tensor W1,
                                                                    torch::Tensor W2,
                                                                    torch::Tensor b1,
                                                                    torch::Tensor b2,
                                                                    torch::Tensor ln_weight,
                                                                    torch::Tensor ln_bias,
                                                                    torch::Tensor cumsum_matrix,
                                                                    torch::Tensor make_last_b_matrix,
                                                                    torch::Tensor make_last_coeff_1_matrix,
                                                                    torch::Tensor make_last_coeff_2_matrix,
                                                                    torch::Tensor XA,
                                                                    torch::Tensor XB,
                                                                    torch::Tensor XC,
                                                                    torch::Tensor Coeff,
                                                                    torch::Tensor Output)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    prefill_whole_loop_gelu_coeff_bias_LN_res_PLN_fp16(W1,
                                                       W2,
                                                       b1,
                                                       b2,
                                                       ln_weight,
                                                       ln_bias,
                                                       cumsum_matrix,
                                                       make_last_b_matrix,
                                                       make_last_coeff_1_matrix,
                                                       make_last_coeff_2_matrix,
                                                       XA,
                                                       XB,
                                                       XC,
                                                       Coeff,
                                                       Output,
                                                       stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for warp test"; // optional module docstring
    m.def("prefill_whole_loop", &prefill_whole_loop_ref);
    m.def("prefill_whole_loop_fp16", prefill_whole_loop_ref_fp16);
    m.def("prefill_whole_loop_gelu", &prefill_whole_loop_gelu_ref);
    m.def("prefill_whole_loop_gelu_coeff_bias_fp16", prefill_whole_loop_gelu_coeff_bias_fp16_ref);
    m.def("prefill_whole_loop_gelu_coeff_bias_LN_fp16", prefill_whole_loop_gelu_coeff_bias_LN_fp16_ref);
    m.def("prefill_whole_loop_gelu_coeff_bias_LN_res_PLN_fp16", prefill_whole_loop_gelu_coeff_bias_LN_res_PLN_fp16_ref);
}