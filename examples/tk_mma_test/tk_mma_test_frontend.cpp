#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>


extern void  matmul_add_bf16(torch::Tensor W1, torch::Tensor b1,
                             torch::Tensor X, torch::Tensor Out, cudaStream_t stream);

extern void  matmul_add_bf16_ref(torch::Tensor W1, torch::Tensor b1,
                                 torch::Tensor X, torch::Tensor Out)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    matmul_add_bf16(W1, b1, X, Out, stream);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for warp test";
    m.def("matmul_add_bf16", &matmul_add_bf16_ref);
}
