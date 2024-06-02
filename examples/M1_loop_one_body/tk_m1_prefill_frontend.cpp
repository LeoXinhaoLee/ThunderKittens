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



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for warp test"; // optional module docstring
    m.def("prefill_loop_body", &prefill_loop_body_ref);
    m.def("prefill_whole_loop", &prefill_whole_loop_ref);
}