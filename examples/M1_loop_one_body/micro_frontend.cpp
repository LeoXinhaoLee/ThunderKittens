#include <torch/extension.h>
#include <vector>


extern void  micro(torch::Tensor W1,
                   torch::Tensor XA, torch::Tensor XB, torch::Tensor XC,
                   torch::Tensor Out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for warp test"; // optional module docstring
    m.def("micro", micro);
}