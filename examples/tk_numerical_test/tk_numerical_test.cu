#include <iostream>
#include <string>
#include <math.h>
#include <assert.h>
//#include <mma_AB.h>
#include <string>
#include <cuda_runtime_api.h>
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

# include "../../src/kittens.cuh"
# include "../../src/common/pyutils/torch_helpers.cuh"

// **** ASYNC INCLUDE *****
#include <cuda/pipeline>
#include <cooperative_groups.h>

#define X_STRIDE 1024  // 16 * 64
#define W_STRIDE 4096  // 64 * 64
#define b_STRIDE 1024  // 16 * 64
#define Coeff_STRIDE 256  // 16 * 16
#define SMEM_POOL 1
#define SMEM_BLOCK SMEM_POOL * (3 * X_STRIDE + Coeff_STRIDE) * 2  // bytes: XA/XB/XC/Coeff_1

using namespace kittens;


template <typename H, typename T>
__global__
void matmul_add_bf16_ker(
        const int NH, const int CS, const int HF,
        T* __W1, T* __b1,
        const T* __X, T* __Output
) {
    H *_W1       = reinterpret_cast<H*>(__W1) + blockIdx.x * (HF*HF);
    H *_b1       = reinterpret_cast<H*>(__b1) + blockIdx.x * (CS*HF);

    const H *_X             = reinterpret_cast<const H*>(__X) + blockIdx.x * (CS*HF);
    H *_Output              = reinterpret_cast<H*>(__Output) + blockIdx.x * (CS*HF);

    rt_bf<4, 4, kittens::ducks::rt_layout::col> W1_reg;
    load(W1_reg, _W1, W1_reg.cols);

    rt_bf<1, 4> b1_bf_reg;
    load(b1_bf_reg, _b1, b1_bf_reg.cols);
    rt_fl<1, 4> b1_reg;
    copy(b1_reg, b1_bf_reg);

    rt_bf<1, 4> X_reg;
    load(X_reg, _X, X_reg.cols);

    rt_fl<1, 4> Z1_fl_reg;
    mma_AB(Z1_fl_reg, X_reg, W1_reg, b1_reg);
    rt_bf<1, 4> Z1_reg;
    copy(Z1_reg, Z1_fl_reg);

    store(_Output, Z1_reg, Z1_reg.cols);  // @xinhao: XC + LN(Z1_bar) can be done outside

}

void
matmul_add_bf16
        (
                torch::Tensor W1,
                torch::Tensor b1,
                torch::Tensor X,
                torch::Tensor Output,
                cudaStream_t stream
        ) {
    auto batch = X.size(0);
    auto head = X.size(1);
    auto CS = X.size(2);
    auto HF = X.size(3);

    using H = __nv_bfloat16;
    using T = c10::BFloat16;
    const int workers = 1;

    auto threads = workers * kittens::WARP_THREADS;

    matmul_add_bf16_ker<H, T><<<batch * head, threads, SMEM_BLOCK, stream>>>(
            batch * head, CS, HF,
            W1.data_ptr<T>(), b1.data_ptr<T>(),
            X.data_ptr<T>(), Output.data_ptr<T>()
    );

}


template <typename H, typename T>
__global__
void matmul_add_fp16_ker(
        const int NH, const int CS, const int HF,
        T* __W1, T* __b1,
        const T* __X, T* __Output
) {
    H *_W1       = reinterpret_cast<H*>(__W1) + blockIdx.x * (HF*HF);
    H *_b1       = reinterpret_cast<H*>(__b1) + blockIdx.x * (CS*HF);

    const H *_X             = reinterpret_cast<const H*>(__X) + blockIdx.x * (CS*HF);
    H *_Output              = reinterpret_cast<H*>(__Output) + blockIdx.x * (CS*HF);

    rt_hf<4, 4, kittens::ducks::rt_layout::col> W1_reg;
    load(W1_reg, _W1, W1_reg.cols);

    rt_hf<1, 4> b1_reg;
    load(b1_reg, _b1, b1_reg.cols);

    rt_hf<1, 4> X_reg;
    load(X_reg, _X, X_reg.cols);

    rt_hf<1, 4> Z1_reg;
    mma_AB(Z1_reg, X_reg, W1_reg, b1_reg);

    store(_Output, Z1_reg, Z1_reg.cols);  // @xinhao: XC + LN(Z1_bar) can be done outside

}

void
matmul_add_fp16
        (
                torch::Tensor W1,
                torch::Tensor b1,
                torch::Tensor X,
                torch::Tensor Output,
                cudaStream_t stream
        ) {

    auto batch = X.size(0);
    auto head = X.size(1);
    auto CS = X.size(2);
    auto HF = X.size(3);

    using H = __half;
    using T = c10::Half;
    const int workers = 1;

    auto threads = workers * kittens::WARP_THREADS;

    matmul_add_fp16_ker<H, T><<<batch * head, threads, SMEM_BLOCK, stream>>>(
            batch * head, CS, HF,
            W1.data_ptr<T>(), b1.data_ptr<T>(),
            X.data_ptr<T>(), Output.data_ptr<T>()
    );

}
