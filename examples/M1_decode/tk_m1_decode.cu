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
#define b_STRIDE 64    // 64
#define Coeff_STRIDE 256  // 16 * 16
#define SMEM_POOL 1
#define SMEM_BLOCK SMEM_POOL * (3 * X_STRIDE + Coeff_STRIDE) * 2  // bytes: XA/XB/XC/Coeff_1

using namespace kittens;

template <typename H, typename T>
__global__
void load_store_vector_ker(
        const int CS, const int HF,
        const T* __XA, T* __Output
) {
    const H *_XA = reinterpret_cast<const H *>(__XA) + blockIdx.x * (CS * HF);
    H *_Output = reinterpret_cast<H *>(__Output) + blockIdx.x * (CS * HF);

    rt_hf<1, 4>::row_vec XA_reg;
//    rt_hf<4, 1>::col_vec XA_reg;
    load(XA_reg, _XA);
    store(_Output, XA_reg);

}

void
load_store_vector_fp16(torch::Tensor XA, torch::Tensor Output, cudaStream_t stream) {

    auto batch = XA.size(0);
    auto head = XA.size(1);
    auto CS = XA.size(2);  // 1
    auto HF = XA.size(3);

    using H = __half;
    using T = c10::Half;
    const int workers = 1;

    auto threads = workers * kittens::WARP_THREADS;

    load_store_vector_ker<H, T><<<batch * head, threads, 0, stream>>>(
            CS, HF,
            XA.data_ptr<T>(), Output.data_ptr<T>()
    );
}

/*
template <typename H, typename T>
__global__
void outer_product_vector_fp16_ker(
        const int CS, const int HF,
        const T* __X, const T* __Y, T* __Output
) {
    const H *_X = reinterpret_cast<const H *>(__X) + blockIdx.x * (CS * HF);
    const H *_Y = reinterpret_cast<const H *>(__Y) + blockIdx.x * (HF * CS);
    H *_Output = reinterpret_cast<H *>(__Output) + blockIdx.x * (HF * HF);

    rt_hf<1, 4>::row_vec X_row_vec;
    rt_hf<4, 1>::col_vec Y_col_vec;
    load(X_row_vec, _X);
    load(Y_col_vec, _Y);
    rt_hf<4, 4> Output_reg;
    mul(Output_reg, Y_col_vec, X_row_vec);
    store(_Output, Output_reg);

}

void
outer_product_vector_fp16(torch::Tensor X, torch::Tensor Y,
                          torch::Tensor Output, cudaStream_t stream) {

    auto batch = X.size(0);
    auto head = X.size(1);
    auto CS = X.size(2);  // 1
    auto HF = X.size(3);

    using H = __half;
    using T = c10::Half;
    const int workers = 1;

    auto threads = workers * kittens::WARP_THREADS;

    outer_product_vector_fp16_ker<H, T><<<batch * head, threads, 0, stream>>>(
            CS, HF,
            X.data_ptr<T>(), Y.data_ptr<T>(), Output.data_ptr<T>()
    );
}
*/

template <typename H, typename T>
__global__
void decode_coeff_LN_bias_fp16_ker(
        const int NH, const int CS, const int HF,
        const T* __W1, const T* __b1,
        T* __W1_grad, T* __b1_grad,
        const T* __ln_weight, const T* __ln_bias,
        const T* __XA, const T* __XB, const T* __XC,
        const T* __token_idx, const T* __ilr_gated,
        T* __Output
) {

    H *_W1_grad       = reinterpret_cast<H*>(__W1_grad) + blockIdx.x * (HF*HF);
    H *_b1_grad       = reinterpret_cast<H*>(__b1_grad) + blockIdx.x * (CS*HF);                   // duplicated rows
    H *_Output        = reinterpret_cast<H*>(__Output) + blockIdx.x * (CS*HF);                    // duplicated rows

    const H *_W1             = reinterpret_cast<const H*>(__W1) + blockIdx.x * (HF*HF);
    const H *_b1             = reinterpret_cast<const H*>(__b1) + blockIdx.x * HF;                // row vec
    const H *_ln_weight      = reinterpret_cast<const H*>(__ln_weight) + (blockIdx.x % NH) * HF;  // row vec
    const H *_ln_bias        = reinterpret_cast<const H*>(__ln_bias) + (blockIdx.x % NH) * HF;    // row vec
    const H *_XA             = reinterpret_cast<const H*>(__XA) + blockIdx.x * HF;                // row vec
    const H *_XB             = reinterpret_cast<const H*>(__XB) + blockIdx.x * (CS*HF);           // duplicated rows
    const H *_XC             = reinterpret_cast<const H*>(__XC) + blockIdx.x * (CS*HF);           // duplicated rows
    const H *_token_idx      = reinterpret_cast<const H*>(__token_idx);
    const H *_ilr_gated      = reinterpret_cast<const H*>(__ilr_gated) + blockIdx.x * HF;

    rt_hf<4, 4, kittens::ducks::rt_layout::col> W1_reg;
    load(W1_reg, _W1, W1_reg.cols);

    rt_hf<1, 4>::row_vec b1_row_vec;
    load(b1_row_vec, _b1);

    rt_hf<1, 4>::row_vec ln_w_row_vec;
    rt_hf<1, 4>::row_vec ln_b_row_vec;
    load(ln_w_row_vec, _ln_weight);
    load(ln_b_row_vec, _ln_bias);

    rt_hf<1, 4> XB_reg;
    load(XB_reg, _XB, XB_reg.cols);
    rt_hf<1, 4> XC_reg;
    load(XC_reg, _XC, XC_reg.cols);

    // Z1 = XB @ W1 + b1
    rt_hf<1, 4> Z1_reg;
    zero(Z1_reg);
    mma_AB(Z1_reg, XB_reg, W1_reg, Z1_reg); // [K,f]r <- [K,f]r @ [f,f]c
    add_col(Z1_reg, Z1_reg, b1_row_vec);    // [K,f] <- [K,f] + [1,f]

    // mu = Z1.sum(dim=-1) / HF
    rt_hf<1, 4>::col_vec Z1_mean_reg;
    row_sum(Z1_mean_reg, Z1_reg);  // [K,f]
    div(Z1_mean_reg, Z1_mean_reg, __float2half(float(HF)));

    // var = ((Z1 - mu) * (Z1 - mu)).sum(dim=-1) / HF
    rt_hf<1, 4> Z1_square_reg;
    sub_row(Z1_square_reg, Z1_reg, Z1_mean_reg);
    mul(Z1_square_reg, Z1_square_reg, Z1_square_reg); // (Z1 - mu) ** 2
    rt_hf<1, 4>::col_vec Z1_std_reg;
    row_sum(Z1_std_reg, Z1_square_reg);  // [K,f]
    div(Z1_std_reg, Z1_std_reg, __float2half(float(HF)));
    // std = sqrt(var + eps)
    add(Z1_std_reg, Z1_std_reg, __float2half(1e-6f));
    sqrt(Z1_std_reg, Z1_std_reg);

    // Z1_hat = (Z1 - mu) / std
    rt_hf<1, 4> Z1_hat;  // normalized Z1 with 0 mean and 1 std
    sub_row(Z1_hat, Z1_reg, Z1_mean_reg);
    div_row(Z1_hat, Z1_hat, Z1_std_reg);
    // LN_out = ln_w * Z1_hat + ln_b
    rt_hf<1, 4> LN_out_reg;  // affined by LN scale and bias
    mul_col(LN_out_reg, Z1_hat, ln_w_row_vec);        // [K,f] <- [K,f] * [1,f]
    add_col(LN_out_reg, LN_out_reg, ln_b_row_vec);    // [K,f] <- [K,f] + [1,f]

    // l2_tgt = XA - XB
    rt_hf<1, 4>::row_vec XA_row_vec;
    load(XA_row_vec, _XA);
    rt_hf<1, 4> l2_target_reg;
    zero(l2_target_reg);
    add_col(l2_target_reg, l2_target_reg, XA_row_vec);  // [K,f] <- [K,f] + [1,f]
    sub(l2_target_reg, l2_target_reg, XB_reg);

    // dl_dLN_out = LN_out - l2_tgt
    rt_hf<1, 4> dl_dZ1_hat;
    sub(dl_dZ1_hat, LN_out_reg, l2_target_reg);
    // dl_dZ1_hat = dl_dLN_out * ln_weight
    mul_col(dl_dZ1_hat, dl_dZ1_hat, ln_w_row_vec);

    // dl_dZ1_term_1 = HF * dl_dZ1_hat
    rt_hf<1, 4> dl_dZ1;
    mul(dl_dZ1, dl_dZ1_hat, __float2half(float(HF)));  // HF * dl_dZ1_hat

    // dl_dZ1_term_2 = dl_dZ1_hat.sum(dim=-1, keepdim=True)
    rt_hf<1, 4>::col_vec dl_dZ1_vec_term;
    row_sum(dl_dZ1_vec_term, dl_dZ1_hat);

    // dl_dZ1_term_3 = Z1_hat * (dl_dZ1_hat * Z1_hat).sum(dim=-1, keepdim=True)
    rt_hf<1, 4> dl_dZ1_term_3;
    mul(dl_dZ1_term_3, dl_dZ1_hat, Z1_hat);
    row_sum(dl_dZ1_vec_term, dl_dZ1_term_3);
    mul_row(dl_dZ1_term_3, Z1_hat, dl_dZ1_vec_term);

    // dl_dZ1 = (dl_dZ1_term_1 - dl_dZ1_term_2 - dl_dZ1_term_3) / (std * HF)
    sub_row(dl_dZ1, dl_dZ1, dl_dZ1_vec_term);
    sub(dl_dZ1, dl_dZ1, dl_dZ1_term_3);
    mul(Z1_std_reg, Z1_std_reg, __float2half(float(HF)));
    div_row(dl_dZ1, dl_dZ1, Z1_std_reg);

    // ilr_mul_dl_dZ1 = ilr * dl_dZ1
    rt_hf<1, 4>::row_vec ilr_gated_row_vec;
    load(ilr_gated_row_vec, _ilr_gated);
    mul_col(dl_dZ1, dl_dZ1, ilr_gated_row_vec);  // [K,f] <- [K,f] * [1,f]

    // delta_b1 = ilr_mul_dl_dZ1
    // b1_grad = b1_grad + delta_b1
    rt_hf<1, 4> b1_grad_reg;  // @xinhao: b1_grad must be tile because dl_dZ1 is tile, which will be stored back
    load(b1_grad_reg, _b1_grad, b1_grad_reg.cols);
    add(b1_grad_reg, b1_grad_reg, dl_dZ1);
    store(_b1_grad, b1_grad_reg, b1_grad_reg.cols);

    // delta_W1 = XB.T @ ilr_mul_dl_dZ1 / K, where K is duplication 16
    rt_hf<4, 4> delta_W1_reg;
    zero(delta_W1_reg);
    rt_hf<1, 4, kittens::ducks::rt_layout::col> &XB_col_reg = swap_layout_inplace(XB_reg);
    rt_hf<1, 4, kittens::ducks::rt_layout::col> &dl_dZ1_col = swap_layout_inplace(dl_dZ1);
    mma_AtB(delta_W1_reg, XB_col_reg, dl_dZ1_col, delta_W1_reg); // [f,f']r <- [K,f]c.T @ [K,f']c
    div(delta_W1_reg, delta_W1_reg, __float2half(float(CS)));    // deduplicate
    // W1_grad = W1_grad + delta_W1
    rt_hf<4, 4> W1_grad_reg;
    load(W1_grad_reg, _W1_grad, W1_grad_reg.cols);
    add(W1_grad_reg, W1_grad_reg, delta_W1_reg);
    store(_W1_grad, W1_grad_reg, W1_grad_reg.cols);

    // W1_bar = W1 - token_idx * W1_grad
    rt_hf<1, 4>::row_vec token_idx_row_vec;
    load(token_idx_row_vec, _token_idx);
    mul_col(W1_grad_reg, W1_grad_reg, token_idx_row_vec);  // [f,f] * [1,f]
    rt_hf<4, 4, kittens::ducks::rt_layout::col> &W1_grad_col_reg = swap_layout_inplace(W1_grad_reg);
    sub(W1_reg, W1_reg, W1_grad_col_reg);

    // b1_bar = b1 - token_idx * b1_grad
    mul_col(b1_grad_reg, b1_grad_reg, token_idx_row_vec); // [K,f] * [1,f]
    sub_col(b1_grad_reg, b1_grad_reg, b1_row_vec);
    mul(b1_grad_reg, b1_grad_reg, __float2half(-1.0f));

    // Z1_bar = XC @ W1_bar + b1_bar
    mma_AB(Z1_reg, XC_reg, W1_reg, b1_grad_reg); // [K,f]r <- [K,f]r @ [f,f]c + [K,f]r

    store(_Output, Z1_reg, Z1_reg.cols);

}

void
decode_coeff_LN_bias_fp16(
        torch::Tensor W1, torch::Tensor b1,
        torch::Tensor W1_grad, torch::Tensor b1_grad,
        torch::Tensor ln_weight, torch::Tensor ln_bias,
        torch::Tensor XA, torch::Tensor XB, torch::Tensor XC,
        torch::Tensor token_idx, torch::Tensor ilr_gated,
        torch::Tensor Output, cudaStream_t stream
) {
    auto batch = XB.size(0);
    auto head = XC.size(1);
    auto CS = XB.size(2);  // CS=16 due to duplication
    auto HF = XB.size(3);

    using H = __half;
    using T = c10::Half;
    const int workers = 1;

    auto threads = workers * kittens::WARP_THREADS;

    decode_coeff_LN_bias_fp16_ker<H, T><<<batch * head, threads, 0, stream>>>(
            head, CS, HF,
            W1.data_ptr<T>(), b1.data_ptr<T>(),
            W1_grad.data_ptr<T>(), b1_grad.data_ptr<T>(),
            ln_weight.data_ptr<T>(), ln_bias.data_ptr<T>(),
            XA.data_ptr<T>(), XB.data_ptr<T>(), XC.data_ptr<T>(),
            token_idx.data_ptr<T>(), ilr_gated.data_ptr<T>(),
            Output.data_ptr<T>()
    );

//    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

}


template <typename H, typename T>
__global__
void decode_coeff_LN_bias_simplified_fp16_ker(
        const int NH, const int CS, const int HF,
        const T* __W1, const T* __b1,
        T* __W1_grad, T* __b1_grad,
        const T* __ln_weight, const T* __ln_bias,
        const T* __XA, const T* __XB, const T* __XC,
        const T* __token_idx, const T* __ilr_gated,
        T* __Output
) {

    H *_W1_grad       = reinterpret_cast<H*>(__W1_grad) + blockIdx.x * (HF*HF);
    H *_b1_grad       = reinterpret_cast<H*>(__b1_grad) + blockIdx.x * HF;
    H *_Output        = reinterpret_cast<H*>(__Output) + blockIdx.x * HF;

    const H *_W1             = reinterpret_cast<const H*>(__W1) + blockIdx.x * (HF*HF);
    const H *_b1             = reinterpret_cast<const H*>(__b1) + blockIdx.x * HF;                // row vec
    const H *_ln_weight      = reinterpret_cast<const H*>(__ln_weight) + (blockIdx.x % NH) * HF;  // row vec
    const H *_ln_bias        = reinterpret_cast<const H*>(__ln_bias) + (blockIdx.x % NH) * HF;    // row vec
    const H *_XA             = reinterpret_cast<const H*>(__XA) + blockIdx.x * HF;                // row vec
    const H *_XB             = reinterpret_cast<const H*>(__XB) + blockIdx.x * HF;                // row vec
    const H *_XC             = reinterpret_cast<const H*>(__XC) + blockIdx.x * HF;                // row vec
    const H *_token_idx      = reinterpret_cast<const H*>(__token_idx);
    const H *_ilr_gated      = reinterpret_cast<const H*>(__ilr_gated) + blockIdx.x * HF;

    rt_hf<4, 4, kittens::ducks::rt_layout::col> W1_reg;
    load(W1_reg, _W1, W1_reg.cols);

    rt_hf<1, 4>::row_vec b1_row_vec;
    load(b1_row_vec, _b1);

    rt_hf<1, 4>::row_vec ln_w_row_vec;
    rt_hf<1, 4>::row_vec ln_b_row_vec;
    load(ln_w_row_vec, _ln_weight);
    load(ln_b_row_vec, _ln_bias);

    rt_hf<1, 4>::row_vec XB_row_vec;
    load(XB_row_vec, _XB);
    rt_hf<1, 4>::row_vec XC_row_vec;
    load(XC_row_vec, _XC);
    rt_hf<1, 4>::row_vec XA_row_vec;
    load(XA_row_vec, _XA);

    rt_hf<1, 4>::row_vec ilr_gated_row_vec;
    load(ilr_gated_row_vec, _ilr_gated);

    // delta_b1 = ilr_mul_dl_dZ1
    // b1_grad = b1_grad + delta_b1
    rt_hf<1, 4>::row_vec b1_grad_row_vec;
    load(b1_grad_row_vec, _b1_grad);
    store(_b1_grad, b1_grad_row_vec);

    // W1_grad = W1_grad + delta_W1
    rt_hf<4, 4> W1_grad_reg;
    load(W1_grad_reg, _W1_grad, W1_grad_reg.cols);
    store(_W1_grad, W1_grad_reg, W1_grad_reg.cols);

    // W1_bar = W1 - token_idx * W1_grad
    rt_hf<1, 4>::row_vec token_idx_row_vec;
    load(token_idx_row_vec, _token_idx);

    store(_Output, XC_row_vec);

}

void
decode_coeff_LN_bias_simplified_fp16(
        torch::Tensor W1, torch::Tensor b1,
        torch::Tensor W1_grad, torch::Tensor b1_grad,
        torch::Tensor ln_weight, torch::Tensor ln_bias,
        torch::Tensor XA, torch::Tensor XB, torch::Tensor XC,
        torch::Tensor token_idx, torch::Tensor ilr_gated,
        torch::Tensor Output, cudaStream_t stream
) {
    auto batch = XB.size(0);
    auto head = XC.size(1);
    auto CS = XB.size(2);  // CS=16 due to duplication
    auto HF = XB.size(3);

    using H = __half;
    using T = c10::Half;
    const int workers = 1;

    auto threads = workers * kittens::WARP_THREADS;

    decode_coeff_LN_bias_simplified_fp16_ker<H, T><<<batch * head, threads, 0, stream>>>(
            head, CS, HF,
            W1.data_ptr<T>(), b1.data_ptr<T>(),
            W1_grad.data_ptr<T>(), b1_grad.data_ptr<T>(),
            ln_weight.data_ptr<T>(), ln_bias.data_ptr<T>(),
            XA.data_ptr<T>(), XB.data_ptr<T>(), XC.data_ptr<T>(),
            token_idx.data_ptr<T>(), ilr_gated.data_ptr<T>(),
            Output.data_ptr<T>()
    );

//    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

}

