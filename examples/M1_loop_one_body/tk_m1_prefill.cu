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
#define SMEM_POOL 1
#define SMEM_BLOCK 3 * SMEM_POOL * X_STRIDE * 2  // bytes

using namespace kittens;


template <typename H, typename T>
__global__
void prefill_loop_body_ker(
        int CS, int HF,
        T* __W1,
        const T* __XA, const T* __XB, const T* __XC,
        T* __Output
) {
    H *_W1       = reinterpret_cast<H*>(__W1) + blockIdx.x*(HF*HF);
    const H *_XA       = reinterpret_cast<const H*>(__XA) + blockIdx.x*(CS*HF);
    const H *_XB       = reinterpret_cast<const H*>(__XB) + blockIdx.x*(CS*HF);
    const H *_XC       = reinterpret_cast<const H*>(__XC) + blockIdx.x*(CS*HF);
    H *_Output = reinterpret_cast<H*>(__Output) + blockIdx.x*(CS*HF);

    /*********
    REGISTER
    **********/
    rt_bf<4, 4, kittens::ducks::rt_layout::col> W1_reg;
    rt_bf<1, 4> XA_reg;
    rt_bf<1, 4> XB_reg;
    rt_bf<1, 4> XC_reg;

    rt_fl<1, 4> Z1_fl_reg;
    rt_bf<1, 4> Z1_reg;

    rt_bf<1, 4> Output_reg;
    rt_fl<1, 4> Z1_bar_term_1_fl_reg;
    rt_bf<1, 4> Z1_bar_term_1_reg;
    rt_fl<1, 4> Z1_bar_term_2_fl_reg;
    rt_bf<1, 4> Z1_bar_term_2_reg;
    rt_fl<1, 1> Attn1_fl_reg;
    rt_bf<1, 1> Attn1_reg;


    load(W1_reg, _W1, W1_reg.cols);
    load(XB_reg, _XB, XB_reg.cols);
    load(XA_reg, _XA, XA_reg.cols);
    load(XC_reg, _XC, XC_reg.cols);

    zero(Z1_fl_reg);
    mma_AB(Z1_fl_reg, XB_reg, W1_reg, Z1_fl_reg); // [K,f] r, [f,f] c -> [K,f] r

    copy(Z1_reg, Z1_fl_reg);
    sub(Z1_reg, Z1_reg, XA_reg);

    rt_bf<1, 4, ducks::rt_layout::col> &Z1_col_reg = swap_layout_inplace(Z1_reg); // row-maj -> col-maj

    zero(Attn1_fl_reg);
    mma_ABt(Attn1_fl_reg, XC_reg, XB_reg, Attn1_fl_reg);  // [N,K] r, [M,K] r -> [N,M] r
    copy(Attn1_reg, Attn1_fl_reg);
    make_causal(Attn1_reg, Attn1_reg, base_types::constants<bf16>::zero());

    zero(Z1_bar_term_1_fl_reg);
    mma_AB(Z1_bar_term_1_fl_reg, XC_reg, W1_reg, Z1_bar_term_1_fl_reg); // [N,K] r, [K,M] c -> [N,M] r
    copy(Z1_bar_term_1_reg, Z1_bar_term_1_fl_reg);

    zero(Z1_bar_term_2_fl_reg);
    mma_AB(Z1_bar_term_2_fl_reg, Attn1_reg, Z1_col_reg, Z1_bar_term_2_fl_reg);  // [K,K] r, [K,f] c -> [K,f] r
    copy(Z1_bar_term_2_reg, Z1_bar_term_2_fl_reg);

    sub(Output_reg, Z1_bar_term_1_reg, Z1_bar_term_2_reg);

    store(_Output, Output_reg, Output_reg.cols);

    rt_bf<1, 4, kittens::ducks::rt_layout::col> &XB_col_reg = swap_layout_inplace(XB_reg);

    rt_fl<4, 4> W1_fl_reg;
    zero(W1_fl_reg);
    mma_AtB(W1_fl_reg, XB_col_reg, Z1_col_reg, W1_fl_reg);

    rt_bf<4, 4> W1_row_reg;
    copy(W1_row_reg, W1_fl_reg);
    rt_bf<4, 4, kittens::ducks::rt_layout::col> &W1_col_reg = swap_layout_inplace(W1_row_reg);

    sub(W1_reg, W1_reg, W1_col_reg);

    store(_W1, W1_reg, W1_reg.cols);

}

void
prefill_loop_body
        (
                torch::Tensor W1,
                torch::Tensor XA,
                torch::Tensor XB,
                torch::Tensor XC,
                torch::Tensor Output,
                cudaStream_t stream
        ) {

    auto batch_mul_head = XA.size(0);
    auto cs    = XA.size(1);
    auto hf    = XA.size(2);

    using H = __nv_bfloat16;
    using T = c10::BFloat16;
    const int workers = 1;
    auto threads = workers * kittens::WARP_THREADS;

    prefill_loop_body_ker<H,T><<<batch_mul_head, threads, 0, stream>>>(
            cs, hf,
            W1.data_ptr<T>(),
            XA.data_ptr<T>(), XB.data_ptr<T>(), XC.data_ptr<T>(),
            Output.data_ptr<T>()
    );

    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


template <typename H, typename T>
__global__
void prefill_whole_loop_ker(
        const int NC, const int CS, const int HF,
        T* __W1,
        const T* __XA, const T* __XB, const T* __XC,
        T* __Output
) {
    H *_W1       = reinterpret_cast<H*>(__W1) + blockIdx.x * (HF*HF);
    const H *_XA       = reinterpret_cast<const H*>(__XA) + blockIdx.x * (NC*CS*HF);
    const H *_XB       = reinterpret_cast<const H*>(__XB) + blockIdx.x * (NC*CS*HF);
    const H *_XC       = reinterpret_cast<const H*>(__XC) + blockIdx.x * (NC*CS*HF);
    H *_Output = reinterpret_cast<H*>(__Output) + blockIdx.x * (NC*CS*HF);

    rt_bf<4, 4, kittens::ducks::rt_layout::col> W1_reg;
    rt_bf<1, 4> XA_reg;
    rt_bf<1, 4> XB_reg;
    rt_bf<1, 4> XC_reg;

    rt_fl<1, 4> Z1_fl_reg;
    rt_bf<1, 4> Z1_reg;

    rt_bf<1, 4> Output_reg;
    rt_fl<1, 4> Z1_bar_term_1_fl_reg;
    rt_bf<1, 4> Z1_bar_term_1_reg;
    rt_fl<1, 4> Z1_bar_term_2_fl_reg;
    rt_bf<1, 4> Z1_bar_term_2_reg;
    rt_fl<1, 1> Attn1_fl_reg;
    rt_bf<1, 1> Attn1_reg;
    rt_fl<4, 4> W1_fl_reg;
    rt_bf<4, 4> W1_row_reg;

    load(W1_reg, _W1, W1_reg.cols);

    for (int i = 0; i < NC; i++) {

        // rt_bf<1, 4> XB_reg;
        load(XB_reg, _XB + i * CS * HF, XB_reg.cols);
        // rt_fl<1, 4> Z1_fl_reg;
        zero(Z1_fl_reg);
        mma_AB(Z1_fl_reg, XB_reg, W1_reg, Z1_fl_reg); // [K,f] r, [f,f] c -> [K,f] r

        // rt_bf<1, 4> XA_reg;
        load(XA_reg, _XA + i * CS * HF, XA_reg.cols);

        // rt_bf<1, 4> Z1_reg;
        copy(Z1_reg, Z1_fl_reg);
        sub(Z1_reg, Z1_reg, XA_reg);

        rt_bf<1, 4, ducks::rt_layout::col> &Z1_col_reg = swap_layout_inplace(Z1_reg);

        // rt_bf<1, 4> XC_reg;
        load(XC_reg, _XC + i * CS * HF, XC_reg.cols);
        // rt_fl<1, 1> Attn1_fl_reg;
        zero(Attn1_fl_reg);
        mma_ABt(Attn1_fl_reg, XC_reg, XB_reg, Attn1_fl_reg);

        // rt_bf<1, 1> Attn1_reg;
        copy(Attn1_reg, Attn1_fl_reg);
        make_causal(Attn1_reg, Attn1_reg, base_types::constants<bf16>::zero());

        // rt_fl<1, 4> Z1_bar_term_1_fl_reg;
        zero(Z1_bar_term_1_fl_reg);
        mma_AB(Z1_bar_term_1_fl_reg, XC_reg, W1_reg, Z1_bar_term_1_fl_reg); // [N,K] r, [K,M] c -> [N,M] r
        // rt_bf<1, 4> Z1_bar_term_1_reg;
        copy(Z1_bar_term_1_reg, Z1_bar_term_1_fl_reg);

        // rt_fl<1, 4> Z1_bar_term_2_fl_reg;
        zero(Z1_bar_term_2_fl_reg);
        mma_AB(Z1_bar_term_2_fl_reg, Attn1_reg, Z1_col_reg, Z1_bar_term_2_fl_reg);  // [K,K] r, [K,f] c -> [K,f] r
        // rt_bf<1, 4> Z1_bar_term_2_reg;
        copy(Z1_bar_term_2_reg, Z1_bar_term_2_fl_reg);

        // rt_bf<1, 4> Output_reg;
        sub(Output_reg, Z1_bar_term_1_reg, Z1_bar_term_2_reg);
        store(_Output + i * CS * HF, Output_reg, Output_reg.cols);
        rt_bf<1, 4, kittens::ducks::rt_layout::col> &XB_col_reg = swap_layout_inplace(XB_reg);

        // rt_fl<4, 4> W1_fl_reg;
        zero(W1_fl_reg);
        mma_AtB(W1_fl_reg, XB_col_reg, Z1_col_reg, W1_fl_reg);

        // rt_bf<4, 4> W1_row_reg;
        copy(W1_row_reg, W1_fl_reg);
        rt_bf<4, 4, kittens::ducks::rt_layout::col> &W1_col_reg = swap_layout_inplace(W1_row_reg);

        sub(W1_reg, W1_reg, W1_col_reg);
    }

    store(_W1, W1_reg, W1_reg.cols);
}


void
prefill_whole_loop
        (
                torch::Tensor W1,
                torch::Tensor XA,
                torch::Tensor XB,
                torch::Tensor XC,
                torch::Tensor Output,
                cudaStream_t stream
        ) {
    auto batch = XA.size(0);
    auto head = XA.size(1);
    auto NC = XA.size(2);
    auto CS = XA.size(3);
    auto HF = XA.size(4);

    using H = __nv_bfloat16;
    using T = c10::BFloat16;
    const int workers = 1;

    auto threads = workers * kittens::WARP_THREADS;

//    std::cout << "M1 TK whole loop" << std::endl;

    prefill_whole_loop_ker<H, T><<<batch * head, threads, 0, stream>>>(
            NC, CS, HF,
            W1.data_ptr<T>(),
            XA.data_ptr<T>(), XB.data_ptr<T>(), XC.data_ptr<T>(),
            Output.data_ptr<T>()
    );


}


template <typename H, typename T>
__global__
void prefill_whole_loop_LN_bias_ker(
        const int NH, const int NC, const int CS, const int HF,
        T* __W1, T* __b1,
        const T* __ln_weight, const T* __ln_bias,
        const T* __XA, const T* __XB, const T* __XC,
        T* __Output
) {
    H *_W1       = reinterpret_cast<H*>(__W1) + blockIdx.x * (HF*HF);
    H *_b1       = reinterpret_cast<H*>(__b1) + blockIdx.x * HF;

    const H *_ln_weight = reinterpret_cast<const H*>(__ln_weight) + (blockIdx.x % NH) * (CS*HF);
    const H *_ln_bias   = reinterpret_cast<const H*>(__ln_bias) + (blockIdx.x % NH) * (CS*HF);

    const H *_XA        = reinterpret_cast<const H*>(__XA) + blockIdx.x * (NC*CS*HF);
    const H *_XB        = reinterpret_cast<const H*>(__XB) + blockIdx.x * (NC*CS*HF);
    const H *_XC        = reinterpret_cast<const H*>(__XC) + blockIdx.x * (NC*CS*HF);
    H *_Output = reinterpret_cast<H*>(__Output) + blockIdx.x * (NC*CS*HF);

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);

    st_bf<1, 4, ducks::st_layout::swizzle> (&XA_smem)[SMEM_POOL] = al.allocate<st_bf<1, 4, ducks::st_layout::swizzle>, SMEM_POOL>();
    st_bf<1, 4, ducks::st_layout::swizzle> (&XB_smem)[SMEM_POOL] = al.allocate<st_bf<1, 4, ducks::st_layout::swizzle>, SMEM_POOL>();
    st_bf<1, 4, ducks::st_layout::swizzle> (&XC_smem)[SMEM_POOL] = al.allocate<st_bf<1, 4, ducks::st_layout::swizzle>, SMEM_POOL>();

    rt_bf<4, 4, kittens::ducks::rt_layout::col> W1_reg;
    load(W1_reg, _W1, W1_reg.cols);

    rt_bf<1, 4> ln_w_reg;
    rt_bf<1, 4> ln_b_reg;
    load(ln_w_reg, _ln_weight, ln_w_reg.cols);
    load(ln_b_reg, _ln_bias, ln_b_reg.cols);

    for (int i = 0; i < NC; i++) {

        if (i % SMEM_POOL == 0) {
            for (int j = 0; j < SMEM_POOL; j++) {
                load(XA_smem[j], _XA + (i + j) * X_STRIDE, 64);
                load(XB_smem[j], _XB + (i + j) * X_STRIDE, 64);
                load(XC_smem[j], _XC + (i + j) * X_STRIDE, 64);
            }
        }

        // Forward
        rt_bf<1, 4> XB_reg;
        load(XB_reg, XB_smem[i % SMEM_POOL]);

        rt_fl<1, 4> Z1_fl_reg;
        zero(Z1_fl_reg);
        mma_AB(Z1_fl_reg, XB_reg, W1_reg, Z1_fl_reg); // [K,f]r, [f,f]c -> [K,f]r
        rt_bf<1, 4> Z1_reg;
        copy(Z1_reg, Z1_fl_reg);

        rt_bf<1, 4> XA_reg;
        load(XA_reg, XA_smem[i % SMEM_POOL]);
        sub(XA_reg, XA_reg, XB_reg);  // l2_tgt = XA - XB

        // LN fwd + bwd
        rt_bf<1, 4>::col_vec Z1_mean_reg;
        row_sum(Z1_mean_reg, Z1_reg);  // [K,f]
        div(Z1_mean_reg, Z1_mean_reg, __float2bfloat16(HF));

        rt_bf<1, 4> Z1_square_reg;
        sub_row(Z1_square_reg, Z1_reg, Z1_mean_reg);
        mul(Z1_square_reg, Z1_square_reg, Z1_square_reg); // (Z1 - mu) ** 2

        rt_bf<1, 4>::col_vec Z1_std_reg;
        row_sum(Z1_std_reg, Z1_square_reg);  // [K,f]
        div(Z1_std_reg, Z1_std_reg, __float2bfloat16(HF));
        add(Z1_std_reg, Z1_std_reg, __float2bfloat16(1e-6f));
        // TODO: sqrt to get std
        sqrt(Z1_std_reg, Z1_std_reg);

        rt_bf<1, 4> Z1_hat;  // normalized Z1 with 0 mean and 1 std
        sub_row(Z1_hat, Z1_reg, Z1_mean_reg);
        div_row(Z1_hat, Z1_hat, Z1_std_reg);
/*
        rt_bf<1, 4> LN_out_reg;  // affined by LN scale and bias
        mul(LN_out_reg, Z1_hat, ln_w_reg);  // [K,f] * [K,f]
        add(LN_out_reg, LN_out_reg, ln_b_reg);

        rt_bf<1, 4> dl_dZ1_hat;
        sub(dl_dZ1_hat, LN_out_reg, XA_reg);
        mul(dl_dZ1_hat, dl_dZ1_hat, ln_w_reg);

        rt_bf<1, 4> dl_dZ1;
        mul(dl_dZ1, dl_dZ1_hat, __float2bfloat16(HF));

        rt_bf<1, 4>::col_vec dl_dZ1_vec_term;
        row_sum(dl_dZ1_vec_term, dl_dZ1_hat);
        sub_row(dl_dZ1, dl_dZ1, dl_dZ1_vec_term);

        rt_bf<1, 4> dl_dZ1_term_3;
        mul(dl_dZ1_term_3, dl_dZ1_hat, Z1_hat);
        row_sum(dl_dZ1_vec_term, dl_dZ1_term_3);
        mul_row(dl_dZ1_term_3, Z1_hat, dl_dZ1_vec_term);

        sub(dl_dZ1, dl_dZ1, dl_dZ1_term_3);
        div_row(dl_dZ1, dl_dZ1, Z1_std_reg);
        div(dl_dZ1, dl_dZ1, __float2bfloat16(HF));

        rt_bf<1, 4, ducks::rt_layout::col> &dl_dZ1_col = swap_layout_inplace(dl_dZ1);  // [K,f]
*/

/* Test mem leak: no LN can match
        rt_bf<1, 4> dl_dZ1;
        copy(dl_dZ1, Z1_reg);
        rt_bf<1, 4, ducks::rt_layout::col> &dl_dZ1_col = swap_layout_inplace(dl_dZ1);
*/

        rt_bf<1, 4> dl_dZ1;
        copy(dl_dZ1, Z1_hat);
        rt_bf<1, 4, ducks::rt_layout::col> &dl_dZ1_col = swap_layout_inplace(dl_dZ1);

        // 2nd forward
        rt_bf<1, 4> XC_reg;
        load(XC_reg, XC_smem[i % SMEM_POOL]);

        rt_fl<1, 1> Attn1_fl_reg;
        zero(Attn1_fl_reg);
        mma_ABt(Attn1_fl_reg, XC_reg, XB_reg, Attn1_fl_reg);

        rt_bf<1, 1> Attn1_reg;
        copy(Attn1_reg, Attn1_fl_reg);
        make_causal(Attn1_reg, Attn1_reg, base_types::constants<bf16>::zero());

        rt_fl<1, 4> Z1_bar_term_1_fl_reg;
        zero(Z1_bar_term_1_fl_reg);
        mma_AB(Z1_bar_term_1_fl_reg, XC_reg, W1_reg, Z1_bar_term_1_fl_reg); // [N,K] r, [K,M] c -> [N,M] r
        rt_bf<1, 4> Z1_bar_term_1_reg;
        copy(Z1_bar_term_1_reg, Z1_bar_term_1_fl_reg);

        rt_fl<1, 4> Z1_bar_term_2_fl_reg;
        zero(Z1_bar_term_2_fl_reg);
        mma_AB(Z1_bar_term_2_fl_reg, Attn1_reg, dl_dZ1_col, Z1_bar_term_2_fl_reg);  // [K,K] r, [K,f] c -> [K,f] r
        rt_bf<1, 4> Z1_bar_term_2_reg;
        copy(Z1_bar_term_2_reg, Z1_bar_term_2_fl_reg);

        sub(Z1_bar_term_1_reg, Z1_bar_term_1_reg, Z1_bar_term_2_reg);
        add(Z1_bar_term_1_reg, XC_reg, Z1_bar_term_1_reg);
        store(_Output + i * CS * HF, Z1_bar_term_1_reg, Z1_bar_term_1_reg.cols);

        rt_bf<1, 4, kittens::ducks::rt_layout::col> &XB_col_reg = swap_layout_inplace(XB_reg);
        rt_fl<4, 4> delta_W1_fl_reg;
        zero(delta_W1_fl_reg);
        mma_AtB(delta_W1_fl_reg, XB_col_reg, dl_dZ1_col, delta_W1_fl_reg);

        rt_bf<4, 4> delta_W1_reg;
        copy(delta_W1_reg, delta_W1_fl_reg);
        rt_bf<4, 4, kittens::ducks::rt_layout::col> &delta_W1_col_reg = swap_layout_inplace(delta_W1_reg);

        sub(W1_reg, W1_reg, delta_W1_col_reg);
    }

    store(_W1, W1_reg, W1_reg.cols);
}


void
prefill_whole_loop_LN_bias
        (
                torch::Tensor W1,
                torch::Tensor b1,
                torch::Tensor ln_weight,
                torch::Tensor ln_bias,
                torch::Tensor XA,
                torch::Tensor XB,
                torch::Tensor XC,
                torch::Tensor Output,
                cudaStream_t stream
        ) {
    auto batch = XA.size(0);
    auto head = XA.size(1);
    auto NC = XA.size(2);
    auto CS = XA.size(3);
    auto HF = XA.size(4);

    using H = __nv_bfloat16;
    using T = c10::BFloat16;
    const int workers = 1;

    auto threads = workers * kittens::WARP_THREADS;

    prefill_whole_loop_LN_bias_ker<H, T><<<batch * head, threads, SMEM_BLOCK, stream>>>(
            head, NC, CS, HF,
            W1.data_ptr<T>(), b1.data_ptr<T>(),
            ln_weight.data_ptr<T>(), ln_bias.data_ptr<T>(),
            XA.data_ptr<T>(), XB.data_ptr<T>(), XC.data_ptr<T>(),
            Output.data_ptr<T>()
    );

}
