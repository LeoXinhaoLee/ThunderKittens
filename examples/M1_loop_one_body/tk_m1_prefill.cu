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
        const T* __cumsum_matrix, const T* __make_last_b_matrix, const T* __make_last_coeff_1_matrix,
        const T* __XA, const T* __XB, const T* __XC, const T* __Coeff,
        T* __Output
) {
    H *_W1       = reinterpret_cast<H*>(__W1) + blockIdx.x * (HF*HF);
    H *_b1       = reinterpret_cast<H*>(__b1) + blockIdx.x * (CS*HF);

    const H *_ln_weight = reinterpret_cast<const H*>(__ln_weight) + (blockIdx.x % NH) * (CS*HF);
    const H *_ln_bias   = reinterpret_cast<const H*>(__ln_bias) + (blockIdx.x % NH) * (CS*HF);

    const H *_cumsum_matrix              = reinterpret_cast<const H*>(__cumsum_matrix);
    const H *_make_last_b_matrix         = reinterpret_cast<const H*>(__make_last_b_matrix);
    const H *_make_last_coeff_1_matrix   = reinterpret_cast<const H*>(__make_last_coeff_1_matrix);

    const H *_XA             = reinterpret_cast<const H*>(__XA) + blockIdx.x * (NC*CS*HF);
    const H *_XB             = reinterpret_cast<const H*>(__XB) + blockIdx.x * (NC*CS*HF);
    const H *_XC             = reinterpret_cast<const H*>(__XC) + blockIdx.x * (NC*CS*HF);
    const H *_Coeff          = reinterpret_cast<const H*>(__Coeff) + blockIdx.x * (NC*CS*CS);
    H *_Output = reinterpret_cast<H*>(__Output) + blockIdx.x * (NC*CS*HF);

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);

    st_bf<1, 4, ducks::st_layout::swizzle> (&XA_smem)[SMEM_POOL] = al.allocate<st_bf<1, 4, ducks::st_layout::swizzle>, SMEM_POOL>();
    st_bf<1, 4, ducks::st_layout::swizzle> (&XB_smem)[SMEM_POOL] = al.allocate<st_bf<1, 4, ducks::st_layout::swizzle>, SMEM_POOL>();
    st_bf<1, 4, ducks::st_layout::swizzle> (&XC_smem)[SMEM_POOL] = al.allocate<st_bf<1, 4, ducks::st_layout::swizzle>, SMEM_POOL>();
    st_bf<1, 1, ducks::st_layout::swizzle> (&Coeff_smem)[SMEM_POOL] = al.allocate<st_bf<1, 1, ducks::st_layout::swizzle>, SMEM_POOL>();

    rt_bf<4, 4, kittens::ducks::rt_layout::col> W1_reg;
    load(W1_reg, _W1, W1_reg.cols);

    rt_bf<1, 4> b1_bf_reg;
    load(b1_bf_reg, _b1, b1_bf_reg.cols);
    rt_fl<1, 4> b1_reg;
    copy(b1_reg, b1_bf_reg);

    rt_bf<1, 4> ln_w_reg_bf;
    rt_bf<1, 4> ln_b_reg_bf;
    load(ln_w_reg_bf, _ln_weight, ln_w_reg_bf.cols);
    load(ln_b_reg_bf, _ln_bias, ln_b_reg_bf.cols);

    rt_fl<1, 4> ln_w_reg;
    rt_fl<1, 4> ln_b_reg;
//    one(ln_w_reg);
//    zero(ln_b_reg);
    copy(ln_w_reg, ln_w_reg_bf);
    copy(ln_b_reg, ln_b_reg_bf);

    rt_bf<1, 1> cumsum_matrix_bf;
    rt_bf<1, 1> make_last_b_matrix_bf;
    rt_bf<1, 4, kittens::ducks::rt_layout::col> make_last_coeff_1_matrix_bf_col;
    load(cumsum_matrix_bf, _cumsum_matrix, cumsum_matrix_bf.cols);
    load(make_last_b_matrix_bf, _make_last_b_matrix, make_last_b_matrix_bf.cols);  // [K,K] @ [K,f] -> [K,f] broadcast last row of b_bar
    load(make_last_coeff_1_matrix_bf_col, _make_last_coeff_1_matrix, make_last_coeff_1_matrix_bf_col.cols);

    for (int i = 0; i < NC; i++) {

        if (i % SMEM_POOL == 0) {
            for (int j = 0; j < SMEM_POOL; j++) {
                load(XA_smem[j], _XA + (i + j) * X_STRIDE, 64);
                load(XB_smem[j], _XB + (i + j) * X_STRIDE, 64);
                load(XC_smem[j], _XC + (i + j) * X_STRIDE, 64);
                load(Coeff_smem[j], _Coeff + (i + j) * Coeff_STRIDE, 16);
            }
        }

        // Forward
        rt_bf<1, 4> XB_reg;
        load(XB_reg, XB_smem[i % SMEM_POOL]);

        rt_fl<1, 4> Z1_fl_reg;
        mma_AB(Z1_fl_reg, XB_reg, W1_reg, b1_reg); // [K,f]r <- [K,f]r @ [f,f]c + [K,f]r

        rt_bf<1, 4> XA_reg;
        load(XA_reg, XA_smem[i % SMEM_POOL]);
        sub(XA_reg, XA_reg, XB_reg);  // l2_tgt = XA - XB

        rt_fl<1, 4> XA_fl_reg;
        copy(XA_fl_reg, XA_reg);

        // LN fwd + bwd
        rt_fl<1, 4>::col_vec Z1_mean_reg;
        row_sum(Z1_mean_reg, Z1_fl_reg);  // [K,f]
        div(Z1_mean_reg, Z1_mean_reg, HF);

        rt_fl<1, 4> Z1_square_reg;
        sub_row(Z1_square_reg, Z1_fl_reg, Z1_mean_reg);
        mul(Z1_square_reg, Z1_square_reg, Z1_square_reg); // (Z1 - mu) ** 2

        rt_fl<1, 4>::col_vec Z1_std_reg;
        row_sum(Z1_std_reg, Z1_square_reg);  // [K,f]
        div(Z1_std_reg, Z1_std_reg, HF);
        add(Z1_std_reg, Z1_std_reg, 1e-6f);
//        add(Z1_std_reg, Z1_std_reg, 1e-4f);
        sqrt(Z1_std_reg, Z1_std_reg);

        rt_fl<1, 4> Z1_hat;  // normalized Z1 with 0 mean and 1 std
        sub_row(Z1_hat, Z1_fl_reg, Z1_mean_reg);
        div_row(Z1_hat, Z1_hat, Z1_std_reg);

        rt_fl<1, 4> LN_out_reg;  // affined by LN scale and bias
        mul(LN_out_reg, Z1_hat, ln_w_reg);  // [K,f] * [K,f]
        add(LN_out_reg, LN_out_reg, ln_b_reg);

        rt_fl<1, 4> dl_dZ1_hat;
        sub(dl_dZ1_hat, LN_out_reg, XA_fl_reg);
        mul(dl_dZ1_hat, dl_dZ1_hat, ln_w_reg);

        rt_fl<1, 4> dl_dZ1;
        mul(dl_dZ1, dl_dZ1_hat, HF);  // HF * dl_dZ1_hat

        rt_fl<1, 4>::col_vec dl_dZ1_vec_term;
        row_sum(dl_dZ1_vec_term, dl_dZ1_hat);
        sub_row(dl_dZ1, dl_dZ1, dl_dZ1_vec_term);   // HF * dl_dZ1_hat - dl_dZ1_hat.sum(dim=-1, keepdim=True)

        rt_fl<1, 4> dl_dZ1_term_3;
        mul(dl_dZ1_term_3, dl_dZ1_hat, Z1_hat);
        row_sum(dl_dZ1_vec_term, dl_dZ1_term_3);
        mul_row(dl_dZ1_term_3, Z1_hat, dl_dZ1_vec_term);

        sub(dl_dZ1, dl_dZ1, dl_dZ1_term_3);
        mul(Z1_std_reg, Z1_std_reg, HF);
        div_row(dl_dZ1, dl_dZ1, Z1_std_reg);

        rt_bf<1, 4> dl_dZ1_bf;
        copy(dl_dZ1_bf, dl_dZ1);

        // Get b1_bar of chunk: b1_bar = b1 - cumsum(dl_dZ1, dim=0): [K,f]
        rt_bf<1, 4, ducks::rt_layout::col> &dl_dZ1_col = swap_layout_inplace(dl_dZ1_bf);  // [K,f]
        rt_fl<1, 4> delta_b1_reg;
        zero(delta_b1_reg);

        rt_bf<1, 1> coeff_bf_reg;
        load(coeff_bf_reg, Coeff_smem[i % SMEM_POOL]);
        rt_bf<1, 1> Attn1_bf_reg;
        mul(Attn1_bf_reg, coeff_bf_reg, cumsum_matrix_bf);
        mma_AB(delta_b1_reg, Attn1_bf_reg, dl_dZ1_col, delta_b1_reg);  // delta_b1 = (coeff * Attn) @ dl_dZ1

        sub(b1_reg, b1_reg, delta_b1_reg);  // b1_bar = b1 - delta_b1

        // 2nd forward
        rt_bf<1, 4> XC_reg;
        load(XC_reg, XC_smem[i % SMEM_POOL]);

        rt_fl<1, 1> Attn1_reg;
        zero(Attn1_reg);
        mma_ABt(Attn1_reg, XC_reg, XB_reg, Attn1_reg);

        copy(Attn1_bf_reg, Attn1_reg);
        make_causal(Attn1_bf_reg, Attn1_bf_reg, base_types::constants<bf16>::zero());
        mul(Attn1_bf_reg, coeff_bf_reg, Attn1_bf_reg);

        rt_fl<1, 4> Z1_bar_term_1_reg;
        mma_AB(Z1_bar_term_1_reg, XC_reg, W1_reg, b1_reg); // [K,f']r <- [K,f]r @ [f,f']c + [K,f']r
        rt_bf<1, 4> Z1_bar_term_1_bf_reg;
        copy(Z1_bar_term_1_bf_reg, Z1_bar_term_1_reg);

        rt_fl<1, 4> Z1_bar_term_2_fl_reg;
        zero(Z1_bar_term_2_fl_reg);
        mma_AB(Z1_bar_term_2_fl_reg, Attn1_bf_reg, dl_dZ1_col, Z1_bar_term_2_fl_reg);  // [K,f]r <- [K,K]r, [K,f]c
        rt_bf<1, 4> Z1_bar_term_2_bf_reg;
        copy(Z1_bar_term_2_bf_reg, Z1_bar_term_2_fl_reg);

        sub(Z1_bar_term_1_bf_reg, Z1_bar_term_1_bf_reg, Z1_bar_term_2_bf_reg);
        store(_Output + i * CS * HF, Z1_bar_term_1_bf_reg, Z1_bar_term_1_bf_reg.cols);  // @xinhao: XC + LN(Z1_bar) can be done outside

        // delta_W1 at the last token in chunk
        rt_fl<1, 4> coeff_1_last_reg;
        zero(coeff_1_last_reg);
        rt_bf<1, 1> &coeff_transpose_bf_reg = transpose_inplace(coeff_bf_reg);
        mma_AB(coeff_1_last_reg, coeff_transpose_bf_reg, make_last_coeff_1_matrix_bf_col, coeff_1_last_reg); // [K,f]r <- [K,K]r, [K,f]c
        rt_bf<1, 4> coeff_1_last_bf_reg;
        copy(coeff_1_last_bf_reg, coeff_1_last_reg);
        mul(XB_reg, XB_reg, coeff_1_last_bf_reg);

        rt_bf<1, 4, kittens::ducks::rt_layout::col> &XB_col_reg = swap_layout_inplace(XB_reg);
        rt_fl<4, 4> delta_W1_reg;
        zero(delta_W1_reg);
        mma_AtB(delta_W1_reg, XB_col_reg, dl_dZ1_col, delta_W1_reg);

        rt_bf<4, 4> delta_W1_bf_reg;
        copy(delta_W1_bf_reg, delta_W1_reg);
        rt_bf<4, 4, kittens::ducks::rt_layout::col> &delta_W1_bf_col_reg = swap_layout_inplace(delta_W1_bf_reg);
        sub(W1_reg, W1_reg, delta_W1_bf_col_reg);

        rt_bf<1, 4> b1_bar_bf_reg;
        copy(b1_bar_bf_reg, b1_reg);
        rt_bf<1, 4, kittens::ducks::rt_layout::col> &b1_bar_bf_col_reg = swap_layout_inplace(b1_bar_bf_reg);
        zero(b1_reg);
        mma_AB(b1_reg, make_last_b_matrix_bf, b1_bar_bf_col_reg, b1_reg);  // [K,f]r <- [K,K]r @ [K,f]c + 0[K,f]r

    }

    store(_W1, W1_reg, W1_reg.cols);
    rt_bf<1, 4> b1_final_bf_reg;
    copy(b1_final_bf_reg, b1_reg);
    store(_b1, b1_final_bf_reg, b1_final_bf_reg.cols);

}


void
prefill_whole_loop_LN_bias
        (
                torch::Tensor W1,
                torch::Tensor b1,
                torch::Tensor ln_weight,
                torch::Tensor ln_bias,
                torch::Tensor cumsum_matrix,
                torch::Tensor make_last_b_matrix,
                torch::Tensor make_last_coeff_1_matrix,
                torch::Tensor XA,
                torch::Tensor XB,
                torch::Tensor XC,
                torch::Tensor Coeff,
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
            cumsum_matrix.data_ptr<T>(), make_last_b_matrix.data_ptr<T>(), make_last_coeff_1_matrix.data_ptr<T>(),
            XA.data_ptr<T>(), XB.data_ptr<T>(), XC.data_ptr<T>(), Coeff.data_ptr<T>(),
            Output.data_ptr<T>()
    );

//    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

}


template <typename H, typename T>
__global__
void ln_backward(
    const int CS, const int HF,
    const T* __dl_dZ1_hat, const T* __Z1_hat, const T* __std,
    T* __Output
) {
    const H *_dl_dZ1_hat = reinterpret_cast<const H*>(__dl_dZ1_hat) + blockIdx.x * (CS*HF);
    const H *_Z1_hat = reinterpret_cast<const H*>(__Z1_hat) + blockIdx.x * (CS*HF);
    const H *_std = reinterpret_cast<const H*>(__std) + blockIdx.x * (CS);
    H *_Output = reinterpret_cast<H*>(__Output) + blockIdx.x * (CS*HF);

    rt_bf<1, 4> dl_dZ1_bf;
    rt_bf<1, 4> Z1_hat_bf;
    rt_bf<1, 4>::col_vec Z1_std_reg_bf;
    load(dl_dZ1_bf, _dl_dZ1_hat, dl_dZ1_bf.cols);
    load(Z1_hat_bf, _Z1_hat, Z1_hat_bf.cols);
    load(Z1_std_reg_bf, _std);

    rt_fl<1, 4> dl_dZ1_hat;
    copy(dl_dZ1_hat, dl_dZ1_bf);
    rt_fl<1, 4> dl_dZ1;
    
    mul(dl_dZ1, dl_dZ1_hat, HF);  // HF * dl_dZ1_hat

    rt_fl<1, 4>::col_vec dl_dZ1_vec_term;
    row_sum(dl_dZ1_vec_term, dl_dZ1_hat);
    sub_row(dl_dZ1, dl_dZ1, dl_dZ1_vec_term);   // HF * dl_dZ1_hat - dl_dZ1_hat.sum(dim=-1, keepdim=True)

    rt_fl<1, 4> dl_dZ1_term_3;
    rt_fl<1, 4> Z1_hat;
    copy(Z1_hat, Z1_hat_bf);
    mul(dl_dZ1_term_3, dl_dZ1_hat, Z1_hat);
    row_sum(dl_dZ1_vec_term, dl_dZ1_term_3);
    mul_row(dl_dZ1_term_3, Z1_hat, dl_dZ1_vec_term);

    sub(dl_dZ1, dl_dZ1, dl_dZ1_term_3);
    div(dl_dZ1, dl_dZ1, HF);
    rt_fl<1, 4>::col_vec Z1_std_reg;
    copy(Z1_std_reg, Z1_std_reg_bf);
    div_row(dl_dZ1, dl_dZ1, Z1_std_reg);

    copy(dl_dZ1_bf, dl_dZ1);
    store(_Output, dl_dZ1_bf, dl_dZ1_bf.cols);
}

void 
prefill_ln_backward(
    torch::Tensor dl_dZ1_hat,
    torch::Tensor Z1_hat,
    torch::Tensor Z1_std,
    torch::Tensor Output,
    cudaStream_t stream
) {
    auto batch_mul_head = dl_dZ1_hat.size(0);
    auto CS = dl_dZ1_hat.size(1);
    auto HF = dl_dZ1_hat.size(2);

    using H = __nv_bfloat16;
    using T = c10::BFloat16;
    const int workers = 1;

    auto threads = workers * kittens::WARP_THREADS;

    ln_backward<H, T><<<batch_mul_head, threads, 0, stream>>>(
        CS, HF,
        dl_dZ1_hat.data_ptr<T>(), Z1_hat.data_ptr<T>(), Z1_std.data_ptr<T>(),
        Output.data_ptr<T>()
    );
}



