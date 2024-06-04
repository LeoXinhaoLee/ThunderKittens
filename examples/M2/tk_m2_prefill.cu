#include <iostream>
#include <string>
#include <math.h>
#include <assert.h>
//#include <mma_AB.h>
#include <cuda_runtime_api.h>
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

# include "../../src/kittens.cuh"
# include "../../src/common/pyutils/torch_helpers.cuh"

// **** ASYNC INCLUDE *****
#include <cuda/pipeline>
#include <cooperative_groups.h>

#define X_STRIDE 1024
#define W_STRIDE 16384

using namespace kittens;


//template <typename H, typename T>
//const int NC,
//        T* __W1, T* __W2,
//const T* __XA, const T* __XB, const T* __XC,
//        T* __Output
__global__
void prefill_whole_loop_ker(
        const int NC,
        __nv_bfloat16* __W1, __nv_bfloat16* __W2,
        const __nv_bfloat16* __XA, const __nv_bfloat16* __XB, const __nv_bfloat16* __XC,
        __nv_bfloat16* __Output
) {
//    H *_W1       = reinterpret_cast<H*>(__W1) + blockIdx.x * W_STRIDE;
//    H *_W2       = reinterpret_cast<H*>(__W2) + blockIdx.x * W_STRIDE;
//    const H *_XA       = reinterpret_cast<const H*>(__XA) + blockIdx.x * NC * X_STRIDE;
//    const H *_XB       = reinterpret_cast<const H*>(__XB) + blockIdx.x * NC * X_STRIDE;
//    const H *_XC       = reinterpret_cast<const H*>(__XC) + blockIdx.x * NC * X_STRIDE;
//    H *_Output = reinterpret_cast<H*>(__Output) + blockIdx.x * NC * X_STRIDE;
    __nv_bfloat16 *_W1       = __W1 + blockIdx.x * W_STRIDE;
    __nv_bfloat16 *_W2       = __W2 + blockIdx.x * W_STRIDE;
    const __nv_bfloat16 *_XA       = __XA + blockIdx.x * NC * X_STRIDE;
    const __nv_bfloat16 *_XC       = __XC + blockIdx.x * NC * X_STRIDE;
    const __nv_bfloat16 *_XB       = __XB + blockIdx.x * NC * X_STRIDE;
    __nv_bfloat16 *_Output = __Output + blockIdx.x * NC * X_STRIDE;

    rt_bf<4, 16, kittens::ducks::rt_layout::col> W1_col_reg;
//    rt_fl<4, 16> delta_W1_fl_reg;
//    rt_bf<4, 16> delta_W1_reg;

    rt_bf<16, 4, kittens::ducks::rt_layout::col> W2_col_reg;
//    rt_fl<16, 4> delta_W2_fl_reg;
//    rt_bf<16, 4> delta_W2_reg;

//    rt_bf<1, 4> XA_reg;
//    rt_bf<1, 4> XB_reg;
//    rt_bf<1, 4> XC_reg;

//    rt_fl<1, 16> Z1_fl_reg;
//    rt_bf<1, 16> Z1_reg;
//    rt_bf<1, 16> dl_dZ1_reg;
//    rt_fl<1, 16> dl_dZ1_fl_reg;

//    rt_fl<1, 16> Z1_bar_term_1_fl_reg;
//    rt_bf<1, 16> Z1_bar_term_1_reg;
//    rt_fl<1, 16> Z1_bar_term_2_fl_reg;
//    rt_bf<1, 16> Z1_bar_term_2_reg;

//    rt_fl<1, 4> Z2_fl_reg;
//    rt_bf<1, 4> dl_dZ2_reg;

//    rt_fl<1, 4> Z2_bar_term_1_fl_reg;
//    rt_bf<1, 4> Z2_bar_term_1_reg;
//    rt_fl<1, 4> Z2_bar_term_2_fl_reg;
//    rt_bf<1, 4> Z2_bar_term_2_reg;

//    rt_fl<1, 1> Attn_fl_reg;
//    rt_bf<1, 1> Attn_reg;

    load(W1_col_reg, _W1, W1_col_reg.cols);
    load(W2_col_reg, _W2, W2_col_reg.cols);

    for (int i = 0; i < NC; i++) {
        // Forward
        rt_fl<1, 16> Z1_fl_reg;
        rt_bf<1, 16> Z1_reg;
        rt_fl<1, 4> Z2_fl_reg;
        rt_bf<1, 4> XB_reg;

        load(XB_reg, _XB + i * X_STRIDE, XB_reg.cols);  // [K,f]
        zero(Z1_fl_reg);  // [K,4f]
        mma_AB(Z1_fl_reg, XB_reg, W1_col_reg, Z1_fl_reg); // [K,f]r, [f,4f]c -> [K,4f]r
        copy(Z1_reg, Z1_fl_reg);
        zero(Z2_fl_reg); // [K,f]
        mma_AB(Z2_fl_reg, Z1_reg, W2_col_reg, Z2_fl_reg); // [K,4f]r, [4f,f]c -> [K,f]r

        // dl_dZ2
        rt_bf<1, 4> dl_dZ2_reg;
        rt_bf<1, 4> XA_reg;

        load(XA_reg, _XA + i * X_STRIDE, XA_reg.cols);  // [K,f]
        copy(dl_dZ2_reg, Z2_fl_reg);
        sub(dl_dZ2_reg, dl_dZ2_reg, XA_reg);  // [K,f]

        // delta W2
        rt_fl<16, 4> delta_W2_fl_reg;
        rt_bf<16, 4> delta_W2_reg;
        rt_bf<1, 16, ducks::rt_layout::col> Z1_col_reg;
        swap_layout(Z1_col_reg, Z1_reg);
        rt_bf<1, 4, ducks::rt_layout::col> dl_dZ2_col_reg;
        swap_layout(dl_dZ2_col_reg, dl_dZ2_reg);  // cannot in-place swap dl_dZ21_reg since it will be needed later
        zero(delta_W2_fl_reg);
        mma_AtB(delta_W2_fl_reg, Z1_col_reg, dl_dZ2_col_reg, delta_W2_fl_reg);  // ([K,4f]c).t @ [K,f]c -> [4f,f]r
        copy(delta_W2_reg, delta_W2_fl_reg);
        rt_bf<16, 4, ducks::rt_layout::col> &delta_W2_col_reg = swap_layout_inplace(delta_W2_reg);  // TODO: tricky

        // dl_dZ1
        rt_bf<1, 16> dl_dZ1_reg;
        rt_fl<1, 16> dl_dZ1_fl_reg;

        zero(dl_dZ1_fl_reg);
        rt_bf<16, 4, kittens::ducks::rt_layout::row> W2_reg;
        swap_layout(W2_reg, W2_col_reg);
        mma_ABt(dl_dZ1_fl_reg, dl_dZ2_reg, W2_reg, dl_dZ1_fl_reg);  // [K,f]r @ [4f,f]r.t -> [K,4f]r
        copy(dl_dZ1_reg, dl_dZ1_fl_reg);

        // delta W1
        rt_fl<4, 16> delta_W1_fl_reg;
        rt_bf<4, 16> delta_W1_reg;
        rt_bf<1, 4, ducks::rt_layout::col> XB_col_reg;
        swap_layout(XB_col_reg, XB_reg);
        rt_bf<1, 16, ducks::rt_layout::col> &dl_dZ1_col_reg = swap_layout_inplace(dl_dZ1_reg);  // [K,4f]r->c TODO: tricy
        zero(delta_W1_fl_reg);
        mma_AtB(delta_W1_fl_reg, XB_col_reg, dl_dZ1_col_reg, delta_W1_fl_reg);  // ([K,f]c).t @ [K,4f]c -> [f,4f]r
        copy(delta_W1_reg, delta_W1_fl_reg);
        rt_bf<4, 16, ducks::rt_layout::col> &delta_W1_col_reg = swap_layout_inplace(delta_W1_reg);  // TODO: tricky

        // Attn1
        rt_fl<1, 1> Attn_fl_reg;
        rt_bf<1, 1> Attn_reg;
        rt_bf<1, 4> XC_reg;

        load(XC_reg, _XC + i * X_STRIDE, XC_reg.cols);  // [K,f]
        zero(Attn_fl_reg);  // [K,K]
        mma_ABt(Attn_fl_reg, XC_reg, XB_reg, Attn_fl_reg);  // [K,f]r @ [K,f]r.t -> [K,K]r
        copy(Attn_reg, Attn_fl_reg);
        make_causal(Attn_reg, Attn_reg, base_types::constants<bf16>::zero());

        // Z1_bar
        rt_fl<1, 16> Z1_bar_term_1_fl_reg;
        rt_bf<1, 16> Z1_bar_term_1_reg;
        rt_fl<1, 16> Z1_bar_term_2_fl_reg;
        rt_bf<1, 16> Z1_bar_term_2_reg;

        zero(Z1_bar_term_1_fl_reg);
        mma_AB(Z1_bar_term_1_fl_reg, XC_reg, W1_col_reg, Z1_bar_term_1_fl_reg);  // [K,f]r, [f,4f]c -> [K,4f]r
        copy(Z1_bar_term_1_reg, Z1_bar_term_1_fl_reg);

        zero(Z1_bar_term_2_fl_reg);
        mma_AB(Z1_bar_term_2_fl_reg, Attn_reg, dl_dZ1_col_reg, Z1_bar_term_2_fl_reg);  // [K,K]r, [K,f]c -> [K,f]r
        copy(Z1_bar_term_2_reg, Z1_bar_term_2_fl_reg);

        sub(Z1_bar_term_1_reg, Z1_bar_term_1_reg, Z1_bar_term_2_reg);  // cannot multiplex Z2_bar and Z2_bar_term_1_reg

        // Attn2
        zero(Attn_fl_reg);  // [K,K]
        mma_ABt(Attn_fl_reg, Z1_bar_term_1_reg, Z1_reg, Attn_fl_reg);  // [K,K]r, [K,f]r -> [K,f]r
        copy(Attn_reg, Attn_fl_reg);
        make_causal(Attn_reg, Attn_reg, base_types::constants<bf16>::zero());

        // Z2_bar
        rt_fl<1, 4> Z2_bar_term_1_fl_reg;
        rt_bf<1, 4> Z2_bar_term_1_reg;
        rt_fl<1, 4> Z2_bar_term_2_fl_reg;
        rt_bf<1, 4> Z2_bar_term_2_reg;

        zero(Z2_bar_term_1_fl_reg);
        mma_AB(Z2_bar_term_1_fl_reg, Z1_bar_term_1_reg, W2_col_reg, Z2_bar_term_1_fl_reg);
        copy(Z2_bar_term_1_reg, Z2_bar_term_1_fl_reg);

        zero(Z2_bar_term_2_fl_reg);
        mma_AB(Z2_bar_term_2_fl_reg, Attn_reg, dl_dZ2_col_reg, Z2_bar_term_2_fl_reg);
        copy(Z2_bar_term_2_reg, Z2_bar_term_2_fl_reg);

        sub(Z2_bar_term_1_reg, Z2_bar_term_1_reg, Z2_bar_term_2_reg);  // cannot multiplex Z2_bar and Z2_bar_term_1_reg

        // Store Output
        store(_Output + i * X_STRIDE, Z2_bar_term_1_reg, Z2_bar_term_1_reg.cols);

        // Updated W1, W2
        sub(W1_col_reg, W1_col_reg, delta_W1_col_reg);
        sub(W2_col_reg, W2_col_reg, delta_W2_col_reg);
    }

    store(_W1, W1_col_reg, W1_col_reg.cols);
    store(_W2, W2_col_reg, W2_col_reg.cols);
}


void
prefill_whole_loop
        (
                torch::Tensor W1, torch::Tensor W2,
                torch::Tensor XA, torch::Tensor XB, torch::Tensor XC,
                torch::Tensor Output,
                cudaStream_t stream
        ) {
    auto batch = XA.size(0);
    auto head = XA.size(1);
    auto NC = XA.size(2);
    auto CS = XA.size(3);
    auto HF = XA.size(4);
    auto HF_prime = W1.size(3);  // [BS,NH,HF,HF_prime]

//    std::cout << "HF: " << HF << std::endl;
//    std::cout << "HF_prime: " << HF_prime << std::endl;

    using H = __nv_bfloat16;
    using T = c10::BFloat16;
    const int workers = 1;

    auto threads = workers * kittens::WARP_THREADS;

//    prefill_whole_loop_ker<H, T><<<batch * head, threads, 0, stream>>>(
//            NC,
//            W1.data_ptr<T>(), W2.data_ptr<T>(),
//            XA.data_ptr<T>(), XB.data_ptr<T>(), XC.data_ptr<T>(),
//            Output.data_ptr<T>()
//    );

    __nv_bfloat16* W1_data_ptr = reinterpret_cast<H*> (W1.data_ptr<T>());
    __nv_bfloat16* W2_data_ptr = reinterpret_cast<H*> (W2.data_ptr<T>());
    const __nv_bfloat16* XA_data_ptr = reinterpret_cast<H*> (XA.data_ptr<T>());
    const __nv_bfloat16* XB_data_ptr = reinterpret_cast<H*> (XB.data_ptr<T>());
    const __nv_bfloat16* XC_data_ptr = reinterpret_cast<H*> (XC.data_ptr<T>());
    __nv_bfloat16* Output_data_ptr = reinterpret_cast<H*> (Output.data_ptr<T>());

    prefill_whole_loop_ker<<<batch * head, threads, 0, stream>>>(
            NC,
            W1_data_ptr, W2_data_ptr,
            XA_data_ptr, XB_data_ptr, XC_data_ptr,
            Output_data_ptr
    );


}

