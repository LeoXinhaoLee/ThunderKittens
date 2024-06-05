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

#define X_STRIDE 1024   // 16 * 64
#define W_STRIDE 16384  // 64 * 256
#define SMEM_POOL 1
#define SMEM_BLOCK 3 * SMEM_POOL * X_STRIDE * 2  // bytes

using namespace kittens;


template <typename H, typename T>
__global__
void prefill_whole_loop_ker(
        const int NC,
        T* __W1, T* __W2,
        const T* __XA, const T* __XB, const T* __XC,
        T* __Output
) {
    H *_W1       = reinterpret_cast<H*>(__W1) + blockIdx.x * W_STRIDE;
    H *_W2       = reinterpret_cast<H*>(__W2) + blockIdx.x * W_STRIDE;
    const H *_XA       = reinterpret_cast<const H*>(__XA) + blockIdx.x * NC * X_STRIDE;
    const H *_XB       = reinterpret_cast<const H*>(__XB) + blockIdx.x * NC * X_STRIDE;
    const H *_XC       = reinterpret_cast<const H*>(__XC) + blockIdx.x * NC * X_STRIDE;
    H *_Output = reinterpret_cast<H*>(__Output) + blockIdx.x * NC * X_STRIDE;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);

    // SMEM: 3x Sx (16 * 64 * 2B) = 6S KB <= 164KB / 8 blocks = 20 -> S <=3
    st_bf<1, 4, ducks::st_layout::swizzle> (&XA_smem)[SMEM_POOL] = al.allocate<st_bf<1, 4, ducks::st_layout::swizzle>, SMEM_POOL>();
    st_bf<1, 4, ducks::st_layout::swizzle> (&XB_smem)[SMEM_POOL] = al.allocate<st_bf<1, 4, ducks::st_layout::swizzle>, SMEM_POOL>();
    st_bf<1, 4, ducks::st_layout::swizzle> (&XC_smem)[SMEM_POOL] = al.allocate<st_bf<1, 4, ducks::st_layout::swizzle>, SMEM_POOL>();

    rt_bf<4, 16, kittens::ducks::rt_layout::col> W1_col_reg;
    rt_bf<16, 4, kittens::ducks::rt_layout::col> W2_col_reg;

    load(W1_col_reg, _W1, W1_col_reg.cols); // 32KB
    load(W2_col_reg, _W2, W2_col_reg.cols); // 32KB - 64KB

    for (int i = 0; i < NC; i++) {

        if (i % SMEM_POOL == 0) {
            for (int j = 0; j < SMEM_POOL; j++) {
                load(XA_smem[j], _XA + (i + j) * X_STRIDE, 64);
                load(XB_smem[j], _XB + (i + j) * X_STRIDE, 64);
                load(XC_smem[j], _XC + (i + j) * X_STRIDE, 64);
            }
        }

        // Forward
        rt_fl<1, 16> Z1_fl_reg; // 16KB - 80KB
        rt_bf<1, 16> Z1_reg; // 8KB - 88KB
        rt_fl<1, 4> Z2_fl_reg; // 4KB - 92KB
        rt_bf<1, 4> XB_reg; // 2KB - 94KB

        load(XB_reg, XB_smem[i % SMEM_POOL]);
        zero(Z1_fl_reg);  // [K,4f]
        mma_AB(Z1_fl_reg, XB_reg, W1_col_reg, Z1_fl_reg); // [K,f]r, [f,4f]c -> [K,4f]r
        copy(Z1_reg, Z1_fl_reg); // 16KB - 78KB
        zero(Z2_fl_reg); // [K,f]
        mma_AB(Z2_fl_reg, Z1_reg, W2_col_reg, Z2_fl_reg); // [K,4f]r, [4f,f]c -> [K,f]r

        // dl_dZ2
        rt_bf<1, 4> dl_dZ2_reg; // 2KB - 80KB
        rt_bf<1, 4> XA_reg; // 2KB - 82KB

        load(XA_reg, XA_smem[i % SMEM_POOL]);
        copy(dl_dZ2_reg, Z2_fl_reg); // 4KB - 78KB
        sub(dl_dZ2_reg, dl_dZ2_reg, XA_reg);  // [K,f] // 2KB - 76KB

        // delta W2
        rt_fl<16, 4> delta_W2_fl_reg; // 64KB - 140KB
        rt_bf<16, 4> delta_W2_reg; // 32KB - 172KB
        rt_bf<1, 16, ducks::rt_layout::col> Z1_col_reg; // 8KB - 180KB
        swap_layout(Z1_col_reg, Z1_reg);
        rt_bf<1, 4, ducks::rt_layout::col> dl_dZ2_col_reg; // 2KB - 182KB
        swap_layout(dl_dZ2_col_reg, dl_dZ2_reg);  // cannot in-place swap dl_dZ21_reg since it will be needed later
        zero(delta_W2_fl_reg);
        mma_AtB(delta_W2_fl_reg, Z1_col_reg, dl_dZ2_col_reg, delta_W2_fl_reg);  // ([K,4f]c).t @ [K,f]c -> [4f,f]r // 8KB - 174KB
        copy(delta_W2_reg, delta_W2_fl_reg); // 64KB - 110KB
        rt_bf<16, 4, ducks::rt_layout::col> &delta_W2_col_reg = swap_layout_inplace(delta_W2_reg);  // TODO: tricky

        // dl_dZ1
        rt_bf<1, 16> dl_dZ1_reg; // 8KB - 118KB
        rt_fl<1, 16> dl_dZ1_fl_reg; // 16KB - 134KB

        zero(dl_dZ1_fl_reg);
        rt_bf<16, 4, kittens::ducks::rt_layout::row> W2_reg; // 32KB - 166KB
        swap_layout(W2_reg, W2_col_reg);
        mma_ABt(dl_dZ1_fl_reg, dl_dZ2_reg, W2_reg, dl_dZ1_fl_reg);  // [K,f]r @ [4f,f]r.t -> [K,4f]r // 32KB+2KB - 132KB
        copy(dl_dZ1_reg, dl_dZ1_fl_reg); // 16KB - 116KB

        // delta W1
        rt_fl<4, 16> delta_W1_fl_reg; // 64KB - 180KB
        rt_bf<4, 16> delta_W1_reg; // 32KB - 212KB
        rt_bf<1, 4, ducks::rt_layout::col> XB_col_reg; // 2KB - 214KB
        swap_layout(XB_col_reg, XB_reg);
        rt_bf<1, 16, ducks::rt_layout::col> &dl_dZ1_col_reg = swap_layout_inplace(dl_dZ1_reg);  // [K,4f]r->c TODO: tricy
        zero(delta_W1_fl_reg);
        mma_AtB(delta_W1_fl_reg, XB_col_reg, dl_dZ1_col_reg, delta_W1_fl_reg);  // ([K,f]c).t @ [K,4f]c -> [f,4f]r // 2KB - 212KB
        copy(delta_W1_reg, delta_W1_fl_reg); // 64KB - 148KB
        rt_bf<4, 16, ducks::rt_layout::col> &delta_W1_col_reg = swap_layout_inplace(delta_W1_reg);  // TODO: tricky

        // Attn1
        rt_fl<1, 1> Attn_fl_reg; // 1KB - 149KB
        rt_bf<1, 1> Attn_reg; // 0.5KB - 149.5KB
        rt_bf<1, 4> XC_reg; // 2KB - 151.5KB

        load(XC_reg, XC_smem[i % SMEM_POOL]);
        zero(Attn_fl_reg);  // [K,K]
        mma_ABt(Attn_fl_reg, XC_reg, XB_reg, Attn_fl_reg);  // [K,f]r @ [K,f]r.t -> [K,K]r // 2KB - 149.5KB
        copy(Attn_reg, Attn_fl_reg);
        make_causal(Attn_reg, Attn_reg, base_types::constants<bf16>::zero());

        // Z1_bar
        rt_fl<1, 16> Z1_bar_term_1_fl_reg; // 16KB - 165.5KB
        rt_bf<1, 16> Z1_bar_term_1_reg; // 8KB - 173.5KB
        rt_fl<1, 16> Z1_bar_term_2_fl_reg; // 16KB - 189.5KB
        rt_bf<1, 16> Z1_bar_term_2_reg; // 8KB - 197.5KB

        zero(Z1_bar_term_1_fl_reg);
        mma_AB(Z1_bar_term_1_fl_reg, XC_reg, W1_col_reg, Z1_bar_term_1_fl_reg);  // [K,f]r, [f,4f]c -> [K,4f]r // 2KB - 195.5KB
        copy(Z1_bar_term_1_reg, Z1_bar_term_1_fl_reg); // 16KB - 179.5KB
        sub(W1_col_reg, W1_col_reg, delta_W1_col_reg); // Updated W1

        zero(Z1_bar_term_2_fl_reg);
        mma_AB(Z1_bar_term_2_fl_reg, Attn_reg, dl_dZ1_col_reg, Z1_bar_term_2_fl_reg);  // [K,K]r, [K,f]c -> [K,f]r // 8KB - 171.5KB
        copy(Z1_bar_term_2_reg, Z1_bar_term_2_fl_reg); // 16KB - 155.5KB

        sub(Z1_bar_term_1_reg, Z1_bar_term_1_reg, Z1_bar_term_2_reg);  // cannot multiplex Z2_bar and Z2_bar_term_1_reg // 8KB - 147.5KB

        // Attn2
        zero(Attn_fl_reg);  // [K,K]
        mma_ABt(Attn_fl_reg, Z1_bar_term_1_reg, Z1_reg, Attn_fl_reg);  // [K,K]r, [K,f]r -> [K,f]r // 8KB - 139.5KB
        copy(Attn_reg, Attn_fl_reg); // 1KB - 138.5KB
        make_causal(Attn_reg, Attn_reg, base_types::constants<bf16>::zero());

        // Z2_bar
        rt_fl<1, 4> Z2_bar_term_1_fl_reg; // 4KB - 142.5KB
        rt_bf<1, 4> Z2_bar_term_1_reg; // 2KB - 144.5KB
        rt_fl<1, 4> Z2_bar_term_2_fl_reg; // 4KB - 148.5KB
        rt_bf<1, 4> Z2_bar_term_2_reg; // 2KB - 150.5KB

        zero(Z2_bar_term_1_fl_reg);
        mma_AB(Z2_bar_term_1_fl_reg, Z1_bar_term_1_reg, W2_col_reg, Z2_bar_term_1_fl_reg); // 8KB - 142.5KB
        copy(Z2_bar_term_1_reg, Z2_bar_term_1_fl_reg); // 4KB - 138.5KB

        // Updated W2
        sub(W2_col_reg, W2_col_reg, delta_W2_col_reg); // 32KB - 68KB (Should be 64KB)

        zero(Z2_bar_term_2_fl_reg);
        mma_AB(Z2_bar_term_2_fl_reg, Attn_reg, dl_dZ2_col_reg, Z2_bar_term_2_fl_reg); // 2.5KB - 140KB
        copy(Z2_bar_term_2_reg, Z2_bar_term_2_fl_reg); // 4KB - 136KB

        sub(Z2_bar_term_1_reg, Z2_bar_term_1_reg, Z2_bar_term_2_reg);  // cannot multiplex Z2_bar and Z2_bar_term_1_reg // 2KB - 134KB

        // Store Output
        store(_Output + i * X_STRIDE, Z2_bar_term_1_reg, Z2_bar_term_1_reg.cols); // 2KB - 132KB

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

    prefill_whole_loop_ker<H, T><<<batch * head, threads, SMEM_BLOCK, stream>>>(
            NC,
            W1.data_ptr<T>(), W2.data_ptr<T>(),
            XA.data_ptr<T>(), XB.data_ptr<T>(), XC.data_ptr<T>(),
            Output.data_ptr<T>()
    );

}
