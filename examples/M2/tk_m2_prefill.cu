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
#define Coeff_STRIDE 256  // 16 * 16
#define SMEM_POOL 1
//#define SMEM_BLOCK 3 * SMEM_POOL * X_STRIDE * 2  // bytes
#define SMEM_BLOCK SMEM_POOL * (3 * X_STRIDE + Coeff_STRIDE) * 2  // bytes: XA/XB/XC/Coeff_1

using namespace kittens;

/*** BF16 Simplified Kernel ***/
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


/*** BF16 gelu Kernel ***/
template <typename H, typename T>
__global__
void prefill_whole_loop_gelu_ker(
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
        gelu(Z1_fl_reg, Z1_fl_reg);
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

        rt_fl<1, 16> dl_dZ1_diff_gelu_fl_reg;
        diff_gelu(dl_dZ1_diff_gelu_fl_reg, dl_dZ1_fl_reg);
        mul(dl_dZ1_fl_reg, dl_dZ1_fl_reg, dl_dZ1_diff_gelu_fl_reg);
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
//        rt_bf<1, 16> Z1_bar_term_2_reg; // 8KB - 197.5KB

        zero(Z1_bar_term_1_fl_reg);
        mma_AB(Z1_bar_term_1_fl_reg, XC_reg, W1_col_reg, Z1_bar_term_1_fl_reg);  // [K,f]r, [f,4f]c -> [K,4f]r // 2KB - 195.5KB
//        copy(Z1_bar_term_1_reg, Z1_bar_term_1_fl_reg); // 16KB - 179.5KB
        sub(W1_col_reg, W1_col_reg, delta_W1_col_reg); // Updated W1

        zero(Z1_bar_term_2_fl_reg);
        mma_AB(Z1_bar_term_2_fl_reg, Attn_reg, dl_dZ1_col_reg, Z1_bar_term_2_fl_reg);  // [K,K]r, [K,f]c -> [K,f]r // 8KB - 171.5KB
//        copy(Z1_bar_term_2_reg, Z1_bar_term_2_fl_reg); // 16KB - 155.5KB

//        sub(Z1_bar_term_1_reg, Z1_bar_term_1_reg, Z1_bar_term_2_reg);  // cannot multiplex Z2_bar and Z2_bar_term_1_reg // 8KB - 147.5KB
        sub(Z1_bar_term_1_fl_reg, Z1_bar_term_1_fl_reg, Z1_bar_term_2_fl_reg);
        gelu(Z1_bar_term_1_fl_reg, Z1_bar_term_1_fl_reg);
        copy(Z1_bar_term_1_reg, Z1_bar_term_1_fl_reg);

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
prefill_whole_loop_gelu
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

    using H = __nv_bfloat16;
    using T = c10::BFloat16;
    const int workers = 1;

    auto threads = workers * kittens::WARP_THREADS;

    prefill_whole_loop_gelu_ker<H, T><<<batch * head, threads, SMEM_BLOCK, stream>>>(
            NC,
            W1.data_ptr<T>(), W2.data_ptr<T>(),
            XA.data_ptr<T>(), XB.data_ptr<T>(), XC.data_ptr<T>(),
            Output.data_ptr<T>()
    );
}


/*** FP16 Simplified Kernel ***/
template <typename H, typename T>
__global__
void prefill_whole_loop_ker_fp16(
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

    st_hf<1, 4, ducks::st_layout::swizzle> (&XA_smem)[SMEM_POOL] = al.allocate<st_hf<1, 4, ducks::st_layout::swizzle>, SMEM_POOL>();
    st_hf<1, 4, ducks::st_layout::swizzle> (&XB_smem)[SMEM_POOL] = al.allocate<st_hf<1, 4, ducks::st_layout::swizzle>, SMEM_POOL>();
    st_hf<1, 4, ducks::st_layout::swizzle> (&XC_smem)[SMEM_POOL] = al.allocate<st_hf<1, 4, ducks::st_layout::swizzle>, SMEM_POOL>();

    rt_hf<4, 16, kittens::ducks::rt_layout::col> W1_col_reg;
    rt_hf<16, 4, kittens::ducks::rt_layout::col> W2_col_reg;

    load(W1_col_reg, _W1, W1_col_reg.cols);
    load(W2_col_reg, _W2, W2_col_reg.cols);

    for (int i = 0; i < NC; i++) {

        if (i % SMEM_POOL == 0) {
            for (int j = 0; j < SMEM_POOL; j++) {
                load(XA_smem[j], _XA + (i + j) * X_STRIDE, 64);
                load(XB_smem[j], _XB + (i + j) * X_STRIDE, 64);
                load(XC_smem[j], _XC + (i + j) * X_STRIDE, 64);
            }
        }

        // Forward
        rt_hf<1, 16> Z1_reg;
        rt_hf<1, 4> dl_dZ2_reg;
        rt_hf<1, 4> XB_reg;
        load(XB_reg, XB_smem[i % SMEM_POOL]);
        zero(Z1_reg);  // [K,4f]
        mma_AB(Z1_reg, XB_reg, W1_col_reg, Z1_reg); // [K,f]r, [f,4f]c -> [K,4f]r
        zero(dl_dZ2_reg); // [K,f]
        mma_AB(dl_dZ2_reg, Z1_reg, W2_col_reg, dl_dZ2_reg); // [K,4f]r, [4f,f]c -> [K,f]r

        rt_hf<1, 4> XA_reg; // 2KB - 82KB
        load(XA_reg, XA_smem[i % SMEM_POOL]);
        sub(dl_dZ2_reg, dl_dZ2_reg, XA_reg);  // [K,f]
        // delta W2
        rt_hf<16, 4> delta_W2_reg;
        rt_hf<1, 16, ducks::rt_layout::col> Z1_col_reg;
        swap_layout(Z1_col_reg, Z1_reg);
        rt_hf<1, 4, ducks::rt_layout::col> dl_dZ2_col_reg;
        swap_layout(dl_dZ2_col_reg, dl_dZ2_reg);  // cannot in-place swap dl_dZ21_reg since it will be needed later
        zero(delta_W2_reg);
        mma_AtB(delta_W2_reg, Z1_col_reg, dl_dZ2_col_reg, delta_W2_reg);  // ([K,4f]c).t @ [K,f]c -> [4f,f]r
        rt_hf<16, 4, ducks::rt_layout::col> &delta_W2_col_reg = swap_layout_inplace(delta_W2_reg);  // TODO: tricky

        // dl_dZ1
        rt_hf<1, 16> dl_dZ1_reg;

        zero(dl_dZ1_reg);
        rt_hf<16, 4, kittens::ducks::rt_layout::row> W2_reg;
        swap_layout(W2_reg, W2_col_reg);
        mma_ABt(dl_dZ1_reg, dl_dZ2_reg, W2_reg, dl_dZ1_reg);  // [K,f]r @ [4f,f]r.t -> [K,4f]r

        // delta W1
        rt_hf<4, 16> delta_W1_reg;
        rt_hf<1, 4, ducks::rt_layout::col> XB_col_reg;
        swap_layout(XB_col_reg, XB_reg);

        rt_hf<1, 16, ducks::rt_layout::col> &dl_dZ1_col_reg = swap_layout_inplace(dl_dZ1_reg);  // [K,4f]r->c TODO: tricy
        zero(delta_W1_reg);
        mma_AtB(delta_W1_reg, XB_col_reg, dl_dZ1_col_reg, delta_W1_reg);  // ([K,f]c).t @ [K,4f]c -> [f,4f]r
        rt_hf<4, 16, ducks::rt_layout::col> &delta_W1_col_reg = swap_layout_inplace(delta_W1_reg);  // TODO: tricky

        // Attn1
        rt_hf<1, 1> Attn_reg;
        rt_hf<1, 4> XC_reg;

        load(XC_reg, XC_smem[i % SMEM_POOL]);
        zero(Attn_reg);  // [K,K]
        mma_ABt(Attn_reg, XC_reg, XB_reg, Attn_reg);  // [K,f]r @ [K,f]r.t -> [K,K]r
        make_causal(Attn_reg, Attn_reg, base_types::constants<half>::zero());
        // Z1_bar
        rt_hf<1, 16> Z1_bar_term_1_reg;
        rt_hf<1, 16> Z1_bar_term_2_reg;

        zero(Z1_bar_term_1_reg);
        mma_AB(Z1_bar_term_1_reg, XC_reg, W1_col_reg, Z1_bar_term_1_reg);  // [K,f]r, [f,4f]c -> [K,4f]r
        sub(W1_col_reg, W1_col_reg, delta_W1_col_reg); // Updated W1

        zero(Z1_bar_term_2_reg);
        mma_AB(Z1_bar_term_2_reg, Attn_reg, dl_dZ1_col_reg, Z1_bar_term_2_reg);  // [K,K]r, [K,f]c -> [K,f]r

        sub(Z1_bar_term_1_reg, Z1_bar_term_1_reg, Z1_bar_term_2_reg);  // cannot multiplex Z2_bar and Z2_bar_term_1_reg

        // Attn2
        zero(Attn_reg);  // [K,K]
        mma_ABt(Attn_reg, Z1_bar_term_1_reg, Z1_reg, Attn_reg);  // [K,K]r, [K,f]r -> [K,f]r
        make_causal(Attn_reg, Attn_reg, base_types::constants<half>::zero());

        // Z2_bar
        rt_hf<1, 4> Z2_bar_term_1_reg;

        zero(Z2_bar_term_1_reg);
        mma_AB(Z2_bar_term_1_reg, Z1_bar_term_1_reg, W2_col_reg, Z2_bar_term_1_reg);
        // Updated W2
        sub(W2_col_reg, W2_col_reg, delta_W2_col_reg);
        
        rt_hf<1, 4> Z2_bar_term_2_reg;
        zero(Z2_bar_term_2_reg);
        mma_AB(Z2_bar_term_2_reg, Attn_reg, dl_dZ2_col_reg, Z2_bar_term_2_reg);
        copy(Z2_bar_term_2_reg, Z2_bar_term_2_reg);

        sub(Z2_bar_term_1_reg, Z2_bar_term_1_reg, Z2_bar_term_2_reg);  // cannot multiplex Z2_bar and Z2_bar_term_1_reg

        // Store Output
        store(_Output + i * X_STRIDE, Z2_bar_term_1_reg, Z2_bar_term_1_reg.cols);

    }

    store(_W1, W1_col_reg, W1_col_reg.cols);
    store(_W2, W2_col_reg, W2_col_reg.cols);
}


void
prefill_whole_loop_fp16
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

    using H = __half;
    using T = c10::Half;
    const int workers = 1;

    auto threads = workers * kittens::WARP_THREADS;

    prefill_whole_loop_ker_fp16<H, T><<<batch * head, threads, SMEM_BLOCK, stream>>>(
            NC,
            W1.data_ptr<T>(), W2.data_ptr<T>(),
            XA.data_ptr<T>(), XB.data_ptr<T>(), XC.data_ptr<T>(),
            Output.data_ptr<T>()
    );

}


/*** FP16 gelu coeff bias Kernel ***/
template <typename H, typename T>
__global__
void prefill_whole_loop_gelu_coeff_bias_ker_fp16(
        const int NH, const int NC, const int CS, const int HF, const int HF_prime,
        T* __W1, T* __W2,
        T* __b1, T* __b2,
        const T* __ln_weight, const T* __ln_bias,
        const T* __cumsum_matrix, const T* __make_last_b_matrix,
        const T* __make_last_coeff_1_matrix, const T* __make_last_coeff_2_matrix,
        const T* __XA, const T* __XB, const T* __XC, const T* __Coeff,
        T* __Output
) {
    H *_W1       = reinterpret_cast<H*>(__W1) + blockIdx.x * (HF * HF_prime);
    H *_W2       = reinterpret_cast<H*>(__W2) + blockIdx.x * (HF_prime * HF);
    H *_b1       = reinterpret_cast<H*>(__b1) + blockIdx.x * (CS * HF_prime);
    H *_b2       = reinterpret_cast<H*>(__b2) + blockIdx.x * (CS * HF);

//    const H *_ln_weight = reinterpret_cast<const H*>(__ln_weight) + (blockIdx.x % NH) * (CS*HF);
//    const H *_ln_bias   = reinterpret_cast<const H*>(__ln_bias) + (blockIdx.x % NH) * (CS*HF);

    const H *_cumsum_matrix              = reinterpret_cast<const H*>(__cumsum_matrix);
    const H *_make_last_b_matrix         = reinterpret_cast<const H*>(__make_last_b_matrix);
    const H *_make_last_coeff_1_matrix   = reinterpret_cast<const H*>(__make_last_coeff_1_matrix);
    const H *_make_last_coeff_2_matrix   = reinterpret_cast<const H*>(__make_last_coeff_2_matrix);

    const H *_XA             = reinterpret_cast<const H*>(__XA) + blockIdx.x * (NC*CS*HF);
    const H *_XB             = reinterpret_cast<const H*>(__XB) + blockIdx.x * (NC*CS*HF);
    const H *_XC             = reinterpret_cast<const H*>(__XC) + blockIdx.x * (NC*CS*HF);
    const H *_Coeff          = reinterpret_cast<const H*>(__Coeff) + blockIdx.x * (NC*CS*CS);
    H *_Output = reinterpret_cast<H*>(__Output) + blockIdx.x * (NC*CS*HF);

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);

    st_hf<1, 4, ducks::st_layout::swizzle> (&XA_smem)[SMEM_POOL] = al.allocate<st_hf<1, 4, ducks::st_layout::swizzle>, SMEM_POOL>();
    st_hf<1, 4, ducks::st_layout::swizzle> (&XB_smem)[SMEM_POOL] = al.allocate<st_hf<1, 4, ducks::st_layout::swizzle>, SMEM_POOL>();
    st_hf<1, 4, ducks::st_layout::swizzle> (&XC_smem)[SMEM_POOL] = al.allocate<st_hf<1, 4, ducks::st_layout::swizzle>, SMEM_POOL>();
    st_hf<1, 1, ducks::st_layout::swizzle> (&Coeff_smem)[SMEM_POOL] = al.allocate<st_hf<1, 1, ducks::st_layout::swizzle>, SMEM_POOL>();

    rt_hf<4, 16, kittens::ducks::rt_layout::col> W1_col_reg;
    rt_hf<16, 4, kittens::ducks::rt_layout::col> W2_col_reg;
    rt_hf<1, 16> b1_reg;
    rt_hf<1, 4> b2_reg;

    load(W1_col_reg, _W1, W1_col_reg.cols);
    load(W2_col_reg, _W2, W2_col_reg.cols);

    load(b1_reg, _b1, b1_reg.cols);
    load(b2_reg, _b2, b2_reg.cols);

    rt_hf<1, 1> cumsum_matrix_bf;
    rt_hf<1, 1> make_last_b_matrix_bf;
    rt_hf<1, 4, kittens::ducks::rt_layout::col> make_last_coeff_1_matrix_col;
    rt_hf<1, 16, kittens::ducks::rt_layout::col> make_last_coeff_2_matrix_col;
    load(cumsum_matrix_bf, _cumsum_matrix, cumsum_matrix_bf.cols);
    load(make_last_b_matrix_bf, _make_last_b_matrix, make_last_b_matrix_bf.cols);  // [K,K] @ [K,f] -> [K,f] broadcast last row of b_bar
    load(make_last_coeff_1_matrix_col, _make_last_coeff_1_matrix, make_last_coeff_1_matrix_col.cols);
    load(make_last_coeff_2_matrix_col, _make_last_coeff_2_matrix, make_last_coeff_2_matrix_col.cols);

    for (int i = 0; i < NC; i++) {

        if (i % SMEM_POOL == 0) {
            for (int j = 0; j < SMEM_POOL; j++) {
                load(XA_smem[j], _XA + (i + j) * X_STRIDE, 64);
                load(XB_smem[j], _XB + (i + j) * X_STRIDE, 64);
                load(XC_smem[j], _XC + (i + j) * X_STRIDE, 64);
                load(Coeff_smem[j], _Coeff + (i + j) * Coeff_STRIDE, 16);
            }
        }

        // Z1 = XB_chunk @ W1 + b1
        rt_hf<1, 4> XB_reg;
        load(XB_reg, XB_smem[i % SMEM_POOL]);
        rt_hf<1, 16> Z1_reg;
        mma_AB(Z1_reg, XB_reg, W1_col_reg, b1_reg); // [K,4f]r <- [K,f]r @ [f,4f]c + [K,4f]

        // X2 = gelu(Z1)
        rt_hf<1, 16> X2_reg;
        gelu(X2_reg, Z1_reg);

        // Z2 = X2 @ W2 + b2
        rt_hf<1, 4> dl_dZ2_reg;
        mma_AB(dl_dZ2_reg, X2_reg, W2_col_reg, b2_reg); // [K,f]r <- [K,4f]r @ [4f,f]c + [K,f]
        rt_hf<1, 4> XA_reg;
        load(XA_reg, XA_smem[i % SMEM_POOL]);

        // dl_dZ2 = Z2 - XA
        sub(dl_dZ2_reg, dl_dZ2_reg, XA_reg);

        // delta b2
        rt_hf<1, 1> coeff_reg;
        rt_hf<1, 1> coeff_transpose_reg;
        load(coeff_reg, Coeff_smem[i % SMEM_POOL]);
        transpose_sep(coeff_transpose_reg, coeff_reg);

        // coeff_last_X2 = (coeff_transpose @ [0...0|1].t) * X2
        rt_hf<1, 16> coeff_last_X2_reg;
        zero(coeff_last_X2_reg);
        mma_AB(coeff_last_X2_reg, coeff_transpose_reg, make_last_coeff_2_matrix_col, coeff_last_X2_reg); // [K,4f]r <- [K,K]r, [K,4f]c
        mul(coeff_last_X2_reg, X2_reg, coeff_last_X2_reg);
        rt_hf<1, 16, ducks::rt_layout::col> &coeff_last_X2_col_reg = swap_layout_inplace(coeff_last_X2_reg);

        // delta W2 = coeff_last_X2.transpose(-1,-2) @ dl_dZ2
        rt_hf<16, 4> delta_W2_reg;
        rt_hf<1, 4, ducks::rt_layout::col> dl_dZ2_col_reg;
        swap_layout(dl_dZ2_col_reg, dl_dZ2_reg);  // cannot in-place swap dl_dZ21_reg since it will be needed later
        zero(delta_W2_reg);
        mma_AtB(delta_W2_reg, coeff_last_X2_col_reg, dl_dZ2_col_reg, delta_W2_reg);  // [4f,f]r <- ([K,4f]c).t @ [K,f]c
        rt_hf<16, 4, ducks::rt_layout::col> &delta_W2_col_reg = swap_layout_inplace(delta_W2_reg);

        // dl_dX2 = dl_dZ2 @ W2.transpose(-1,-2)
        rt_hf<1, 16> dl_dZ1_reg;
        zero(dl_dZ1_reg);
        rt_hf<16, 4, kittens::ducks::rt_layout::row> W2_reg;
        swap_layout(W2_reg, W2_col_reg);
        mma_ABt(dl_dZ1_reg, dl_dZ2_reg, W2_reg, dl_dZ1_reg);  //  [K,4f]r <- [K,f]r @ [4f,f]r.t TODO: Is transpose_inplace twice faster?

        // dl_dZ1 = dl_dX2 * diff_gelu(Z1)
        rt_hf<1, 16> &diff_gelu_Z1_reg = Z1_reg;
        diff_gelu(diff_gelu_Z1_reg, Z1_reg);
        mul(dl_dZ1_reg, dl_dZ1_reg, diff_gelu_Z1_reg);

        // delta b1 = (coeff_chunk * Attn_b) @ dl_dZ1
        rt_hf<1, 16> delta_b1_reg;
        rt_hf<1, 1> Attn_reg;
        rt_hf<1, 16, ducks::rt_layout::col> &dl_dZ1_col_reg = swap_layout_inplace(dl_dZ1_reg);  // [K,4f]r->c
        zero(delta_b1_reg);
        mul(Attn_reg, coeff_reg, cumsum_matrix_bf);
        mma_AB(delta_b1_reg, Attn_reg, dl_dZ1_col_reg, delta_b1_reg);  // [K,4f]r <- [K,K]r @ [K,4f]c
        // b1_bar = b1 - delta_b1
        sub(b1_reg, b1_reg, delta_b1_reg);

        // delta b2 = (coeff_chunk * Attn_b) @ dl_dZ2
        rt_hf<1, 4> delta_b2_reg;
        zero(delta_b2_reg);
        mma_AB(delta_b2_reg, Attn_reg, dl_dZ2_col_reg, delta_b2_reg);  // [K,f]r <- [K,K]r @ [K,f]c
        // b2_bar = b2 - delta_b2
        sub(b2_reg, b2_reg, delta_b2_reg);

        // coeff_last_X1 = (coeff_transpose @ [0...0|1].t) * X1
        rt_hf<1, 4> coeff_last_X1_reg;
        zero(coeff_last_X1_reg);
        mma_AB(coeff_last_X1_reg, coeff_transpose_reg, make_last_coeff_1_matrix_col, coeff_last_X1_reg); // [K,f]r <- [K,K]r, [K,f]c
        mul(coeff_last_X1_reg, XB_reg, coeff_last_X1_reg);
        rt_hf<1, 4, ducks::rt_layout::col> &coeff_last_X1_col_reg = swap_layout_inplace(coeff_last_X1_reg);

        // delta W1 = coeff_last_X1.transpose(-1,-2) @ dl_dZ1
        rt_hf<4, 16> delta_W1_reg;
        zero(delta_W1_reg);
        mma_AtB(delta_W1_reg, coeff_last_X1_col_reg, dl_dZ1_col_reg, delta_W1_reg);  // [f,4f]r <- ([K,f]c).t @ [K,4f]c
        rt_hf<4, 16, ducks::rt_layout::col> &delta_W1_col_reg = swap_layout_inplace(delta_W1_reg);

        // Attn1 = coeff * Tril(XC @ XB.t)
        rt_hf<1, 4> XC_reg;
        load(XC_reg, XC_smem[i % SMEM_POOL]);
        zero(Attn_reg);  // [K,K]
        mma_ABt(Attn_reg, XC_reg, XB_reg, Attn_reg);  // [K,f]r @ [K,f]r.t -> [K,K]r
        make_causal(Attn_reg, Attn_reg, base_types::constants<half>::zero());
        mul(Attn_reg, coeff_reg, Attn_reg);

        // Z1_bar = XC @ W1 - Attn1 @ dl_dZ1 + b1_bar
        rt_hf<1, 16> Z1_bar_term_1_reg;
        mma_AB(Z1_bar_term_1_reg, XC_reg, W1_col_reg, b1_reg);  // [K,4f]r <- [K,f]r @ [f,4f]c + [K,4f]
        // Update W1; W1 = W1 - delta_W1 (last in chunk) since it's no longer needed in the current chunk
        sub(W1_col_reg, W1_col_reg, delta_W1_col_reg);
        // Update b1
        rt_hf<1, 16, kittens::ducks::rt_layout::col> b1_bar_col_reg;
        swap_layout(b1_bar_col_reg, b1_reg);
        zero(b1_reg);
        mma_AB(b1_reg, make_last_b_matrix_bf, b1_bar_col_reg, b1_reg);  // [K,4f]r <- [K,K]r @ [K,4f]c + 0[K,4f]r

        rt_hf<1, 16> Z1_bar_term_2_reg;
        zero(Z1_bar_term_2_reg);
        mma_AB(Z1_bar_term_2_reg, Attn_reg, dl_dZ1_col_reg, Z1_bar_term_2_reg);  // [K,f]r <- [K,K]r @ [K,f]c

        sub(Z1_bar_term_1_reg, Z1_bar_term_1_reg, Z1_bar_term_2_reg);

        // X2_bar = gelu(Z1_bar)
        rt_hf<1, 16> &X2_bar_reg = Z1_bar_term_1_reg;
        gelu(X2_bar_reg, Z1_bar_term_1_reg);

        // Attn2 = coeff * Tril(X2_bar @ X2.t)
        zero(Attn_reg);
        mma_ABt(Attn_reg, X2_bar_reg, X2_reg, Attn_reg);  // [K,K]r, [K,f]r -> [K,f]r
        make_causal(Attn_reg, Attn_reg, base_types::constants<half>::zero());
        mul(Attn_reg, coeff_reg, Attn_reg);

        // Z2_bar = X2_bar @ W2 - Attn2 @ dl_dZ2 + b2_bar
        rt_hf<1, 4> Z2_bar_term_1_reg;
        mma_AB(Z2_bar_term_1_reg, X2_bar_reg, W2_col_reg, b2_reg);
        // Updated W2
        sub(W2_col_reg, W2_col_reg, delta_W2_col_reg);
        // Update b2
        rt_hf<1, 4, kittens::ducks::rt_layout::col> b2_bar_col_reg;
        swap_layout(b2_bar_col_reg, b2_reg);
        zero(b2_reg);
        mma_AB(b2_reg, make_last_b_matrix_bf, b2_bar_col_reg, b2_reg);  // [K,f]r <- [K,K]r @ [K,f]c + 0[K,f]r

        rt_hf<1, 4> Z2_bar_term_2_reg;
        zero(Z2_bar_term_2_reg);
        mma_AB(Z2_bar_term_2_reg, Attn_reg, dl_dZ2_col_reg, Z2_bar_term_2_reg);

        sub(Z2_bar_term_1_reg, Z2_bar_term_1_reg, Z2_bar_term_2_reg);

        // Store Output
        store(_Output + i * X_STRIDE, Z2_bar_term_1_reg, Z2_bar_term_1_reg.cols);

    }

    store(_W1, W1_col_reg, W1_col_reg.cols);
    store(_W2, W2_col_reg, W2_col_reg.cols);
    store(_b1, b1_reg, b1_reg.cols);
    store(_b2, b2_reg, b2_reg.cols);

}


void
prefill_whole_loop_gelu_coeff_bias_fp16
        (
                torch::Tensor W1,
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
                cudaStream_t stream
        ) {
    auto batch = XA.size(0);
    auto head = XA.size(1);
    auto NC = XA.size(2);
    auto CS = XA.size(3);
    auto HF = XA.size(4);
    auto HF_prime = W1.size(3);  // [BS,NH,HF,HF_prime]

    using H = __half;
    using T = c10::Half;
    const int workers = 1;

    auto threads = workers * kittens::WARP_THREADS;

    prefill_whole_loop_gelu_coeff_bias_ker_fp16<H, T><<<batch * head, threads, SMEM_BLOCK, stream>>>(
            head, NC, CS, HF, HF_prime,
            W1.data_ptr<T>(), W2.data_ptr<T>(),
            b1.data_ptr<T>(), b2.data_ptr<T>(),
            ln_weight.data_ptr<T>(), ln_bias.data_ptr<T>(),
            cumsum_matrix.data_ptr<T>(), make_last_b_matrix.data_ptr<T>(),
            make_last_coeff_1_matrix.data_ptr<T>(), make_last_coeff_2_matrix.data_ptr<T>(),
            XA.data_ptr<T>(), XB.data_ptr<T>(), XC.data_ptr<T>(), Coeff.data_ptr<T>(),
            Output.data_ptr<T>()
    );

}


/*** FP16 gelu coeff bias LN Kernel ***/
template <typename H, typename T>
__global__
void prefill_whole_loop_gelu_coeff_bias_LN_ker_fp16(
        const int NH, const int NC, const int CS, const int HF, const int HF_prime,
        T* __W1, T* __W2,
        T* __b1, T* __b2,
        const T* __ln_weight, const T* __ln_bias,
        const T* __cumsum_matrix, const T* __make_last_b_matrix,
        const T* __make_last_coeff_1_matrix, const T* __make_last_coeff_2_matrix,
        const T* __XA, const T* __XB, const T* __XC, const T* __Coeff,
        T* __Output
) {
    H *_W1       = reinterpret_cast<H*>(__W1) + blockIdx.x * (HF * HF_prime);
    H *_W2       = reinterpret_cast<H*>(__W2) + blockIdx.x * (HF_prime * HF);
    H *_b1       = reinterpret_cast<H*>(__b1) + blockIdx.x * (CS * HF_prime);
    H *_b2       = reinterpret_cast<H*>(__b2) + blockIdx.x * (CS * HF);

    const H *_ln_weight = reinterpret_cast<const H*>(__ln_weight) + (blockIdx.x % NH) * (CS*HF);
    const H *_ln_bias   = reinterpret_cast<const H*>(__ln_bias) + (blockIdx.x % NH) * (CS*HF);

    const H *_cumsum_matrix              = reinterpret_cast<const H*>(__cumsum_matrix);
    const H *_make_last_b_matrix         = reinterpret_cast<const H*>(__make_last_b_matrix);
    const H *_make_last_coeff_1_matrix   = reinterpret_cast<const H*>(__make_last_coeff_1_matrix);
    const H *_make_last_coeff_2_matrix   = reinterpret_cast<const H*>(__make_last_coeff_2_matrix);

    const H *_XA             = reinterpret_cast<const H*>(__XA) + blockIdx.x * (NC*CS*HF);
    const H *_XB             = reinterpret_cast<const H*>(__XB) + blockIdx.x * (NC*CS*HF);
    const H *_XC             = reinterpret_cast<const H*>(__XC) + blockIdx.x * (NC*CS*HF);
    const H *_Coeff          = reinterpret_cast<const H*>(__Coeff) + blockIdx.x * (NC*CS*CS);
    H *_Output = reinterpret_cast<H*>(__Output) + blockIdx.x * (NC*CS*HF);

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);

    st_hf<1, 4, ducks::st_layout::swizzle> (&XA_smem)[SMEM_POOL] = al.allocate<st_hf<1, 4, ducks::st_layout::swizzle>, SMEM_POOL>();
    st_hf<1, 4, ducks::st_layout::swizzle> (&XB_smem)[SMEM_POOL] = al.allocate<st_hf<1, 4, ducks::st_layout::swizzle>, SMEM_POOL>();
    st_hf<1, 4, ducks::st_layout::swizzle> (&XC_smem)[SMEM_POOL] = al.allocate<st_hf<1, 4, ducks::st_layout::swizzle>, SMEM_POOL>();
    st_hf<1, 1, ducks::st_layout::swizzle> (&Coeff_smem)[SMEM_POOL] = al.allocate<st_hf<1, 1, ducks::st_layout::swizzle>, SMEM_POOL>();

    rt_hf<4, 16, kittens::ducks::rt_layout::col> W1_col_reg;
    rt_hf<16, 4, kittens::ducks::rt_layout::col> W2_col_reg;
    rt_hf<1, 16> b1_reg;
    rt_hf<1, 4> b2_reg;

    load(W1_col_reg, _W1, W1_col_reg.cols);
    load(W2_col_reg, _W2, W2_col_reg.cols);

    load(b1_reg, _b1, b1_reg.cols);
    load(b2_reg, _b2, b2_reg.cols);

    rt_hf<1, 4> ln_w_reg;
    rt_hf<1, 4> ln_b_reg;
    load(ln_w_reg, _ln_weight, ln_w_reg.cols);
    load(ln_b_reg, _ln_bias, ln_b_reg.cols);

    rt_hf<1, 1> cumsum_matrix_bf;
    rt_hf<1, 1> make_last_b_matrix_bf;
    rt_hf<1, 4, kittens::ducks::rt_layout::col> make_last_coeff_1_matrix_col;
    rt_hf<1, 16, kittens::ducks::rt_layout::col> make_last_coeff_2_matrix_col;
    load(cumsum_matrix_bf, _cumsum_matrix, cumsum_matrix_bf.cols);
    load(make_last_b_matrix_bf, _make_last_b_matrix, make_last_b_matrix_bf.cols);  // [K,K] @ [K,f] -> [K,f] broadcast last row of b_bar
    load(make_last_coeff_1_matrix_col, _make_last_coeff_1_matrix, make_last_coeff_1_matrix_col.cols);
    load(make_last_coeff_2_matrix_col, _make_last_coeff_2_matrix, make_last_coeff_2_matrix_col.cols);

    for (int i = 0; i < NC; i++) {

        if (i % SMEM_POOL == 0) {
            for (int j = 0; j < SMEM_POOL; j++) {
                load(XA_smem[j], _XA + (i + j) * X_STRIDE, 64);
                load(XB_smem[j], _XB + (i + j) * X_STRIDE, 64);
                load(XC_smem[j], _XC + (i + j) * X_STRIDE, 64);
                load(Coeff_smem[j], _Coeff + (i + j) * Coeff_STRIDE, 16);
            }
        }

        // Z1 = XB_chunk @ W1 + b1
        rt_hf<1, 4> XB_reg;
        load(XB_reg, XB_smem[i % SMEM_POOL]);
        rt_hf<1, 16> Z1_reg;
        mma_AB(Z1_reg, XB_reg, W1_col_reg, b1_reg); // [K,4f]r <- [K,f]r @ [f,4f]c + [K,4f]

        // X2 = gelu(Z1)
        rt_hf<1, 16> X2_reg;
        gelu(X2_reg, Z1_reg);

        // Z2 = X2 @ W2 + b2
        rt_hf<1, 4> Z2_reg;
        mma_AB(Z2_reg, X2_reg, W2_col_reg, b2_reg); // [K,f]r <- [K,4f]r @ [4f,f]c + [K,f]

        // l2_tgt = XA - XB
        rt_hf<1, 4> l2_target_reg;
        load(l2_target_reg, XA_smem[i % SMEM_POOL]);
        sub(l2_target_reg, l2_target_reg, XB_reg);

        // LN fwd + bwd
        rt_hf<1, 4>::col_vec Z2_mean_reg;
        row_sum(Z2_mean_reg, Z2_reg);  // [K,f]
        div(Z2_mean_reg, Z2_mean_reg, __float2half(float(HF)));

        rt_hf<1, 4> Z2_square_reg;
        sub_row(Z2_square_reg, Z2_reg, Z2_mean_reg);
        mul(Z2_square_reg, Z2_square_reg, Z2_square_reg); // (Z1 - mu) ** 2

        rt_hf<1, 4>::col_vec Z2_std_reg;
        row_sum(Z2_std_reg, Z2_square_reg);  // [K,f]
        div(Z2_std_reg, Z2_std_reg, __float2half(float(HF)));
        add(Z2_std_reg, Z2_std_reg, __float2half(1e-6f));
        sqrt(Z2_std_reg, Z2_std_reg);

        rt_hf<1, 4> Z2_hat;  // normalized Z1 with 0 mean and 1 std
        sub_row(Z2_hat, Z2_reg, Z2_mean_reg);
        div_row(Z2_hat, Z2_hat, Z2_std_reg);

        rt_hf<1, 4> LN_out_reg;  // affined by LN scale and bias
        mul(LN_out_reg, Z2_hat, ln_w_reg);  // [K,f] * [K,f]
        add(LN_out_reg, LN_out_reg, ln_b_reg);

        rt_hf<1, 4> dl_dZ2_hat;
        sub(dl_dZ2_hat, LN_out_reg, l2_target_reg);
        mul(dl_dZ2_hat, dl_dZ2_hat, ln_w_reg);

        rt_hf<1, 4> dl_dZ2_reg;
        mul(dl_dZ2_reg, dl_dZ2_hat, __float2half(float(HF)));  // HF * dl_dZ1_hat

        rt_hf<1, 4>::col_vec dl_dZ2_vec_term;
        row_sum(dl_dZ2_vec_term, dl_dZ2_hat);
        sub_row(dl_dZ2_reg, dl_dZ2_reg, dl_dZ2_vec_term);   // HF * dl_dZ1_hat - dl_dZ1_hat.sum(dim=-1, keepdim=True)

        rt_hf<1, 4> dl_dZ2_term_3;
        mul(dl_dZ2_term_3, dl_dZ2_hat, Z2_hat);
        row_sum(dl_dZ2_vec_term, dl_dZ2_term_3);
        mul_row(dl_dZ2_term_3, Z2_hat, dl_dZ2_vec_term);

        sub(dl_dZ2_reg, dl_dZ2_reg, dl_dZ2_term_3);
        mul(Z2_std_reg, Z2_std_reg, __float2half(float(HF)));
        div_row(dl_dZ2_reg, dl_dZ2_reg, Z2_std_reg);

        // delta b2
        rt_hf<1, 1> coeff_reg;
        rt_hf<1, 1> coeff_transpose_reg;
        load(coeff_reg, Coeff_smem[i % SMEM_POOL]);
        transpose_sep(coeff_transpose_reg, coeff_reg);

        // coeff_last_X2 = (coeff_transpose @ [0...0|1].t) * X2
        rt_hf<1, 16> coeff_last_X2_reg;
        zero(coeff_last_X2_reg);
        mma_AB(coeff_last_X2_reg, coeff_transpose_reg, make_last_coeff_2_matrix_col, coeff_last_X2_reg); // [K,4f]r <- [K,K]r, [K,4f]c
        mul(coeff_last_X2_reg, X2_reg, coeff_last_X2_reg);
        rt_hf<1, 16, ducks::rt_layout::col> &coeff_last_X2_col_reg = swap_layout_inplace(coeff_last_X2_reg);

        // delta W2 = coeff_last_X2.transpose(-1,-2) @ dl_dZ2
        rt_hf<16, 4> delta_W2_reg;
        rt_hf<1, 4, ducks::rt_layout::col> dl_dZ2_col_reg;
        swap_layout(dl_dZ2_col_reg, dl_dZ2_reg);  // cannot in-place swap dl_dZ21_reg since it will be needed later
        zero(delta_W2_reg);
        mma_AtB(delta_W2_reg, coeff_last_X2_col_reg, dl_dZ2_col_reg, delta_W2_reg);  // [4f,f]r <- ([K,4f]c).t @ [K,f]c
        rt_hf<16, 4, ducks::rt_layout::col> &delta_W2_col_reg = swap_layout_inplace(delta_W2_reg);

        // dl_dX2 = dl_dZ2 @ W2.transpose(-1,-2)
        rt_hf<1, 16> dl_dZ1_reg;
        zero(dl_dZ1_reg);
        rt_hf<16, 4, kittens::ducks::rt_layout::row> W2_reg;
        swap_layout(W2_reg, W2_col_reg);
        mma_ABt(dl_dZ1_reg, dl_dZ2_reg, W2_reg, dl_dZ1_reg);  //  [K,4f]r <- [K,f]r @ [4f,f]r.t TODO: Is transpose_inplace twice faster?

        // dl_dZ1 = dl_dX2 * diff_gelu(Z1)
        rt_hf<1, 16> &diff_gelu_Z1_reg = Z1_reg;
        diff_gelu(diff_gelu_Z1_reg, Z1_reg);
        mul(dl_dZ1_reg, dl_dZ1_reg, diff_gelu_Z1_reg);

        // delta b1 = (coeff_chunk * Attn_b) @ dl_dZ1
        rt_hf<1, 16> delta_b1_reg;
        rt_hf<1, 1> Attn_reg;
        rt_hf<1, 16, ducks::rt_layout::col> &dl_dZ1_col_reg = swap_layout_inplace(dl_dZ1_reg);  // [K,4f]r->c
        zero(delta_b1_reg);
        mul(Attn_reg, coeff_reg, cumsum_matrix_bf);
        mma_AB(delta_b1_reg, Attn_reg, dl_dZ1_col_reg, delta_b1_reg);  // [K,4f]r <- [K,K]r @ [K,4f]c
        // b1_bar = b1 - delta_b1
        sub(b1_reg, b1_reg, delta_b1_reg);

        // delta b2 = (coeff_chunk * Attn_b) @ dl_dZ2
        rt_hf<1, 4> delta_b2_reg;
        zero(delta_b2_reg);
        mma_AB(delta_b2_reg, Attn_reg, dl_dZ2_col_reg, delta_b2_reg);  // [K,f]r <- [K,K]r @ [K,f]c
        // b2_bar = b2 - delta_b2
        sub(b2_reg, b2_reg, delta_b2_reg);

        // coeff_last_X1 = (coeff_transpose @ [0...0|1].t) * X1
        rt_hf<1, 4> coeff_last_X1_reg;
        zero(coeff_last_X1_reg);
        mma_AB(coeff_last_X1_reg, coeff_transpose_reg, make_last_coeff_1_matrix_col, coeff_last_X1_reg); // [K,f]r <- [K,K]r, [K,f]c
        mul(coeff_last_X1_reg, XB_reg, coeff_last_X1_reg);
        rt_hf<1, 4, ducks::rt_layout::col> &coeff_last_X1_col_reg = swap_layout_inplace(coeff_last_X1_reg);

        // delta W1 = coeff_last_X1.transpose(-1,-2) @ dl_dZ1
        rt_hf<4, 16> delta_W1_reg;
        zero(delta_W1_reg);
        mma_AtB(delta_W1_reg, coeff_last_X1_col_reg, dl_dZ1_col_reg, delta_W1_reg);  // [f,4f]r <- ([K,f]c).t @ [K,4f]c
        rt_hf<4, 16, ducks::rt_layout::col> &delta_W1_col_reg = swap_layout_inplace(delta_W1_reg);

        // Attn1 = coeff * Tril(XC @ XB.t)
        rt_hf<1, 4> XC_reg;
        load(XC_reg, XC_smem[i % SMEM_POOL]);
        zero(Attn_reg);  // [K,K]
        mma_ABt(Attn_reg, XC_reg, XB_reg, Attn_reg);  // [K,f]r @ [K,f]r.t -> [K,K]r
        make_causal(Attn_reg, Attn_reg, base_types::constants<half>::zero());
        mul(Attn_reg, coeff_reg, Attn_reg);

        // Z1_bar = XC @ W1 - Attn1 @ dl_dZ1 + b1_bar
        rt_hf<1, 16> Z1_bar_term_1_reg;
        mma_AB(Z1_bar_term_1_reg, XC_reg, W1_col_reg, b1_reg);  // [K,4f]r <- [K,f]r @ [f,4f]c + [K,4f]
        // Update W1; W1 = W1 - delta_W1 (last in chunk) since it's no longer needed in the current chunk
        sub(W1_col_reg, W1_col_reg, delta_W1_col_reg);
        // Update b1
        rt_hf<1, 16, kittens::ducks::rt_layout::col> b1_bar_col_reg;
        swap_layout(b1_bar_col_reg, b1_reg);
        zero(b1_reg);
        mma_AB(b1_reg, make_last_b_matrix_bf, b1_bar_col_reg, b1_reg);  // [K,4f]r <- [K,K]r @ [K,4f]c + 0[K,4f]r

        rt_hf<1, 16> Z1_bar_term_2_reg;
        zero(Z1_bar_term_2_reg);
        mma_AB(Z1_bar_term_2_reg, Attn_reg, dl_dZ1_col_reg, Z1_bar_term_2_reg);  // [K,f]r <- [K,K]r @ [K,f]c

        sub(Z1_bar_term_1_reg, Z1_bar_term_1_reg, Z1_bar_term_2_reg);

        // X2_bar = gelu(Z1_bar)
        rt_hf<1, 16> &X2_bar_reg = Z1_bar_term_1_reg;
        gelu(X2_bar_reg, Z1_bar_term_1_reg);

        // Attn2 = coeff * Tril(X2_bar @ X2.t)
        zero(Attn_reg);
        mma_ABt(Attn_reg, X2_bar_reg, X2_reg, Attn_reg);  // [K,K]r, [K,f]r -> [K,f]r
        make_causal(Attn_reg, Attn_reg, base_types::constants<half>::zero());
        mul(Attn_reg, coeff_reg, Attn_reg);

        // Z2_bar = X2_bar @ W2 - Attn2 @ dl_dZ2 + b2_bar
        rt_hf<1, 4> Z2_bar_term_1_reg;
        mma_AB(Z2_bar_term_1_reg, X2_bar_reg, W2_col_reg, b2_reg);
        // Updated W2
        sub(W2_col_reg, W2_col_reg, delta_W2_col_reg);
        // Update b2
        rt_hf<1, 4, kittens::ducks::rt_layout::col> b2_bar_col_reg;
        swap_layout(b2_bar_col_reg, b2_reg);
        zero(b2_reg);
        mma_AB(b2_reg, make_last_b_matrix_bf, b2_bar_col_reg, b2_reg);  // [K,f]r <- [K,K]r @ [K,f]c + 0[K,f]r

        rt_hf<1, 4> Z2_bar_term_2_reg;
        zero(Z2_bar_term_2_reg);
        mma_AB(Z2_bar_term_2_reg, Attn_reg, dl_dZ2_col_reg, Z2_bar_term_2_reg);

        sub(Z2_bar_term_1_reg, Z2_bar_term_1_reg, Z2_bar_term_2_reg);

        // Store Output
        store(_Output + i * X_STRIDE, Z2_bar_term_1_reg, Z2_bar_term_1_reg.cols);

    }

    store(_W1, W1_col_reg, W1_col_reg.cols);
    store(_W2, W2_col_reg, W2_col_reg.cols);
    store(_b1, b1_reg, b1_reg.cols);
    store(_b2, b2_reg, b2_reg.cols);

}


void
prefill_whole_loop_gelu_coeff_bias_LN_fp16
        (
                torch::Tensor W1,
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
                cudaStream_t stream
        ) {
    auto batch = XA.size(0);
    auto head = XA.size(1);
    auto NC = XA.size(2);
    auto CS = XA.size(3);
    auto HF = XA.size(4);
    auto HF_prime = W1.size(3);  // [BS,NH,HF,HF_prime]

    using H = __half;
    using T = c10::Half;
    const int workers = 1;

    auto threads = workers * kittens::WARP_THREADS;

    prefill_whole_loop_gelu_coeff_bias_LN_ker_fp16<H, T><<<batch * head, threads, SMEM_BLOCK, stream>>>(
            head, NC, CS, HF, HF_prime,
            W1.data_ptr<T>(), W2.data_ptr<T>(),
            b1.data_ptr<T>(), b2.data_ptr<T>(),
            ln_weight.data_ptr<T>(), ln_bias.data_ptr<T>(),
            cumsum_matrix.data_ptr<T>(), make_last_b_matrix.data_ptr<T>(),
            make_last_coeff_1_matrix.data_ptr<T>(), make_last_coeff_2_matrix.data_ptr<T>(),
            XA.data_ptr<T>(), XB.data_ptr<T>(), XC.data_ptr<T>(), Coeff.data_ptr<T>(),
            Output.data_ptr<T>()
    );

}
