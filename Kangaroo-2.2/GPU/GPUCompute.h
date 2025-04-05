#ifndef GPUCOMPUTEH
#define GPUCOMPUTEH

#include <cuda.h>
#include <stdint.h>
#include "GPUMath.h"

// Use grp sizee 32 for higher occupancy(can be tuned though)
#ifndef GPU_GRP_SIZE
#define GPU_GRP_SIZE 32
#endif

// nb_run iterations per kernel call (if not set, default) - configurable at runtime
#ifndef NB_RUN
#define NB_RUN 4096
#endif

__device__ void ComputeKangaroos(uint64_t* kangaroos, uint32_t maxFound, uint32_t* out_global, uint64_t dpMask,
                                 int nb_iterations = NB_RUN)
{
    extern __shared__ uint8_t smem_pool[];
    // Dynamic shared memory region to allow flexible partitioning
    uint64_t* shared_px  = reinterpret_cast<uint64_t*>(smem_pool);
    uint64_t* shared_py  = shared_px + GPU_GRP_SIZE * 4;
    uint64_t* shared_dist = shared_py + GPU_GRP_SIZE * 4;
#ifdef USE_SYMMETRY
    uint64_t* shared_lastJump = shared_dist + GPU_GRP_SIZE * 2;
#endif

    int local_warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int localIdx = threadIdx.x % GPU_GRP_SIZE;   // subgroup small index
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Per block shared output counters to batch atomics:
    __shared__ uint32_t block_count;
    __shared__ uint32_t block_offsets[1024]; // Enough space for per block outputs, might need to adjust this if many outputs expected
    __shared__ uint32_t block_pos;           // index into outputItem buffer

    if(threadIdx.x == 0){
        block_count = 0;
        block_pos = atomicAdd(out_global, 0);
    }
    __syncthreads();

    // ------------------------------ Load kangaroos from global
#ifdef USE_SYMMETRY
    LoadKangaroos(kangaroos, (uint64_t(*)[4])shared_px, (uint64_t(*)[4])shared_py, (uint64_t(*)[2])shared_dist, shared_lastJump);
#else
    LoadKangaroos(kangaroos, (uint64_t(*)[4])shared_px, (uint64_t(*)[4])shared_py, (uint64_t(*)[2])shared_dist);
#endif

    __syncthreads();

#pragma unroll 8
    for(int run=0; run<nb_iterations; ++run){

        uint32_t jmp = static_cast<uint32_t>(shared_px[localIdx*4]) & (NB_JUMP - 1);
#ifdef USE_SYMMETRY
        if(jmp == shared_lastJump[localIdx])
            jmp = (shared_lastJump[localIdx] + 1) & (NB_JUMP-1);
        shared_lastJump[localIdx] = jmp;
#endif

        uint64_t dx[4], dy[4], _s[4], _p[4], rx[4], ry[4];

        ModSub256(dx, &shared_px[localIdx*4], jPx[jmp]);     // dx = X - Xj
        _ModInvGrouped(&dx);                                // invert dx
        ModSub256(dy, &shared_py[localIdx*4], jPy[jmp]);     // dy = Y - Yj

        _ModMult(_s, dy, dx);                                // s = dy/dx
        _ModSqr(_p, _s);                                     // p = s^2

        ModSub256(rx, _p, jPx[jmp]);
        ModSub256(rx, rx, &shared_px[localIdx*4]);
        ModSub256(ry, &shared_px[localIdx*4], rx);
        _ModMult(ry, _s);
        ModSub256(ry, ry, &shared_py[localIdx*4]);

        Load256(&shared_px[localIdx*4], rx);
        Load256(&shared_py[localIdx*4], ry);

        Add128(&shared_dist[localIdx*2], jD[jmp]);

#ifdef USE_SYMMETRY
        if(ModPositive256(&shared_py[localIdx*4])){
            ModNeg256Order(&shared_dist[localIdx*2]);
        }
#endif
        if((shared_px[localIdx*4 + 3] & dpMask) == 0){
            uint32_t myPos;
            myPos= atomicAdd(&block_count,1);

            if(myPos < 1024){ // room
                block_offsets[myPos] = localIdx; // tag local kangaroo idx
            }
        }
        __syncwarp(); // warp local sync enough, skip global __syncthreads()
    }

    __syncthreads();

    // One thread per block does final global atomicAdd, bulk emits results
    if(threadIdx.x == 0){
        uint32_t gPos = atomicAdd(out_global, block_count);
        block_pos = gPos;
    }
    __syncthreads();

    for(uint32_t i=threadIdx.x; i<block_count; i+=blockDim.x){
        uint32_t localKidx = block_offsets[i];

        if(block_pos + i < maxFound){

            uint64_t kIdx = (uint64_t)(IDX) + (uint64_t)(localKidx) * blockDim.x +
                            (uint64_t)(blockIdx.x) * (blockDim.x * GPU_GRP_SIZE);

            OutputDP(&shared_px[localKidx*4], &shared_dist[localKidx*2], &kIdx);
        }
    }

    __syncthreads();

#ifdef USE_SYMMETRY
    StoreKangaroos(kangaroos, (uint64_t(*)[4])shared_px, (uint64_t(*)[4])shared_py, (uint64_t(*)[2])shared_dist, shared_lastJump);
#else
    StoreKangaroos(kangaroos, (uint64_t(*)[4])shared_px, (uint64_t(*)[4])shared_py, (uint64_t(*)[2])shared_dist);
#endif
}



#endif // GPUCOMPUTEH
