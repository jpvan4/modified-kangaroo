__device__ void ComputeKangaroos(uint64_t *kangaroos, uint32_t maxFound, uint32_t *out, uint64_t dpMask) {
    __shared__ uint64_t shared_px[GPU_GRP_SIZE][4];
    __shared__ uint64_t shared_py[GPU_GRP_SIZE][4];
    __shared__ uint64_t shared_dist[GPU_GRP_SIZE][2];
#ifdef USE_SYMMETRY
    __shared__ uint64_t shared_lastJump[GPU_GRP_SIZE];
#endif

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int groupId = threadId / GPU_GRP_SIZE;

    // Load kangaroo data into shared memory
#ifdef USE_SYMMETRY
    LoadKangaroos(kangaroos, shared_px, shared_py, shared_dist, shared_lastJump);
#else
    LoadKangaroos(kangaroos, shared_px, shared_py, shared_dist);
#endif

    // Ensure all threads have loaded data into shared memory
    __syncthreads();

for (int run = 0; run < NB_RUN; run++) {
    int localIdx = threadIdx.x % GPU_GRP_SIZE;
    uint32_t jmp = static_cast<uint32_t>(shared_px[localIdx][0]) & (NB_JUMP - 1);
#ifdef USE_SYMMETRY
    if (jmp == shared_lastJump[localIdx]) {
        jmp = (shared_lastJump[localIdx] + 1) % NB_JUMP;
    }
    shared_lastJump[localIdx] = jmp;
#endif

    uint64_t dx[4]; // Ensure dx is defined as an array of 4 uint64_t
    ModSub256(dx, shared_px[localIdx], jPx[jmp]); // Pass dx as a pointer
    _ModInvGrouped(&dx); // Pass dx directly since it is already an array
    uint64_t dy[4]; // Ensure dy has the correct size
    ModSub256(dy, shared_py[localIdx], jPy[jmp]);
    uint64_t _s[4];
    _ModMult(_s, dy, dx);
    uint64_t _p[4];
    _ModSqr(_p, _s);
    uint64_t rx[4];
    ModSub256(rx, _p, jPx[jmp]);
    ModSub256(rx, shared_px[localIdx]);
    uint64_t ry[4];
    ModSub256(ry, shared_px[localIdx], rx);
    _ModMult(ry, _s);
    ModSub256(ry, shared_py[localIdx]);
    Load256(shared_px[localIdx], rx);
    Load256(shared_py[localIdx], ry);
    Add128(shared_dist[localIdx], jD[jmp]);

#ifdef USE_SYMMETRY
    if (ModPositive256(shared_py[localIdx])) {
        ModNeg256Order(shared_dist[localIdx]);
    }
#endif

    if ((shared_px[localIdx][3] & dpMask) == 0) {
        uint32_t pos = atomicAdd(out, 1);
        if (pos < maxFound) {
            uint64_t kIdx = static_cast<uint64_t>(IDX) + static_cast<uint64_t>(localIdx) * blockDim.x +
                            static_cast<uint64_t>(blockIdx.x) * (blockDim.x * GPU_GRP_SIZE);
            OutputDP(shared_px[localIdx], shared_dist[localIdx], &kIdx);
        }
    }
    __syncthreads();
}
#ifdef USE_SYMMETRY
    StoreKangaroos(kangaroos, shared_px, shared_py, shared_dist, shared_lastJump);
#else
    StoreKangaroos(kangaroos, shared_px, shared_py, shared_dist);
#endif
}

// Error handling function
__device__ void CheckCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        // Handle error accordingly (e.g., set a flag, exit, etc.)
    }
}