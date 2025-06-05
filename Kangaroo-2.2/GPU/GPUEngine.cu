/*
* This file is part of the BTCCollider distribution (https://github.com/JeanLucPons/Kangaroo).
* Copyright (c) 2020 Jean Luc PONS.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/




/*
* This file is part of the BTCCollider distribution (https://github.com/JeanLucPons/Kangaroo).
* Copyright (c) 2020 Jean Luc PONS.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef WIN64
#include <unistd.h>
#include <stdio.h>
#endif

#include "GPUEngine.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include "../Timer.h"

#include "GPUMath.h"
#include "GPUCompute.h"
#include "GPUGroup.h"

// ---------------------------------------------------------------------------------------

__global__ void __launch_bounds__(blockSize, minBlocksPerSM) comp_kangaroos(uint64_t *kangaroos,uint32_t maxFound,uint32_t *found,uint64_t dpMask) {

  int xPtr = (blockIdx.x*blockDim.x*GPU_GRP_SIZE) * KSIZE; // x[4] , y[4] , d[2], lastJump
  ComputeKangaroos(kangaroos + xPtr,maxFound,found,dpMask);

}

// ---------------------------------------------------------------------------------------

#ifdef GPU_CHECK
__global__ void check_gpu() {

  uint64_t mask = 0x1;

  for(int i=0;i<63;i++) {
    float f = (float)(mask);
    int zeros = (*(uint32_t*)(&f) >> 23) - 127;
    if(zeros!=i) {
      printf("Warning, trailing zero count wrong for %d, recompile with -DNOFASTCTZ\n",zeros);  
    }
    mask = mask << 1;
  }

}
#endif

// ---------------------------------------------------------------------------------------

using namespace std;

int _ConvertSMVer2Cores(int major,int minor) {

  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
             // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
    { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
    { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
    { 0x30, 192 },
    { 0x32, 192 },
    { 0x35, 192 },
    { 0x37, 192 },
    { 0x50, 128 },
    { 0x52, 128 },
    { 0x53, 128 },
    { 0x60,  64 },
    { 0x61, 128 },
    { 0x62, 128 },
    { 0x70,  64 },
    { 0x72,  64 },
    { 0x75,  64 },
    { 0x86, 128 },
    { -1, -1 } };

  int index = 0;

  while(nGpuArchCoresPerSM[index].SM != -1) {
    if(nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  return 0;

}

void GPUEngine::SetWildOffset(Int* offset) {
  wildOffset.Set(offset);
}

GPUEngine::GPUEngine(int nbThreadGroup,int nbThreadPerGroup,int gpuId,uint32_t maxFound) {

  // Initialise CUDA
  this->nbThreadPerGroup = nbThreadPerGroup;
  initialised = false;
  cudaError_t err;

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if(error_id != cudaSuccess) {
    printf("GPUEngine: CudaGetDeviceCount %s\n",cudaGetErrorString(error_id));
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if(deviceCount == 0) {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  err = cudaSetDevice(gpuId);
  if(err != cudaSuccess) {
    printf("GPUEngine: %s\n",cudaGetErrorString(err));
    return;
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp,gpuId);

  this->nbThread = nbThreadGroup * nbThreadPerGroup;
  this->maxFound = maxFound;
  this->outputSize = (maxFound*ITEM_SIZE + 4);

  char tmp[512];
  sprintf(tmp,"GPU #%d %s (%dx%d cores) Grid(%dx%d)",
    gpuId,deviceProp.name,deviceProp.multiProcessorCount,
    _ConvertSMVer2Cores(deviceProp.major,deviceProp.minor),
    nbThread / nbThreadPerGroup,
    nbThreadPerGroup);
  deviceName = std::string(tmp);

  // Prefer L1 (We do not use __shared__ at all)
  err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  if(err != cudaSuccess) {
    printf("GPUEngine: %s\n",cudaGetErrorString(err));
    return;
  }

  // Allocate memory
  inputKangaroo = NULL;
  inputKangarooPinned = NULL;
  outputItem = NULL;
  outputItemPinned = NULL;
  jumpPinned = NULL;

  // Input kangaroos
  kangarooSize = nbThread * GPU_GRP_SIZE * KSIZE * 8;
  err = cudaMalloc((void **)&inputKangaroo,kangarooSize);
  if(err != cudaSuccess) {
    printf("GPUEngine: Allocate input memory: %s\n",cudaGetErrorString(err));
    return;
  }
  kangarooSizePinned = nbThreadPerGroup * GPU_GRP_SIZE *  KSIZE * 8;
  err = cudaHostAlloc(&inputKangarooPinned,kangarooSizePinned,cudaHostAllocWriteCombined | cudaHostAllocMapped);
  if(err != cudaSuccess) {
    printf("GPUEngine: Allocate input pinned memory: %s\n",cudaGetErrorString(err));
    return;
  }

  // OutputHash
  err = cudaMalloc((void **)&outputItem,outputSize);
  if(err != cudaSuccess) {
    printf("GPUEngine: Allocate output memory: %s\n",cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&outputItemPinned,outputSize,cudaHostAllocMapped);
  if(err != cudaSuccess) {
    printf("GPUEngine: Allocate output pinned memory: %s\n",cudaGetErrorString(err));
    return;
  }

  // Jump array
  jumpSize = NB_JUMP * 8 * 4;
  err = cudaHostAlloc(&jumpPinned,jumpSize,cudaHostAllocMapped);
  if(err != cudaSuccess) {
    printf("GPUEngine: Allocate jump pinned memory: %s\n",cudaGetErrorString(err));
    return;
  }

  lostWarning = false;
  initialised = true;
  wildOffset.SetInt32(0);

#ifdef GPU_CHECK
  check_gpu<<<1,1>>>();
#endif

  cudaStreamCreate(&computeStream);
  cudaStreamCreate(&transferStream);

  // allocate one big host structure for all kangaroo parameters to support SoA
  // allocate new pinned buffer for all thread groups to avoid tiny copies
  fullPinnedBufferSize = kangarooSize;
  cudaHostAlloc(&fullPinnedPinnedBuf, fullPinnedBufferSize, cudaHostAllocMapped);
  // (define fullPinnedPinnedBuf/fullPinnedBufferSize as class members)

}

GPUEngine::~GPUEngine() {
  if(inputKangaroo) cudaFree(inputKangaroo);
  if(outputItem) cudaFree(outputItem);
  if(inputKangarooPinned) cudaFreeHost(inputKangarooPinned);
  if(outputItemPinned) cudaFreeHost(outputItemPinned);
  if(jumpPinned) cudaFreeHost(jumpPinned);
  cudaFreeHost(fullPinnedPinnedBuf);
  cudaStreamDestroy(computeStream);
  cudaStreamDestroy(transferStream);
}

int GPUEngine::GetMemory() {
  return kangarooSize + outputSize + jumpSize;
}

int GPUEngine::GetGroupSize() {
  return GPU_GRP_SIZE;
}

bool GPUEngine::GetGridSize(int gpuId,int *x,int *y) {

  if(*x <= 0 || *y <= 0) {

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if(error_id != cudaSuccess) {
      printf("GPUEngine: CudaGetDeviceCount %s\n",cudaGetErrorString(error_id));
      return false;
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if(deviceCount == 0) {
      printf("GPUEngine: There are no available device(s) that support CUDA\n");
      return false;
    }

    if(gpuId >= deviceCount) {
      printf("GPUEngine::GetGridSize() Invalid gpuId\n");
      return false;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,gpuId);

    if(*x <= 0) *x = 2 * deviceProp.multiProcessorCount;
    if(*y <= 0) *y = 2 * _ConvertSMVer2Cores(deviceProp.major,deviceProp.minor);
    if(*y <= 0) *y = 128;

  }

  return true;

}

void *GPUEngine::AllocatePinnedMemory(size_t size) {

  void *buff;

  cudaError_t err = cudaHostAlloc(&buff,size,cudaHostAllocPortable);
  if(err != cudaSuccess) {
    printf("GPUEngine: AllocatePinnedMemory: %s\n",cudaGetErrorString(err));
    return NULL;
  }

  return buff;

}

void GPUEngine::FreePinnedMemory(void *buff) {
  cudaFreeHost(buff);
}

void GPUEngine::PrintCudaInfo() {

  cudaError_t err;

  const char *sComputeMode[] =
  {
    "Multiple host threads",
    "Only one host thread",
    "No host thread",
    "Multiple process threads",
    "Unknown",
    NULL
  };

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if(error_id != cudaSuccess) {
    printf("GPUEngine: CudaGetDeviceCount %s\n",cudaGetErrorString(error_id));
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if(deviceCount == 0) {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  for(int i = 0; i<deviceCount; i++) {

    err = cudaSetDevice(i);
    if(err != cudaSuccess) {
      printf("GPUEngine: cudaSetDevice(%d) %s\n",i,cudaGetErrorString(err));
      return;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,i);
    printf("GPU #%d %s (%dx%d cores) (Cap %d.%d) (%.1f MB) (%s)\n",
      i,deviceProp.name,deviceProp.multiProcessorCount,
      _ConvertSMVer2Cores(deviceProp.major,deviceProp.minor),
      deviceProp.major,deviceProp.minor,(double)deviceProp.totalGlobalMem / 1048576.0,
      sComputeMode[deviceProp.computeMode]);

  }

}

int GPUEngine::GetNbThread() {
  return nbThread;
}

void GPUEngine::SetKangaroos(Int *px,Int *py,Int *d) {

  // fill SoA into large host pinned buffer (instead of tiny copies)
  // layout: block stride contiguous
  for(int i=0;i<totalKangaroos;i++){
    kangaroo_struct_t *dst = &((kangaroo_struct_t*)fullPinnedPinnedBuf)[i];

    memcpy(dst->px, px[i].bits64, 4*8);
    memcpy(dst->py, py[i].bits64, 4*8);
    Int dAdj; dAdj.Set(&d[i]);
    if(i % 2 == WILD) dAdj.ModAddK1order(&wildOffset);
    memcpy(dst->d, dAdj.bits64, 2*8);
#ifdef USE_SYMMETRY
    dst->lastJump = NB_JUMP;
#endif
  }

  cudaMemcpyAsync(inputKangaroo, fullPinnedPinnedBuf, kangarooSize, cudaMemcpyHostToDevice, transferStream);
}

void GPUEngine::GetKangaroos(Int *px,Int *py,Int *d) {

  if(inputKangarooPinned==NULL ) {
    printf("GPUEngine: GetKangaroos: Cannot retreive kangaroos, mem has been freed\n");
    return;
  }

  // Sets the kangaroos of each thread
  int gSize = KSIZE * GPU_GRP_SIZE;
  int strideSize = nbThreadPerGroup * KSIZE;
  int nbBlock = nbThread / nbThreadPerGroup;
  int blockSize = nbThreadPerGroup * gSize;
  int idx = 0;

  for(int b = 0; b < nbBlock; b++) {

    uint32_t offset = b * blockSize;
    cudaMemcpy(inputKangarooPinned,inputKangaroo + offset,kangarooSizePinned,cudaMemcpyDeviceToHost);

    for(int g = 0; g < GPU_GRP_SIZE; g++) {

      for(int t = 0; t < nbThreadPerGroup; t++) {

        // X
        px[idx].bits64[0] = inputKangarooPinned[g * strideSize + t + 0 * nbThreadPerGroup];
        px[idx].bits64[1] = inputKangarooPinned[g * strideSize + t + 1 * nbThreadPerGroup];
        px[idx].bits64[2] = inputKangarooPinned[g * strideSize + t + 2 * nbThreadPerGroup];
        px[idx].bits64[3] = inputKangarooPinned[g * strideSize + t + 3 * nbThreadPerGroup];
        px[idx].bits64[4] = 0;

        // Y
        py[idx].bits64[0] = inputKangarooPinned[g * strideSize + t + 4 * nbThreadPerGroup];
        py[idx].bits64[1] = inputKangarooPinned[g * strideSize + t + 5 * nbThreadPerGroup];
        py[idx].bits64[2] = inputKangarooPinned[g * strideSize + t + 6 * nbThreadPerGroup];
        py[idx].bits64[3] = inputKangarooPinned[g * strideSize + t + 7 * nbThreadPerGroup];
        py[idx].bits64[4] = 0;

        // Distance
        Int dOff;
        dOff.SetInt32(0);
        dOff.bits64[0] = inputKangarooPinned[g * strideSize + t + 8 * nbThreadPerGroup];
        dOff.bits64[1] = inputKangarooPinned[g * strideSize + t + 9 * nbThreadPerGroup];
        if(idx % 2 == WILD) dOff.ModSubK1order(&wildOffset);
        d[idx].Set(&dOff);

        idx++;
      }

    }
  }

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    printf("GPUEngine: GetKangaroos: %s\n",cudaGetErrorString(err));
  }

}

void GPUEngine::SetKangaroo(uint64_t kIdx,Int *px,Int *py,Int *d) {

  kangaroo_struct_t k;
  memcpy(k.px, px->bits64, 4*8);
  memcpy(k.py, py->bits64, 4*8);
  Int dAdj; dAdj.Set(d);
  if(kIdx % 2 == WILD) dAdj.ModAddK1order(&wildOffset);
  memcpy(k.d, dAdj.bits64, 2*8);
#ifdef USE_SYMMETRY
  k.lastJump = NB_JUMP;
#endif

  cudaMemcpyAsync(((kangaroo_struct_t*)inputKangaroo)+kIdx, &k, sizeof(k), cudaMemcpyHostToDevice, transferStream);
}

bool GPUEngine::callKernel() {
  cudaMemsetAsync(outputItem,0,4,computeStream);
  size_t smem_size = (GPU_GRP_SIZE * (4+4+2)
#ifdef USE_SYMMETRY
                    + GPU_GRP_SIZE
#endif
                     ) * sizeof(uint64_t);
  comp_kangaroos<<<gridSize, blockSize, smem_size, computeStream>>>(inputKangaroo,maxFound,outputItem,dpMask);
  return cudaGetLastError()==cudaSuccess;
}

void GPUEngine::SetParams(uint64_t dpMask,Int *distance,Int *px,Int *py) {
  
  this->dpMask = dpMask;

  for(int i=0;i< NB_JUMP;i++)
    memcpy(jumpPinned + 2*i,distance[i].bits64,16);
  cudaMemcpyToSymbol(jD,jumpPinned,jumpSize/2);
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    printf("GPUEngine: SetParams: Failed to copy to constant memory: %s\n",cudaGetErrorString(err));
    return;
  }

  for(int i = 0; i < NB_JUMP; i++)
    memcpy(jumpPinned + 4 * i,px[i].bits64,32);
  cudaMemcpyToSymbol(jPx,jumpPinned,jumpSize);
  err = cudaGetLastError();
  if(err != cudaSuccess) {
    printf("GPUEngine: SetParams: Failed to copy to constant memory: %s\n",cudaGetErrorString(err));
    return;
  }

  for(int i = 0; i < NB_JUMP; i++)
    memcpy(jumpPinned + 4 * i,py[i].bits64,32);
  cudaMemcpyToSymbol(jPy,jumpPinned,jumpSize);
  err = cudaGetLastError();
  if(err != cudaSuccess) {
    printf("GPUEngine: SetParams: Failed to copy to constant memory: %s\n",cudaGetErrorString(err));
    return;
  }

}

bool GPUEngine::callKernelAndWait() {

  // Debug function
  callKernel();
  cudaMemcpy(outputItemPinned,outputItem,outputSize,cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    printf("GPUEngine: callKernelAndWait: %s\n",cudaGetErrorString(err));
    return false;
  }

  return true;

}

bool GPUEngine::Launch(std::vector<ITEM> &hashFound,bool spinWait) {


  hashFound.clear();

  // Get the result

  if(spinWait) {

    cudaMemcpy(outputItemPinned,outputItem,outputSize,cudaMemcpyDeviceToHost);

  } else {

    // Use cudaMemcpyAsync to avoid default spin wait of cudaMemcpy wich takes 100% CPU
    cudaEvent_t evt;
    cudaEventCreate(&evt);
    cudaMemcpyAsync(outputItemPinned,outputItem,4,cudaMemcpyDeviceToHost,0);
    cudaEventRecord(evt,0);
    while(cudaEventQuery(evt) == cudaErrorNotReady) {
      // Sleep 1 ms to free the CPU
      Timer::SleepMillis(1);
    }
    cudaEventDestroy(evt);

  }

  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    printf("GPUEngine: Launch: %s\n",cudaGetErrorString(err));
    return false;
  }

  // Look for prefix found
  uint32_t nbFound = outputItemPinned[0];
  if(nbFound > maxFound) {
    // prefix has been lost
    if(!lostWarning) {
      printf("\nWarning, %d items lost\nHint: Search with less threads (-g) or increse dp (-d)\n",(nbFound - maxFound));
      lostWarning = true;
    }
    nbFound = maxFound;
  }

  // When can perform a standard copy, the kernel is eneded
  cudaMemcpy(outputItemPinned,outputItem,nbFound*ITEM_SIZE + 4,cudaMemcpyDeviceToHost);

  for(uint32_t i = 0; i < nbFound; i++) {
    uint32_t *itemPtr = outputItemPinned + (i*ITEM_SIZE32 + 1);
    ITEM it;

    it.kIdx = *((uint64_t*)(itemPtr + 12));

    uint64_t *x = (uint64_t *)itemPtr;
    it.x.bits64[0] = x[0];
    it.x.bits64[1] = x[1];
    it.x.bits64[2] = x[2];
    it.x.bits64[3] = x[3];
    it.x.bits64[4] = 0;

    uint64_t *d = (uint64_t *)(itemPtr + 8);
    it.d.bits64[0] = d[0];
    it.d.bits64[1] = d[1];
    it.d.bits64[2] = 0;
    it.d.bits64[3] = 0;
    it.d.bits64[4] = 0;
    if(it.kIdx % 2 == WILD) it.d.ModSubK1order(&wildOffset);

    hashFound.push_back(it);
  }

  return callKernel();

}
