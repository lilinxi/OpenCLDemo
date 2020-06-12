/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"

__global__ void AddOneKernel(const int* in, const int N, int* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    out[i] = in[i] + 1;
  }
}

void AddOneKernelLauncher(const int* in, const int N, int* out) {

	clock_t time_start=clock();
 	cudaEvent_t start,stop;
	// Generate events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsedTime = 0.0f; // Initialize elapsedTime;
	float totalTime = 0.0f;
  
	for(int j = 0;j<1000;j++)
	{
		// Trigger event 'start'
		cudaEventRecord(start, 0);

		TF_CHECK_OK(::tensorflow::GpuLaunchKernel(AddOneKernel, 32, 256, 0, nullptr,
                                              in, N, out));

		//此处相比源码有改写，加入了统计时间信息，大家注意下
    
		cudaEventRecord(stop, 0); // Trigger Stop event
		cudaEventSynchronize(stop); // Sync events (BLOCKS till last (stop in this case) has been recorded!)
		cudaEventElapsedTime(&elapsedTime, start, stop); // Calculate runtime, write to elapsedTime -- cudaEventElapsedTime returns value in milliseconds. Resolution ~0.5ms
		totalTime += elapsedTime;

		cudaDeviceSynchronize();
	}

 	clock_t time_end=clock();
	std::cout<<"cuda time use:"<<1000 * (time_end-time_start)/(double)CLOCKS_PER_SEC<<"us"<<std::endl;
	std::cout<<"event record time:"<<totalTime<<"us"<<std::endl;
	std::cout<<std::endl;
	// Destroy CUDA Event API Events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

#endif
