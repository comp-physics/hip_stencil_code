#include <hip/hip_runtime.h>
#include <math.h>
#include <iostream>
#include <chrono>

#define THREADS_PER_BLOCK	1024

__global__ void hip_stencil_2(double *in, double *out, double h, int num_elements)
{
	int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
	if (i < num_elements-1)
		out[i] = (in[i+1] - in[i]) / h;
}

__global__ void hip_stencil_5(double *in, double *out, double h, int num_elements)
{
	int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x + 2;
	if (i < num_elements-2)
		out[i] = (-in[i+2] + 8*(in[i+1] - in[i-1]) + in[i-2]) / h;
}


int main(int argc, char **argv)
{
	hipDeviceProp_t devProp;
	hipGetDeviceProperties(&devProp, 0);
	std::cout << " System minor " << devProp.minor << std::endl;
	std::cout << " System major " << devProp.major << std::endl;
	std::cout << " System name " << devProp.name << std::endl;

	int N = 180000000;
	double pi = 3.141592653589793238462643383279;
	double L = 2 * pi;
	double h = L / double(N);

	double *d_in, *d_out;
	double *h_in = new double[N];
	double *h_out = new double[N];
	hipMalloc(&d_in, sizeof(double) * N);
	hipMalloc(&d_out, sizeof(double) * N);

	// Initialize
	std::cout << "Initializing ..." << std::endl;
	for (int i=0; i < N; i++)
		h_in[i] = sin((i+1) * h);

	hipMemcpy(d_in, h_in, sizeof(double) * N, hipMemcpyHostToDevice);

	int blocks = ceil((double)N / THREADS_PER_BLOCK);

	std::cout << "Running " << blocks << " blocks, " << THREADS_PER_BLOCK << " threads/block" << std::endl;
	auto begin_time = std::chrono::system_clock::now();
	hipLaunchKernelGGL(hip_stencil_5, dim3(blocks), dim3(THREADS_PER_BLOCK), 0, 0, d_in, d_out, 12 * h, N);
	hipDeviceSynchronize();
	auto end_time = std::chrono::system_clock::now();
	auto ms_duration = std::chrono::duration<float, std::milli>(end_time - begin_time);
	std::cout << "Elapsed time: " << ms_duration.count() << " ms." << std::endl;

	hipMemcpy(h_out, d_out, sizeof(double) * N, hipMemcpyDeviceToHost);

	// Check err
	double err(0.0);
	for (int i = 2; i < N-2; i++)
		err += abs(h_out[i] - cos((i+1) * h));
	err /= double(N-1);
	std::cout << "Error: " << err << std::endl;

	hipFree(d_in);
	hipFree(d_out);

	delete[] h_in;
	delete[] h_out;

	return 0;
}
