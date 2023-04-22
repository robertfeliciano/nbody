#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define EPSILON 1e-8f
#define BLOCKSZ 256

using namespace std::chrono;
using timer = high_resolution_clock;

const int iters = 10;   // number of iterations for the simulation to run

/**
 * @brief simulated system 
 * includes an array of the position vectors of every body
 * as well as an array of the velocity vectors of every body
 * in 3D space
 */
typedef struct System {
    float4* p;
    float4* v;
} System;

/**
 * @brief create the simulation by initialize the bodies
 * 
 * @param bods a pointer to body system
 * @param fields the number of total fields we need to fill up
 */
void init_bodies(float* bods, int fields){
    for (int i = 0; i < fields; i++){
        bods[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

__global__ void simulate_interaction(float4* p, float4* v, float dt, int n){
    int b = blockDim.x * blockIdx.x + threadIdx.x;
    if (b < n){
        // forces in the x, y, z direction
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;
        
        // iterate over all the other bodies in the simulation
        // this means iterating over the whole grid
        for (int t = 0; t < gridDim.x; t++){
            __shared__ float3 others[BLOCKSZ];
            float4 tpos = p[t * blockDim.x + threadIdx.x];
            others[threadIdx.x] = make_float3(tpos.x, tpos.y, tpos.z);
            __syncthreads();

            #pragma unroll
            for (int j = 0; j < BLOCKSZ; j++){
                float dx = others[j].x - p[b].x;
                float dy = others[j].y - p[b].y;
                float dz = others[j].z - p[b].z;
                float d = dx*dx + dy*dy + dz*dz + EPSILON;
                float denom = rsqrtf(d);
                float denom_cubed = denom * denom * denom;

                fx += dx * denom_cubed; 
                fy += dy * denom_cubed; 
                fz += dz * denom_cubed;
            }
            __syncthreads();
        }
        v[b].x += dt*fx;
        v[b].y += dt*fy;
        v[b].z += dt*fz;
    }
}

int main(int argc, char* argv[]){

    int n = 30000;
    if (argc > 1)
        n = atoi(argv[1]);
        
    const float dt = 0.01f; // time delta
        
    int bytes = n*2*sizeof(float4);
    float* tmp = (float*) malloc(bytes);
    System bodies = { (float4*) tmp, ((float4*) tmp) + n};

    init_bodies(tmp, 8*n);

    float* d_tmp;
    cudaMalloc(&d_tmp, bytes);
    System d_bodies = { (float4*) tmp, ((float4*) tmp) + n };

    int dimGrid = (n + BLOCKSZ - 1)/BLOCKSZ;

    for (int i = 0; i < iters; i++){
        // cudaEventRecord was giving me zeros all the time. no idea why
        // decided to go with chrono because who cares

        // first kernel launch takes forever
        // https://stackoverflow.com/questions/57709333/cuda-kernel-runs-faster-the-second-time-it-is-run-why

        cudaMemcpy(d_tmp, tmp, bytes, cudaMemcpyHostToDevice);
        // call kernel
        auto start = timer::now();
        simulate_interaction<<<dimGrid, BLOCKSZ>>>(d_bodies.p, d_bodies.v, dt, n);
        cudaMemcpy(tmp, d_tmp, bytes, cudaMemcpyDeviceToHost);

        #pragma omp simd
        for (int b = 0; b < n; b++){
            bodies.p[b].x += bodies.v[b].x*dt;
            bodies.p[b].y += bodies.v[b].y*dt;
            bodies.p[b].z += bodies.v[b].z*dt;
        }
        
        auto end = timer::now();
        auto elapsed = duration_cast<microseconds>(end - start).count();
        float elapsed_ms = static_cast<float>(elapsed) / 1000;

        printf("Iter %d took %.2f milliseconds on the device\n", i, elapsed_ms);
    }
    free(tmp);
    cudaFree(d_tmp);
}