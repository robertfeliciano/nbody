#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>

#define EPSILON 1e-8f
#define BLOCKSZ 768
#define G 6.67e-11f

using namespace std::chrono;
using timer = high_resolution_clock;

int iters = 2;   // number of iterations for the simulation to run

/**
 * @brief simulated system 
 * includes an array of the position vectors of every body
 * as well as an array of the velocity vectors of every body
 * in 3D space.
 *
 * using a structure of arrays for better performance

   just too keep things simple i will store the mass in the fourth field
   of the p array
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
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(100,200);
    std::uniform_int_distribution<int> mass_distribution(3000,9000);
    int on_mass = 0;
    for (int i = 0; i < fields; i++){
        if (on_mass == 3){
            bods[i] = static_cast<float>(mass_distribution(generator));
            on_mass = 0;
            continue;
        }
        else {
            bods[i] = static_cast<float>(distribution(generator));
        }
        on_mass++;
    }
}

__global__ void simulate_interaction(float4* p, float4* v, float dt, int n){
    float4 center_obj = { 0.0f, 0.0f, 0.0f, 5000.0f };
    int b = blockDim.x * blockIdx.x + threadIdx.x;
    if (b < n){
        // forces in the x, y, z direction
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;
        
        // iterate over all the other bodies in the simulation
        // this means iterating over the whole grid
        for (int t = 0; t < gridDim.x; t++){
            __shared__ float4 others[BLOCKSZ];
            float4 curr = p[t * blockDim.x + threadIdx.x];
            // load other threads' info into shared memory
            others[threadIdx.x] = make_float4(curr.x, curr.y, curr.z, curr.w);
            __syncthreads();

            #pragma unroll
            for (int j = 0; j < BLOCKSZ; j++){
                float dx = others[j].x - p[b].x;
                float dy = others[j].y - p[b].y;
                float dz = others[j].z - p[b].z;
                float d = dx*dx + dy*dy + dz*dz + EPSILON * EPSILON;
                float denom = rsqrtf(d);
                float denom_cubed = denom * denom * denom;

                float m_j = others[j].w;

                fx += m_j * dx * denom_cubed; 
                fy += m_j * dy * denom_cubed; 
                fz += m_j * dz * denom_cubed;
            }
            __syncthreads();
        }       
        
        // calculate interaction with center mass
        float dx = p[b].x - center_obj.x;
        float dy = p[b].y - center_obj.y;
        float dz = p[b].z - center_obj.z;
        float d = dx*dx + dy*dy + dz*dz + EPSILON * EPSILON;
        float denom = rsqrtf(d);
        float denom_cubed = denom * denom * denom;

        float m_c = center_obj.w;

        fx -= m_c * dx * denom_cubed; 
        fy -= m_c * dy * denom_cubed; 
        fz -= m_c * dz * denom_cubed;

        v[b].x += dt * G * fx;
        v[b].y += dt * G * fy;
        v[b].z += dt * G * fz;

        p[b].x += v[b].x*dt;
        p[b].y += v[b].y*dt;
        p[b].z += v[b].z*dt;
    }
}

int main(int argc, char* argv[]){

    int n = 30000;
    if (argc > 1)
        n = atoi(argv[1]);
    if (argc > 2)
        iters = atoi(argv[2]);
        
    const float dt = 0.01f; // time delta
        
    int bytes = n*2*sizeof(float4);
    float* tmp = (float*) malloc(bytes);
    System bodies = { (float4*) tmp, ((float4*) tmp) + n};  // n is the offset to get to the velocity

    init_bodies(tmp, 8*n);


    float* d_tmp;
    cudaMalloc(&d_tmp, bytes);
    System d_bodies = { (float4*) d_tmp, ((float4*) d_tmp) + n};

    int dimGrid = (n + BLOCKSZ - 1)/BLOCKSZ;

    for (int i = 0; i < iters; i++){
        // first kernel launch takes forever
        // https://stackoverflow.com/questions/57709333/cuda-kernel-runs-faster-the-second-time-it-is-run-why

        cudaMemcpy(d_tmp, tmp, bytes, cudaMemcpyHostToDevice);
        // cudaEventRecord was giving me zeros all the time. no idea why
        // decided to go with chrono because who cares
        auto start = timer::now();

        // call kernel
        simulate_interaction<<<dimGrid, BLOCKSZ>>>(d_bodies.p, d_bodies.v, dt, n);
        
        auto end = timer::now();

        cudaMemcpy(tmp, d_tmp, bytes, cudaMemcpyDeviceToHost);

        // #pragma omp simd
        // for (int b = 0; b < n; b++){
        //     bodies.p[b].x += bodies.v[b].x*dt;
        //     bodies.p[b].y += bodies.v[b].y*dt;
        //     bodies.p[b].z += bodies.v[b].z*dt;
        // }
        auto elapsed = duration_cast<microseconds>(end - start).count();
        float elapsed_ms = static_cast<float>(elapsed) / 1000;

        printf("Iter %d took %.2f milliseconds on the device\n", i, elapsed_ms);
    }

    free(tmp);
    cudaFree(d_tmp);
}