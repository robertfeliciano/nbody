#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>

#define EPSILON 1e-8f
#define G 6.67e-11f
#define BLOCKSZ 768

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
        bods[i] = static_cast<float>(distribution(generator));
        if (on_mass == 3){
            bods[i] = static_cast<float>(mass_distribution(generator));
            on_mass = 0;
            continue;
        }
        on_mass++;
    }
}

__global__ void simulate_interaction(float4* p, float4* v, float dt, int n){
    float4 center_obj = { 0.0f, 0.0f, 0.0f, 5000.0f };

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        float fx = 0.0f; float fy = 0.0f; float fz = 0.0f;

        for (int j = 0; j < n; j++) {
            
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + EPSILON * EPSILON;
            float invDist = rsqrtf(distSqr);
            float denom_cubed = invDist * invDist * invDist;
    
            float m_j = p[j].w;

            fx += m_j * dx * denom_cubed; 
            fy += m_j * dy * denom_cubed; 
            fz += m_j * dz * denom_cubed;
        }

        // calculate interaction with center mass
        float dx = p[i].x - center_obj.x;
        float dy = p[i].y - center_obj.y;
        float dz = p[i].z - center_obj.z;
        float d = dx*dx + dy*dy + dz*dz + EPSILON * EPSILON;
        float denom = rsqrtf(d);
        float denom_cubed = denom * denom * denom;

        float m_c = center_obj.w;

        fx -= m_c * dx * denom_cubed; 
        fy -= m_c * dy * denom_cubed; 
        fz -= m_c * dz * denom_cubed;
  
        v[i].x += dt * G * fx;
        v[i].y += dt * G * fy;
        v[i].z += dt * G * fz;

        p[i].x += v[i].x*dt;
        p[i].y += v[i].y*dt;
        p[i].z += v[i].z*dt;
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

    // copy bodies for the cpu to use 

    float* d_tmp;
    cudaMalloc(&d_tmp, bytes);
    System d_bodies = { (float4*) d_tmp, ((float4*) d_tmp) + n};

    int dimGrid = (n + BLOCKSZ - 1)/BLOCKSZ;

    for (int i = 0; i < iters; i++){
        // first kernel launch takes forever
        // https://stackoverflow.com/questions/57709333/cuda-kernel-runs-faster-the-second-time-it-is-run-why

        cudaMemcpy(d_tmp, tmp, bytes, cudaMemcpyHostToDevice);
        // call kernel
        // #ifndef CHECK
        // cudaEventRecord was giving me zeros all the time. no idea why
        // decided to go with chrono because who cares
        auto start = timer::now();
        // #endif

        simulate_interaction<<<dimGrid, BLOCKSZ>>>(d_bodies.p, d_bodies.v, dt, n);

        // #ifndef CHECK
        auto end = timer::now();
        cudaMemcpy(tmp, d_tmp, bytes, cudaMemcpyDeviceToHost);
        auto elapsed = duration_cast<microseconds>(end - start).count();
        float elapsed_ms = static_cast<float>(elapsed) / 1000;

        printf("Iter %d took %.2f milliseconds on the device\n", i, elapsed_ms);
        // #endif
    }

    free(tmp);
    cudaFree(d_tmp);
}