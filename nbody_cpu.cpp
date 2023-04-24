#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define EPSILON 1e-8f

using namespace std::chrono;
using timer = high_resolution_clock;

const float dt = 0.01f; // time delta
const int iters = 10;   // number of iterations for the simulation to run
const float G = 6.67e-11;

/**
 * @brief simulated body 
 * includes it's position in 3D space
 * as well as its velocity
 */
typedef struct Body {
    float x, y, z;
    float vx, vy, vz;
} Body;

/**
 * @brief create the simulation by initialize the bodies
 * 
 * @param bods a pointer to an array of bodies
 * @param fields the number of total fields we need to fill up
 */
void init_bodies(float* bods, int fields){
    for (int i = 0; i < fields; i++){
        bods[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

/**
 * @brief simulate all interactions among all n bodies
 * 
 * @param b pointer to an array of bodies
 * @param n number of bodies in the simulation
 */
inline void simulate_interaction(Body* b, int n){
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++){
        // forces in the x, y, z direction
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;

        for (int j = 0; j < n; j++){
            float dx = b[j].x - b[i].x;
            float dy = b[j].y - b[i].y;
            float dz = b[j].z - b[i].z;
            float d = dx*dx + dy*dy + dz*dz + EPSILON;
            float denom = 1.0f / sqrtf(d);
            float denom_cubed = denom * denom * denom;

            fx += dx * denom_cubed; 
            fy += dy * denom_cubed; 
            fz += dz * denom_cubed;
        }

        b[i].vx += dt*fx;
        b[i].vy += dt*fy;
        b[i].vz += dt*fz;
    }
}

int main(int argc, char* argv[]){
    int n = 30000;
    if (argc > 1)
        n = atoi(argv[1]);
        
    int bytes = n*sizeof(Body);
    float* tmp = (float*) malloc(bytes);
    Body* bodies = (Body*) tmp;

    init_bodies(tmp, 6*n);
    
    for (int iter = 0; iter < iters; iter++){
        auto start = timer::now();

        simulate_interaction(bodies, n);

        // note: OpenMP SIMD is only noticable when compiled with -O1 or -O2
        // as -O3 tries to auto-vectorize loops like these
        #pragma omp simd
        for (int i = 0; i < n; i++){
            bodies[i].x += bodies[i].vx*dt;
            bodies[i].y += bodies[i].vy*dt;
            bodies[i].z += bodies[i].vz*dt;
        }

        auto end = timer::now();
        auto elapsed = duration_cast<milliseconds>(end - start).count();

        printf("Iter %d took %ld milliseconds\n", iter, elapsed);
    }

    free(tmp);
}