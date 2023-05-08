#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>

#define EPSILON 1e-8f
#define G 6.67e-11f;

using namespace std::chrono;
using timer = high_resolution_clock;

const float dt = 0.01f; // time delta
const int iters = 10;   // number of iterations for the simulation to run

/**
 * @brief simulated body 
 * includes it's position in 3D space
 * as well as its velocity
 */
typedef struct Body {
    float x, y, z, m;
    float vx, vy, vz, empty;
} Body;

Body center_obj = { 
    center_obj.m = 5000.0f
    // everything else will be initialized to zero
};

/**
 * @brief create the simulation by initialize the bodies
 * 
 * @param bods a pointer to an array of bodies
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
            float d = dx*dx + dy*dy + dz*dz + EPSILON * EPSILON;
            float denom = 1.0f / sqrtf(d);
            float denom_cubed = denom * denom * denom;

            fx += dx * denom_cubed; 
            fy += dy * denom_cubed; 
            fz += dz * denom_cubed;
        }
        
        // calculate interaction with center mass
        float dx = b[i].x - center_obj.x;
        float dy = b[i].y - center_obj.y;
        float dz = b[i].z - center_obj.z;
        float d = dx*dx + dy*dy + dz*dz + EPSILON * EPSILON;
        float denom = sqrtf(d);
        float denom_cubed = denom * denom * denom;

        float m_c = center_obj.m;

        fx -= m_c * dx * denom_cubed; 
        fy -= m_c * dy * denom_cubed; 
        fz -= m_c * dz * denom_cubed;


        b[i].vx += dt*fx*G;
        b[i].vy += dt*fy*G;
        b[i].vz += dt*fz*G;

        b[i].x += dt * b[i].vx;
        b[i].y += dt * b[i].vy;
        b[i].z += dt * b[i].vz;
    }
}

int main(int argc, char* argv[]){
    int n = 30000;
    if (argc > 1)
        n = atoi(argv[1]);
        
    int bytes = n*sizeof(Body);
    float* tmp = (float*) malloc(bytes);
    Body* bodies = (Body*) tmp;

    init_bodies(tmp, 8*n);
    
    for (int iter = 0; iter < iters; iter++){
        auto start = timer::now();

        simulate_interaction(bodies, n);

        auto end = timer::now();
        auto elapsed = duration_cast<microseconds>(end - start).count();
        float elapsed_ms = static_cast<float>(elapsed) / 1000;

        printf("Iter %d took %.2f milliseconds on the device\n", iter, elapsed_ms);
    }

    free(tmp);
}