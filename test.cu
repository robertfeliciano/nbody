#include <stdio.h>
#include <stdlib.h>

int main(){

    float3 f = make_float3(5, 2, 3);

    printf("%f\n", f.x);
    printf("%f\n", f.y);
    printf("%f\n", f.z);
}