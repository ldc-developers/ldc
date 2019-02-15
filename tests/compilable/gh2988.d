// RUN: %ldc -c %s

import core.simd;

float4 getVector() { return float4(1.0f); }

void foo()
{
    // front-end implicitly casts from float4 to float[4]
    float x = getVector()[0];
}
