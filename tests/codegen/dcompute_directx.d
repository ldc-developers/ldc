// REQUIRES: target_DirectX
// RUN: %ldc -c -mdcompute-targets=dx-630 -m64 -mdcompute-file-prefix=directx -output-ll %s
@compute(CompileFor.deviceOnly) module dcompute_directx;
import ldc.dcompute;

@kernel void test(int) {}
