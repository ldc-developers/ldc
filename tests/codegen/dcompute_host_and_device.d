// Check that we can generate code for both the host and device in one compiler invocation
// REQUIRES: target_NVPTX
// RUN: %ldc -mdcompute-targets=cuda-350 -m64 -output-ll -mdcompute-file-prefix=host_and_device -Iinputs -output-o %s %S/inputs/kernel.d
// RUN: FileCheck %s --check-prefix=PTX < host_and_device_cuda350_64.ptx
// RUN: FileCheck %s --check-prefix=LL < dcompute_host_and_device.ll

import inputs.kernel : k_foo;

import ldc.dcompute;

int tlGlobal;
__gshared int gGlobal;

void main(string[] args)
{
    tlGlobal = 0;
    gGlobal  = 0;
    string s = foo.mangleof;
    string k_s = k_foo.mangleof;

    GlobalPointer!float global_x;
    foo(global_x);
}

void foo(GlobalPointer!float x_in) {
    // CHECK-LABEL: k_foo:
    SharedPointer!float shared_x;
	PrivatePointer!float private_x;
	ConstantPointer!float const_x;

    // LL: %4 = getelementptr inbounds %"ldc.dcompute.Pointer!(cast(AddrSpace)2u, float).Pointer", %"ldc.dcompute.Pointer!(cast(AddrSpace)2u, float).Pointer"* %shared_x, i32 0, i32 0
    // LL: %5 = load float*, float** %4
    // LL: %6 = getelementptr inbounds float, float* %5, i64 0
    // LL: %7 = getelementptr inbounds %"ldc.dcompute.Pointer!(cast(AddrSpace)1u, float).Pointer", %"ldc.dcompute.Pointer!(cast(AddrSpace)1u, float).Pointer"* %x_in, i32 0, i32 0
    // LL: %8 = load float*, float** %7
    // LL: %9 = getelementptr inbounds float, float* %8, i64 0
    // LL: %10 = load float, float* %9
    // LL: store float %10, float* %6
	shared_x[0] = x_in[0];
  
    // LL: %11 = getelementptr inbounds %"ldc.dcompute.Pointer!(cast(AddrSpace)0u, float).Pointer", %"ldc.dcompute.Pointer!(cast(AddrSpace)0u, float).Pointer"* %private_x, i32 0, i32 0
    // LL: %12 = load float*, float** %11
    // LL: %13 = getelementptr inbounds float, float* %12, i64 0
    // LL: %14 = getelementptr inbounds %"ldc.dcompute.Pointer!(cast(AddrSpace)1u, float).Pointer", %"ldc.dcompute.Pointer!(cast(AddrSpace)1u, float).Pointer"* %x_in, i32 0, i32 0
    // LL: %15 = load float*, float** %14
    // LL: %16 = getelementptr inbounds float, float* %15, i64 0
    // LL: %17 = load float, float* %16
    // LL: store float %17, float* %13
	private_x[0] = x_in[0];
  
    // LL: %18 = getelementptr inbounds %"ldc.dcompute.Pointer!(cast(AddrSpace)1u, float).Pointer", %"ldc.dcompute.Pointer!(cast(AddrSpace)1u, float).Pointer"* %x_in, i32 0, i32 0
    // LL: %19 = load float*, float** %18
    // LL: %20 = getelementptr inbounds float, float* %19, i64 0
    // LL: %21 = getelementptr inbounds %"ldc.dcompute.Pointer!(cast(AddrSpace)3u, immutable(float)).Pointer", %"ldc.dcompute.Pointer!(cast(AddrSpace)3u, immutable(float)).Pointer"* %const_x, i32 0, i32 0
    // LL: %22 = load float*, float** %21
    // LL: %23 = getelementptr inbounds float, float* %22, i64 0
    // LL: %24 = load float, float* %23
    // LL: store float %24, float* %20
	x_in[0] = const_x[0];

    // LL: %25 = getelementptr inbounds %"ldc.dcompute.Pointer!(cast(AddrSpace)1u, float).Pointer", %"ldc.dcompute.Pointer!(cast(AddrSpace)1u, float).Pointer"* %x_in, i32 0, i32 0
    // LL: %26 = load float*, float** %25
    // LL: %27 = getelementptr inbounds float, float* %26, i64 0
    // LL: %28 = getelementptr inbounds %"ldc.dcompute.Pointer!(cast(AddrSpace)2u, float).Pointer", %"ldc.dcompute.Pointer!(cast(AddrSpace)2u, float).Pointer"* %shared_x, i32 0, i32 0
    // LL: %29 = load float*, float** %28
    // LL: %30 = getelementptr inbounds float, float* %29, i64 0
    // LL: %31 = load float, float* %30
    // LL: store float %31, float* %27
    x_in[0] = shared_x[0];

    // LL: %32 = getelementptr inbounds %"ldc.dcompute.Pointer!(cast(AddrSpace)1u, float).Pointer", %"ldc.dcompute.Pointer!(cast(AddrSpace)1u, float).Pointer"* %x_in, i32 0, i32 0
    // LL: %33 = load float*, float** %32
    // LL: %34 = getelementptr inbounds float, float* %33, i64 0
    // LL: %35 = getelementptr inbounds %"ldc.dcompute.Pointer!(cast(AddrSpace)0u, float).Pointer", %"ldc.dcompute.Pointer!(cast(AddrSpace)0u, float).Pointer"* %private_x, i32 0, i32 0
    // LL: %36 = load float*, float** %35
    // LL: %37 = getelementptr inbounds float, float* %36, i64 0
    // LL: %38 = load float, float* %37
    // LL: store float %38, float* %34
	x_in[0] = private_x[0];
}

// CHECK-LABEL: k_foo:
// PTX: ld.global.f32
// PTX: st.shared.f32
// PTX: st.local.f32
// PTX: ld.const.f32
// PTX: ld.shared.f32
// PTX: ld.local.f32
