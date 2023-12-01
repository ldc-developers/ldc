// REQUIRES: target_SPIRV

// FIXME: hits an assertion with SPIRV-LLVM-Translator for LLVM 15, see https://github.com/ldc-developers/ldc/pull/4010#issuecomment-1191820165
// XFAIL: atleast_llvm1500 && atmost_llvm1509

// RUN: %ldc -c -mdcompute-targets=ocl-220 -m64 -I%S/inputs -mdcompute-file-prefix=%t -output-ll -output-o %s && FileCheck %s < %t_ocl220_64.ll
@compute(CompileFor.deviceOnly) module dcompute_cl_images;
import ldc.dcompute;
import ldc.opencl;

// CHECK-DAG: %"ldc.dcompute.Pointer!(AddrSpace.Global, image1d_ro_t).Pointer" = type { {{%opencl.image1d_ro_t addrspace\(1\)\*|ptr addrspace\(1\)}} }
// CHECK-DAG: %"ldc.dcompute.Pointer!(AddrSpace.Global, image1d_wo_t).Pointer" = type { {{%opencl.image1d_wo_t addrspace\(1\)\*|ptr addrspace\(1\)}} }
// CHECK-DAG: %"ldc.dcompute.Pointer!(AddrSpace.Shared, sampler_t).Pointer" = type { {{%opencl.sampler_t addrspace\(2\)\*|ptr addrspace\(2\)}} }

pragma(mangle,"__translate_sampler_initializer")
    Sampler makeSampler(int);

pragma(mangle,"_Z11read_imagef11ocl_image1d_ro11ocl_sampleri")
    __vector(float[4]) read(GlobalPointer!image1d_ro_t, Sampler, int);

pragma(mangle,"_Z12write_imagef11ocl_image1d_woiDv4_f")
    void write(GlobalPointer!image1d_wo_t,int,__vector(float[4]));

@kernel void img(GlobalPointer!image1d_ro_t src, GlobalPointer!image1d_wo_t dst)
{
// CHECK: %{{[0-9+]}} = call spir_func {{%opencl.sampler_t addrspace\(2\)\*|ptr addrspace\(2\)}} @__translate_sampler_initializer(i32 0) {{.*}}
// CHECK: %{{[0-9+]}} = call spir_func <4 x float> @_Z11read_imagef11ocl_image1d_ro11ocl_sampleri({{%opencl.image1d_ro_t addrspace\(1\)\*|ptr addrspace\(1\)}} %.DComputePointerRewrite_arg, {{%opencl.sampler_t addrspace\(2\)\*|ptr addrspace\(2\)}} %.DComputePointerRewrite_arg1, i32 0) {{.*}}
    auto x = src.read(makeSampler(0), 0);
// CHECK: call spir_func void @_Z12write_imagef11ocl_image1d_woiDv4_f({{%opencl.image1d_wo_t addrspace\(1\)\*|ptr addrspace\(1\)}} %.DComputePointerRewrite_arg2, i32 0, <4 x float> %{{[0-9+]}})
    dst.write(0,x);
}

@kernel void img2(GlobalPointer!image1d_ro_t src, Sampler samp)
{
// CHECK: %{{[0-9+]}} = call spir_func <4 x float> @_Z11read_imagef11ocl_image1d_ro11ocl_sampleri({{%opencl.image1d_ro_t addrspace\(1\)\*|ptr addrspace\(1\)}} %.DComputePointerRewrite_arg, {{%opencl.sampler_t addrspace\(2\)\*|ptr addrspace\(2\)}} %.DComputePointerRewrite_arg1, i32 0) {{.*}}
    auto x = src.read(samp, 0);
}

// metadata
// CHECK: !{{[0-9+]}} = !{!"read_only", !"write_only"}
// CHECK: !{{[0-9+]}} = !{!"image1d_ro_t", !"image1d_wo_t"}

// CHECK: !{{[0-9+]}} = !{!"read_only", !"none"}
// CHECK: !{{[0-9+]}} = !{!"image1d_ro_t", !"sampler_t"}
