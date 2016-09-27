//RUN: %ldc -mdcompute-targets=cuda-350 -output-ll %s && FileCheck %s < %t.ll
@compute module cuda.aspaces;

import dcompute.attributes;
import dcompute.types.pointer;
//CHECK: %"dcompute.types.pointer.Pointer!(1u, float).Pointer" = type { float addrspace(1)* }
//CHECK: %"dcompute.types.pointer.Pointer!(2u, float).Pointer" = type { float addrspace(3)* }
//CHECK: %"dcompute.types.pointer.Pointer!(3u, float).Pointer" = type { float addrspace(4)* }
//CHECK: %"dcompute.types.pointer.Pointer!(4u, float).Pointer" = type { float* }
@kernel extern(C) void CudaAddressSpaceTest(
                                GlobalPointer!float gf,
                                SharedPointer!float sf,
                                ConstantPointer!float cf,
                                GenericPointer!float ggf)
{
    
}