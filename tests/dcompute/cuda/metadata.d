//RUN: %ldc -mdcompute-targets=cuda-350 -output-ll %s && FileCheck %s < %t.ll
@compute module cuda.metadata;

import dcompute.attributes;
import dcompute.types.pointer;

@kernel extern(C) void CudaMetadataTest( GlobalPointer!float gf)
{
    
}

//CHECK: !nvvm.annotations = !{!{0-9+}}
//CHECK: !{0-9+} = {void ({.*})* @CudaMetadataTest, !"kernel", i32 1}
