// Check that we can disable embedding of device code
// REQUIRES: target_NVPTX
// RUN: %ldc -c -mdcompute-targets=cuda-350 -m64 -output-ll -mdcompute-file-prefix=no_embed -fembed-dcompute=false -I%S/inputs -output-o %s %S/inputs/kernel.d
// RUN: FileCheck %s --check-prefix=LL < dcompute_no_embed.ll

import inputs.kernel : k_foo;
import ldc.dcompute;

void main(string[] args)
{
    string s = foo.mangleof;
    string k_s = k_foo.mangleof;

    GlobalPointer!float global_x;
    foo(global_x);
}

// LL-NOT: __dcompute_ptx_internal_

void foo(GlobalPointer!float x_in) {
    SharedPointer!float shared_x;
    *shared_x = *x_in;
}
