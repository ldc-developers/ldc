// Check that we can generate code for both the host and device in one compiler invocation
// REQUIRES: target_NVPTX
// RUN: %ldc -mdcompute-targets=cuda-350 -mdcompute-file-prefix=host_and_device -Iinputs %s %S/inputs/kernel.d && FileCheck %s --check-prefix=PTX < host_and_device_cuda350_64.ptx

import inputs.kernel : foo;

import ldc.dcompute;

int tlGlobal;
__gshared int gGlobal;

void main(string[] args)
{
    // Use these in host code to make sure the llType gets reset for the compute pass
    GlobalPointer!float global_x;
    SharedPointer!float shared_x;
    PrivatePointer!float private_x;
    ConstantPointer!float const_x;

    tlGlobal = 0;
    gGlobal  = 0;
    string s = foo.mangleof;
}

// PTX: ld.global.f32
// PTX: st.shared.f32
// PTX: st.local.f32
// PTX: ld.const.f32
