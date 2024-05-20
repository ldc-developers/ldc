// Check that we can generate code for both the host and device in one compiler invocation
// REQUIRES: target_NVPTX
// RUN: %ldc -c -mdcompute-targets=cuda-350 -m64 -output-ll -mdcompute-file-prefix=host_and_device -Iinputs -output-o %s %S/inputs/kernel.d
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
    // LL-LABEL: foo
    SharedPointer!float shared_x;
	PrivatePointer!float private_x;
	ConstantPointer!float const_x;

    // LL: [[s_load_reg:%[0-9]*]] = load ptr, ptr {{%[0-9]*}}
    // LL: [[s_addr_reg:%[0-9]*]] = load ptr, ptr {{%[0-9]*}}
    // LL: [[s_store_reg:%[0-9]*]] = load float, ptr [[s_addr_reg]]
    // LL: store float [[s_store_reg]], ptr [[s_load_reg]]
	*shared_x = *x_in;
  
    // LL: [[p_load_reg:%[0-9]*]] = load ptr, ptr {{%[0-9]*}}
    // LL: [[p_addr_reg:%[0-9]*]] = load ptr, ptr {{%[0-9]*}}
    // LL: [[p_store_reg:%[0-9]*]] = load float, ptr [[p_addr_reg]]
    // LL: store float [[p_store_reg]], ptr [[p_load_reg]]
	*private_x = *x_in;
  
    // LL: [[c_load_reg:%[0-9]*]] = load ptr, ptr {{%[0-9]*}}
    // LL: [[c_addr_reg:%[0-9]*]] = load ptr, ptr {{%[0-9]*}}
    // LL: [[c_store_reg:%[0-9]*]] = load float, ptr [[c_addr_reg]]
    // LL: store float [[c_store_reg]], ptr [[c_load_reg]]
	*x_in = *const_x;

    // LL: [[g1_load_reg:%[0-9]*]] = load ptr, ptr {{%[0-9]*}}
    // LL: [[g1_addr_reg:%[0-9]*]] = load ptr, ptr {{%[0-9]*}}
    // LL: [[g1_store_reg:%[0-9]*]] = load float, ptr [[g1_addr_reg]]
    // LL: store float [[g1_store_reg]], ptr [[g1_load_reg]]
    *x_in = *shared_x;

    // LL: [[g2_load_reg:%[0-9]*]] = load ptr, ptr {{%[0-9]*}}
    // LL: [[g2_addr_reg:%[0-9]*]] = load ptr, ptr {{%[0-9]*}}
    // LL: [[g2_store_reg:%[0-9]*]] = load float, ptr [[g2_addr_reg]]
    // LL: store float [[g2_store_reg]], ptr [[g2_load_reg]]
	*x_in = *private_x;
}

// PTX-LABEL: k_foo
// PTX: ld.global.f32
// PTX: st.shared.f32
// PTX: st.local.f32
// PTX: ld.const.f32
// PTX: ld.shared.f32
// PTX: ld.local.f32
