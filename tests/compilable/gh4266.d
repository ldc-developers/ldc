// REQUIRES: target_NVPTX
// RUN: %ldc -c -mdcompute-targets=cuda-350 %s

@compute(CompileFor.deviceOnly) module gh4266;
import ldc.dcompute;

pragma(LDC_intrinsic, "llvm.nvvm.barrier0")
void barrier0();

void callbarrier() {
    barrier0();
}
