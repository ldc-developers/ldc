@compute(CompileFor.deviceOnly) module vulkan_vector_add;
import ldc.dcompute;

@kernel() void vector_add(
    GlobalPointer!float A0, GlobalPointer!float A1, GlobalPointer!float A2,
    GlobalPointer!float B0, GlobalPointer!float B1, GlobalPointer!float B2,
    GlobalPointer!float C0, GlobalPointer!float C1, GlobalPointer!float C2
) {
    *C0 = *A0 + *B0;
    *C1 = *A1 + *B1;
    *C2 = *A2 + *B2;
}
