// RUN: %ldc -betterC -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -betterC -defaultlib= -run %s

extern(C):

// CHECK-NOT: ModuleInfoZ
// CHECK-NOT: ModuleRefZ
// CHECK-NOT: call void @ldc.register_dso
version (CRuntime_Microsoft) {
    int mainCRTStartup() {
        return 0;
    }
}

void gh2425()
{
    char[16] buf;
    const str = "hello world";
    const ln = str.length;
    // _d_array_slice_copy() inlined by SimplifyDRuntimeCalls pass
    buf[0..ln] = str[0..ln];
}

int main() {
    gh2425();
    return 0;
}
