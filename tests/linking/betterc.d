// RUN: %ldc -betterC -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -betterC -defaultlib= -run %s

// CHECK-NOT: ModuleInfoZ
// CHECK-NOT: ModuleRefZ
// CHECK-NOT: call void @ldc.register_dso
version (CRuntime_Microsoft) {
    extern(C) int mainCRTStartup() {
        return 0;
    }
}

extern (C) int main() {
    return 0;
}
