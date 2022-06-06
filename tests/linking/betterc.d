// RUN: %ldc -betterC -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -betterC -run %s

// CHECK-NOT: ModuleInfoZ
// CHECK-NOT: ModuleRefZ
// CHECK-NOT: call void @ldc.register_dso

extern (C) int main(int argc, char** argv) {
    assert(argc == 1); // make sure we can link against C assert
    return 0;
}
