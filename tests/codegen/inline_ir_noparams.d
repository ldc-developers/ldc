// RUN: %ldc -O -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK: define {{.*}}11unreachableFZv
void unreachable()
{
    import ldc.llvmasm;
    // CHECK-NEXT: unreachable
    __ir!("unreachable", void)();
// CHECK-NEXT: }
}

extern bool flag;

// CHECK: define {{.*}}3bar
int bar()
{
    int r = 123;
    if (flag)
    {
        r = 456;
        unreachable();
    }
    // CHECK: ret i32 123
    return r;
}
