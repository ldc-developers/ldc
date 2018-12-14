// CHECK-LABEL: define{{.*}} @bar(
extern(C) void bar()
{
    // CHECK: localbarvar
    int localbarvar;
}

// Make sure we can use inline IR in non-textual IR compiles:
double inlineIR(double a)
{
    import ldc.llvmasm: __ir;
    auto s = __ir!(`ret double %0`, double)(a);
    return s;
}
