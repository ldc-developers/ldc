// CHECK-LABEL: define{{.*}} @bar(
extern(C) void bar()
{
    // CHECK: localbarvar
    int localbarvar;
}

// Make sure we can use inline IR in non-textual IR compiles:
pragma(LDC_inline_ir) R __ir(string s, R, P...)(P);
double inlineIR(double a)
{
    auto s = __ir!(`ret double %0`, double)(a);
    return s;
}
