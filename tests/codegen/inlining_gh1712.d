// Test inlining related to Github issue 1712

// REQUIRES: atleast_llvm307

// RUN: %ldc %s -c -output-ll -O3 -of=%t.ll && FileCheck %s < %t.ll
// RUN: %ldc -O3 -run %s

module mod;

void test9785_2()
{
    int j = 3;

    void loop(scope const void function(int x) dg)
    {
        pragma(inline, true);
        dg(++j);
    }

    static void func(int x)
    {
        pragma(inline, true);
        assert(x == 4);
    }

    loop(&func);
}

void main()
{
    test9785_2();
}

// There was a bug where pragma(inline, true) nested functions were incorrectly emitted with `available_externally` linkage.

// CHECK: define
// CHECK-NOT: available_externally
// CHECK-SAME: @{{.*}}D3mod10test9785_2FZv

// CHECK: define
// CHECK-NOT: available_externally
// CHECK-SAME: @{{.*}}D3mod10test9785_2FZ4loopMFMxPFiZvZv

// CHECK: define
// CHECK-NOT: available_externally
// CHECK-SAME: @{{.*}}D3mod10test9785_2FZ4funcFiZv
