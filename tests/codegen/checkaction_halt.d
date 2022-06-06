// RUN: %ldc -checkaction=halt -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK: define {{.*}}_D16checkaction_halt3foo
void foo(int x)
{
    assert(x, "msg");

    // CHECK:      assertFailed:
    // CHECK-NEXT:   call void @llvm.trap()
    // CHECK-NEXT:   unreachable
}
