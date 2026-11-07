// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll
// RUN: %ldc -c -output-ll -O2 -of=%t.opt.ll %s && FileCheck %s --check-prefix=OPT < %t.opt.ll

extern (C):

int tlsVar;

// CHECK-LABEL: define{{.*}} @foo(
// OPT-LABEL: define{{.*}} @foo(
int foo()
{
    // CHECK: call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@tlsVar)
    // CHECK: call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@tlsVar)
    // CHECK-NOT: llvm.threadlocal.address
    // CHECK: ret i32
    // OPT: call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@tlsVar)
    // OPT-NOT: llvm.threadlocal.address
    // OPT: ret i32
    tlsVar++;
    return tlsVar;
}
