// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

extern (C):

int tlsVar;
extern int externTlsVar;
__gshared int gsharedVar;

// CHECK-LABEL: define{{.*}} @readTls(
int readTls()
{
    // CHECK: call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@tlsVar)
    return tlsVar;
}

// CHECK-LABEL: define{{.*}} @writeTls(
void writeTls(int v)
{
    // CHECK: call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@externTlsVar)
    externTlsVar = v;
}

// CHECK-LABEL: define{{.*}} @addressTls(
int* addressTls()
{
    // CHECK: [[ADDR:%[0-9]+]] = call {{.*}}ptr @llvm.threadlocal.address.p0(ptr {{.*}}@tlsVar)
    // CHECK: ret ptr [[ADDR]]
    return &tlsVar;
}

// CHECK-LABEL: define{{.*}} @readGshared(
int readGshared()
{
    // CHECK-NOT: llvm.threadlocal.address
    // CHECK: load i32, ptr @gsharedVar
    return gsharedVar;
}
