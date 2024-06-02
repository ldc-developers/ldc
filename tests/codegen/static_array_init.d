// RUN: %ldc -output-ll %s -of=%t.ll
// RUN: FileCheck %s < %t.ll

void bytes_scalar()
{
    immutable(byte)[32] myBytes = 123;

    // CHECK:      define {{.*}}_D17static_array_init12bytes_scalarFZv
    // CHECK-NEXT:   %myBytes = alloca [32 x i8], align 1
    // CHECK:   call void @llvm.memset{{.*}}(ptr{{[a-z0-9 ]*}} {{%.*}}, i8 123, i{{(32|64)}} 32
}

void bytes_scalar(byte arg)
{
    immutable(byte)[32] myBytes = arg;

    // CHECK:      define {{.*}}_D17static_array_init12bytes_scalarFgZv
    // CHECK-NEXT:   %arg = alloca i8, align 1
    // CHECK-NEXT:   %myBytes = alloca [32 x i8], align 1
    // CHECK-NEXT:   store i8 %arg_arg, ptr %arg
    // CHECK:   {{%.}} = load {{.*}}ptr %arg
    // CHECK-NEXT:   call void @llvm.memset{{.*}}(ptr{{[a-z0-9 ]*}} {{%.*}}, i8 {{%.}}, i{{(32|64)}} 32
}

void ints_scalar()
{
    const(int[32]) myInts = 123;

    // CHECK:      define {{.*}}_D17static_array_init11ints_scalarFZv
    // CHECK:      arrayinit.cond:
    // CHECK-NEXT:   %[[I1:[0-9]+]] = load {{.*}}ptr %arrayinit.itr
    // CHECK-NEXT:   %arrayinit.condition = icmp ne i{{(32|64)}} %[[I1]], 32
    // CHECK:        store i32 123, ptr %arrayinit.arrayelem
}

void ints_scalar(int arg)
{
    const(int[32]) myInts = arg;

    // CHECK:      define {{.*}}_D17static_array_init11ints_scalarFiZv
    // CHECK:      arrayinit.cond:
    // CHECK-NEXT:   %[[I2:[0-9]+]] = load {{.*}}ptr %arrayinit.itr
    // CHECK-NEXT:   %arrayinit.condition = icmp ne i{{(32|64)}} %[[I2]], 32
    // CHECK:        %[[E2:[0-9]+]] = load {{.*}}ptr %arg
    // CHECK-NEXT:   store i32 %[[E2]], ptr %arrayinit.arrayelem
}

void bytes()
{
    immutable(byte[4]) myBytes = [ 1, 2, 3, 4 ];

    // CHECK:      define {{.*}}_D17static_array_init5bytesFZv
    // CHECK-NEXT:   %myBytes = alloca [4 x i8], align 1
    // CHECK-NEXT:   store [4 x i8] c"\01\02\03\04", ptr %myBytes
}

void bytes(byte[] arg)
{
    const(byte)[4] myBytes = arg;

    // CHECK:      define {{.*}}_D17static_array_init5bytesFAgZv
    // CHECK:        %myBytes = alloca [4 x i8], align 1
    // CHECK:        call void @llvm.memcpy{{.*}}(ptr{{[a-z0-9 ]*}} %{{.*}}, ptr{{[a-z0-9 ]*}} %.ptr, i{{(32|64)}} 4
}

void ints()
{
    immutable(int)[4] myInts = [ 1, 2, 3, 4 ];

    // CHECK:      define {{.*}}_D17static_array_init4intsFZv
    // CHECK-NEXT:   %myInts = alloca [4 x i32], align 4
    // CHECK-NEXT:   store [4 x i32] [i32 1, i32 2, i32 3, i32 4], ptr %myInts
}

void ints(ref int[4] arg)
{
    const(int[4]) myInts = arg;

    // CHECK:      define {{.*}}_D17static_array_init4intsFKG4iZv
    // CHECK-NEXT:   %myInts = alloca [4 x i32], align 4
    // CHECK:   call void @llvm.memcpy{{.*}}(ptr{{[a-z0-9 ]*}} %{{.*}}, ptr{{[a-z0-9 ]*}} %{{.*}}, i{{(32|64)}} 16
}

void bytes_scalar_2d()
{
    immutable(byte)[4][8] myBytes = 123;

    // CHECK:      define {{.*}}_D17static_array_init15bytes_scalar_2dFZv
    // CHECK-NEXT:   %myBytes = alloca [8 x [4 x i8]], align 1
    // CHECK:   call void @llvm.memset{{.*}}(ptr{{[a-z0-9 ]*}} %{{.*}}, i8 123, i{{(32|64)}} 32
}

void ints_scalar_2d(immutable int arg)
{
    const(int[4])[8] myInts = arg;

    // CHECK:      define {{.*}}_D17static_array_init14ints_scalar_2dFyiZv
    // CHECK:      arrayinit.cond:
    // CHECK-NEXT:   %[[I3:[0-9]+]] = load {{.*}}ptr %arrayinit.itr
    // CHECK-NEXT:   %arrayinit.condition = icmp ne i{{(32|64)}} %[[I3]], 32
    // CHECK:        %[[E3:[0-9]+]] = load {{.*}}ptr %arg
    // CHECK-NEXT:   store i32 %[[E3]], ptr %arrayinit.arrayelem
}
