// RUN: %ldc -output-ll %s -of=%t.ll
// RUN: FileCheck %s < %t.ll

void bytes_scalar()
{
    immutable(byte)[32] myBytes = 123;

    // CHECK:      define {{.*}}_D17static_array_init12bytes_scalarFZv
    // CHECK-NEXT:   %myBytes = alloca [32 x i8], align 1
    // CHECK-NEXT:   %1 = bitcast [32 x i8]* %myBytes to i8*
    // CHECK-NEXT:   call void @llvm.memset{{.*}}(i8*{{[a-z0-9 ]*}} %1, i8 123, i{{(32|64)}} 32
}

void bytes_scalar(byte arg)
{
    immutable(byte)[32] myBytes = arg;

    // CHECK:      define {{.*}}_D17static_array_init12bytes_scalarFgZv
    // CHECK-NEXT:   %arg = alloca i8, align 1
    // CHECK-NEXT:   %myBytes = alloca [32 x i8], align 1
    // CHECK-NEXT:   store i8 %arg_arg, i8* %arg
    // CHECK-NEXT:   %1 = bitcast [32 x i8]* %myBytes to i8*
    // CHECK-NEXT:   %2 = load {{.*}}i8* %arg
    // CHECK-NEXT:   call void @llvm.memset{{.*}}(i8*{{[a-z0-9 ]*}} %1, i8 %2, i{{(32|64)}} 32
}

void ints_scalar()
{
    const(int[32]) myInts = 123;

    // CHECK:      define {{.*}}_D17static_array_init11ints_scalarFZv
    // CHECK:      arrayinit.cond:
    // CHECK-NEXT:   %[[I1:[0-9]+]] = load {{.*i(32|64)}}* %arrayinit.itr
    // CHECK-NEXT:   %arrayinit.condition = icmp ne i{{(32|64)}} %[[I1]], 32
    // CHECK:        store i32 123, i32* %arrayinit.arrayelem
}

void ints_scalar(int arg)
{
    const(int[32]) myInts = arg;

    // CHECK:      define {{.*}}_D17static_array_init11ints_scalarFiZv
    // CHECK:      arrayinit.cond:
    // CHECK-NEXT:   %[[I2:[0-9]+]] = load {{.*i(32|64)}}* %arrayinit.itr
    // CHECK-NEXT:   %arrayinit.condition = icmp ne i{{(32|64)}} %[[I2]], 32
    // CHECK:        %[[E2:[0-9]+]] = load {{.*}}i32* %arg
    // CHECK-NEXT:   store i32 %[[E2]], i32* %arrayinit.arrayelem
}

void bytes()
{
    immutable(byte[4]) myBytes = [ 1, 2, 3, 4 ];

    // CHECK:      define {{.*}}_D17static_array_init5bytesFZv
    // CHECK-NEXT:   %myBytes = alloca [4 x i8], align 1
    // CHECK-NEXT:   store [4 x i8] c"\01\02\03\04", [4 x i8]* %myBytes
}

void bytes(byte[] arg)
{
    const(byte)[4] myBytes = arg;

    // CHECK:      define {{.*}}_D17static_array_init5bytesFAgZv
    // CHECK:        %myBytes = alloca [4 x i8], align 1
    // CHECK:        %1 = bitcast [4 x i8]* %myBytes to i8*
    // CHECK:        call void @llvm.memcpy{{.*}}(i8*{{[a-z0-9 ]*}} %1, i8*{{[a-z0-9 ]*}} %.ptr, i{{(32|64)}} 4
}

void ints()
{
    immutable(int)[4] myInts = [ 1, 2, 3, 4 ];

    // CHECK:      define {{.*}}_D17static_array_init4intsFZv
    // CHECK-NEXT:   %myInts = alloca [4 x i32], align 4
    // CHECK-NEXT:   store [4 x i32] [i32 1, i32 2, i32 3, i32 4], [4 x i32]* %myInts
}

void ints(ref int[4] arg)
{
    const(int[4]) myInts = arg;

    // CHECK:      define {{.*}}_D17static_array_init4intsFKG4iZv
    // CHECK-NEXT:   %myInts = alloca [4 x i32], align 4
    // CHECK-NEXT:   %1 = bitcast [4 x i32]* %myInts to i32*
    // CHECK-NEXT:   %2 = bitcast i32* %1 to i8*
    // CHECK-NEXT:   %3 = bitcast [4 x i32]* %arg to i32*
    // CHECK-NEXT:   %4 = bitcast i32* %3 to i8*
    // CHECK-NEXT:   call void @llvm.memcpy{{.*}}(i8*{{[a-z0-9 ]*}} %2, i8*{{[a-z0-9 ]*}} %4, i{{(32|64)}} 16
}

void bytes_scalar_2d()
{
    immutable(byte)[4][8] myBytes = 123;

    // CHECK:      define {{.*}}_D17static_array_init15bytes_scalar_2dFZv
    // CHECK-NEXT:   %myBytes = alloca [8 x [4 x i8]], align 1
    // CHECK-NEXT:   %1 = bitcast [8 x [4 x i8]]* %myBytes to [32 x i8]*
    // CHECK-NEXT:   %2 = bitcast [32 x i8]* %1 to i8*
    // CHECK-NEXT:   call void @llvm.memset{{.*}}(i8*{{[a-z0-9 ]*}} %2, i8 123, i{{(32|64)}} 32
}

void ints_scalar_2d(immutable int arg)
{
    const(int[4])[8] myInts = arg;

    // CHECK:      define {{.*}}_D17static_array_init14ints_scalar_2dFyiZv
    // CHECK:      arrayinit.cond:
    // CHECK-NEXT:   %[[I3:[0-9]+]] = load {{.*i(32|64)}}* %arrayinit.itr
    // CHECK-NEXT:   %arrayinit.condition = icmp ne i{{(32|64)}} %[[I3]], 32
    // CHECK:        %[[E3:[0-9]+]] = load {{.*}}i32* %arg
    // CHECK-NEXT:   store i32 %[[E3]], i32* %arrayinit.arrayelem
}
