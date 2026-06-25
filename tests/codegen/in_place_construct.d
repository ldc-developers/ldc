// Tests in-place construction of variables.

// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// 256 bits, returned via sret:
struct S
{
    long a, b, c, d;
}

// CHECK-LABEL: define{{.*}} @{{.*}}_D18in_place_construct13returnLiteralFZSQBm1S
S returnLiteral()
{
    // make sure the literal is emitted directly into the sret pointee
    // CHECK: %1 = getelementptr inbounds {{.*}}%in_place_construct.S, ptr %.sret_arg, i32 0, i32 0
    // CHECK: store i64 1, ptr %1
    // CHECK: %2 = getelementptr inbounds {{.*}}%in_place_construct.S, ptr %.sret_arg, i32 0, i32 1
    // CHECK: store i64 2, ptr %2
    // CHECK: %3 = getelementptr inbounds {{.*}}%in_place_construct.S, ptr %.sret_arg, i32 0, i32 2
    // CHECK: store i64 3, ptr %3
    // CHECK: %4 = getelementptr inbounds {{.*}}%in_place_construct.S, ptr %.sret_arg, i32 0, i32 3
    // CHECK: store i64 4, ptr %4
    return S(1, 2, 3, 4);
}

// CHECK-LABEL: define{{.*}} @{{.*}}_D18in_place_construct12returnRValueFZSQBl1S
S returnRValue()
{
    // make sure the sret pointer is forwarded
    // CHECK: call {{.*}}_D18in_place_construct13returnLiteralFZSQBm1S
    // CHECK-SAME: ptr {{.*}} %.sret_arg
    return returnLiteral();
}

// CHECK-LABEL: define{{.*}} @{{.*}}_D18in_place_construct10returnNRVOFZSQBj1S
S returnNRVO()
{
    // make sure NRVO zero-initializes the sret pointee directly
    // CHECK: call void @llvm.memset.{{.*}}(ptr{{.*}} %.sret_arg, i8 0,
    const S r;
    return r;
}

// CHECK-LABEL: define{{.*}} @{{.*}}_D18in_place_construct19RVO_withOutContractFZSQBs1S
S RVO_withOutContract()
out { assert(__result.c == 3); }
do
{
    // make sure the literal is emitted directly into the sret pointee
    // CHECK-NEXT: %1 = getelementptr inbounds {{.*}}%in_place_construct.S, ptr %.sret_arg, i32 0, i32 0
    // CHECK-NEXT: store i64 1, ptr %1
    // CHECK-NEXT: %2 = getelementptr inbounds {{.*}}%in_place_construct.S, ptr %.sret_arg, i32 0, i32 1
    // CHECK-NEXT: store i64 2, ptr %2
    // CHECK-NEXT: %3 = getelementptr inbounds {{.*}}%in_place_construct.S, ptr %.sret_arg, i32 0, i32 2
    // CHECK-NEXT: store i64 3, ptr %3
    // CHECK-NEXT: %4 = getelementptr inbounds {{.*}}%in_place_construct.S, ptr %.sret_arg, i32 0, i32 3
    // CHECK-NEXT: store i64 4, ptr %4
    return S(1, 2, 3, 4);

    // make sure `__result` inside the out contract is just an alias to the sret pointee
    // CHECK:      %5 = getelementptr inbounds {{.*}}%in_place_construct.S, ptr %.sret_arg, i32 0, i32 2
    // CHECK-NEXT: %6 = load {{.*}}ptr %5
    // CHECK-NEXT: icmp eq i64 %6, 3
}

// CHECK-LABEL: define{{.*}} @{{.*}}_D18in_place_construct20NRVO_withOutContractFZSQBt1S
S NRVO_withOutContract()
out { assert(__result.a == 0); }
do
{
    // __result is a ref
    // CHECK-NEXT: %result = alloca ptr

    // make sure NRVO zero-initializes the sret pointee directly
    // CHECK-NEXT: call void @llvm.memset.{{.*}}(ptr{{.*}} %.sret_arg, i8 0,
    const S r;
    return r;

    // make sure the __result ref is initialized with the sret pointer
    // CHECK-NEXT: store ptr %.sret_arg, ptr %result

    // CHECK:      %1 = load ptr, ptr %result
    // CHECK-NEXT: %2 = getelementptr inbounds {{.*}}%in_place_construct.S, ptr %1, i32 0, i32 0
    // CHECK-NEXT: %3 = load {{.*}}ptr %2
    // CHECK-NEXT: icmp eq i64 %3, 0
}

// CHECK-LABEL: define{{.*}} @{{.*}}_D18in_place_construct7structsFZv
void structs()
{
    // CHECK: %literal = alloca %in_place_construct.S
    // CHECK: %a = alloca %in_place_construct.S
    // CHECK: %b = alloca %in_place_construct.S
    // CHECK: %c = alloca %in_place_construct.S

    // make sure the literal is emitted directly into the lvalue
    // CHECK: %1 = getelementptr inbounds {{.*}}%in_place_construct.S, ptr %literal, i32 0, i32 0
    // CHECK: store i64 5, ptr %1
    // CHECK: %2 = getelementptr inbounds {{.*}}%in_place_construct.S, ptr %literal, i32 0, i32 1
    // CHECK: store i64 6,  ptr %2
    // CHECK: %3 = getelementptr inbounds {{.*}}%in_place_construct.S, ptr %literal, i32 0, i32 2
    // CHECK: store i64 7,  ptr %3
    // CHECK: %4 = getelementptr inbounds {{.*}}%in_place_construct.S, ptr %literal, i32 0, i32 3
    // CHECK: store i64 8,  ptr %4
    const literal = S(5, 6, 7, 8);

    // make sure the variables are in-place constructed via sret
    // CHECK: call {{.*}}_D18in_place_construct13returnLiteralFZSQBm1S
    // CHECK-SAME: ptr{{.*}} %a
    const a = returnLiteral();
    // CHECK: call {{.*}}_D18in_place_construct12returnRValueFZSQBl1S
    // CHECK-SAME: ptr{{.*}} %b
    const b = returnRValue();
    // CHECK: call {{.*}}_D18in_place_construct10returnNRVOFZSQBj1S
    // CHECK-SAME: ptr{{.*}} %c
    const c = returnNRVO();

    RVO_withOutContract();
    NRVO_withOutContract();
}

// CHECK-LABEL: define{{.*}} @{{.*}}_D18in_place_construct12staticArraysFZv
void staticArrays()
{
    // CHECK: %sa = alloca [2 x i32]

    // make sure static array literals are in-place constructed too
    // CHECK: store [2 x i32] [i32 1, i32 2], ptr %sa
    const(int[2]) sa = [ 1, 2 ];
}

struct Container { S s; }

// CHECK-LABEL: define{{.*}} @{{.*}}_D18in_place_construct19hierarchyOfLiteralsFZv
void hierarchyOfLiterals()
{
    // CHECK: %sa = alloca [1 x %in_place_construct.Container]
    // CHECK: store [1 x %in_place_construct.Container] [%in_place_construct.Container { %in_place_construct.S { i64 11, i64 12, i64 13, i64 14 } }], ptr %sa
    Container[1] sa = [ Container(S(11, 12, 13, 14)) ];
}

// CHECK-LABEL: define{{.*}} @{{.*}}_Dmain
void main()
{
    structs();
    staticArrays();
    hierarchyOfLiterals();
}
