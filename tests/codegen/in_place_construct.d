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
    // CHECK: %1 = getelementptr inbounds {{.*}}%in_place_construct.S{{\*|, ptr}} %.sret_arg, i32 0, i32 0
    // CHECK: store i64 1, {{i64\*|ptr}} %1
    // CHECK: %2 = getelementptr inbounds {{.*}}%in_place_construct.S{{\*|, ptr}} %.sret_arg, i32 0, i32 1
    // CHECK: store i64 2, {{i64\*|ptr}} %2
    // CHECK: %3 = getelementptr inbounds {{.*}}%in_place_construct.S{{\*|, ptr}} %.sret_arg, i32 0, i32 2
    // CHECK: store i64 3, {{i64\*|ptr}} %3
    // CHECK: %4 = getelementptr inbounds {{.*}}%in_place_construct.S{{\*|, ptr}} %.sret_arg, i32 0, i32 3
    // CHECK: store i64 4, {{i64\*|ptr}} %4
    return S(1, 2, 3, 4);
}

// CHECK-LABEL: define{{.*}} @{{.*}}_D18in_place_construct12returnRValueFZSQBl1S
S returnRValue()
{
    // make sure the sret pointer is forwarded
    // CHECK: call {{.*}}_D18in_place_construct13returnLiteralFZSQBm1S
    // CHECK-SAME: {{%in_place_construct.S\*|ptr}} {{.*}} %.sret_arg
    return returnLiteral();
}

// CHECK-LABEL: define{{.*}} @{{.*}}_D18in_place_construct10returnNRVOFZSQBj1S
S returnNRVO()
{
    // make sure NRVO zero-initializes the sret pointee directly
    // CHECK: call void @llvm.memset.{{.*}}({{i8\*|ptr}}{{.*}}, i8 0,
    const S r;
    return r;
}

// CHECK-LABEL: define{{.*}} @{{.*}}_D18in_place_construct15withOutContractFZSQBo1S
S withOutContract()
out { assert(__result.a == 0); }
do
{
    // make sure NRVO zero-initializes the sret pointee directly
    // CHECK: call void @llvm.memset.{{.*}}({{i8\*|ptr}}{{.*}}, i8 0,
    const S r;
    return r;

    // make sure `__result` inside the out contract is just an alias to the sret pointee
    // CHECK: %{{1|2}} = getelementptr inbounds {{.*}}%in_place_construct.S{{\*|, ptr}} %.sret_arg, i32 0, i32 0
    // CHECK: %{{2|3}} = load {{.*}}{{i64\*|ptr}} %{{1|2}}
    // CHECK: %{{3|4}} = icmp eq i64 %{{2|3}}, 0
}

// CHECK-LABEL: define{{.*}} @{{.*}}_D18in_place_construct7structsFZv
void structs()
{
    // CHECK: %literal = alloca %in_place_construct.S
    // CHECK: %a = alloca %in_place_construct.S
    // CHECK: %b = alloca %in_place_construct.S
    // CHECK: %c = alloca %in_place_construct.S

    // make sure the literal is emitted directly into the lvalue
    // CHECK: %1 = getelementptr inbounds {{.*}}%in_place_construct.S{{\*|, ptr}} %literal, i32 0, i32 0
    // CHECK: store i64 5, {{i64\*|ptr}} %1
    // CHECK: %2 = getelementptr inbounds {{.*}}%in_place_construct.S{{\*|, ptr}} %literal, i32 0, i32 1
    // CHECK: store i64 6,  {{i64\*|ptr}} %2
    // CHECK: %3 = getelementptr inbounds {{.*}}%in_place_construct.S{{\*|, ptr}} %literal, i32 0, i32 2
    // CHECK: store i64 7,  {{i64\*|ptr}} %3
    // CHECK: %4 = getelementptr inbounds {{.*}}%in_place_construct.S{{\*|, ptr}} %literal, i32 0, i32 3
    // CHECK: store i64 8,  {{i64\*|ptr}} %4
    const literal = S(5, 6, 7, 8);

    // make sure the variables are in-place constructed via sret
    // CHECK: call {{.*}}_D18in_place_construct13returnLiteralFZSQBm1S
    // CHECK-SAME: {{(%in_place_construct\.S\*|ptr).*}} %a
    const a = returnLiteral();
    // CHECK: call {{.*}}_D18in_place_construct12returnRValueFZSQBl1S
    // CHECK-SAME: {{(%in_place_construct\.S\*|ptr).*}} %b
    const b = returnRValue();
    // CHECK: call {{.*}}_D18in_place_construct10returnNRVOFZSQBj1S
    // CHECK-SAME: {{(%in_place_construct\.S\*|ptr).*}} %c
    const c = returnNRVO();

    withOutContract();
}

// CHECK-LABEL: define{{.*}} @{{.*}}_D18in_place_construct12staticArraysFZv
void staticArrays()
{
    // CHECK: %sa = alloca [2 x i32]

    // make sure static array literals are in-place constructed too
    // CHECK: store [2 x i32] [i32 1, i32 2], {{\[2 x i32\]\*|ptr}} %sa
    const(int[2]) sa = [ 1, 2 ];
}

struct Container { S s; }

// CHECK-LABEL: define{{.*}} @{{.*}}_D18in_place_construct19hierarchyOfLiteralsFZv
void hierarchyOfLiterals()
{
    // CHECK: %sa = alloca [1 x %in_place_construct.Container]
    // CHECK: store [1 x %in_place_construct.Container] [%in_place_construct.Container { %in_place_construct.S { i64 11, i64 12, i64 13, i64 14 } }], {{\[1 x %in_place_construct.Container\]\*|ptr}} %sa
    Container[1] sa = [ Container(S(11, 12, 13, 14)) ];
}

// CHECK-LABEL: define{{.*}} @{{.*}}_Dmain
void main()
{
    structs();
    staticArrays();
    hierarchyOfLiterals();
}
