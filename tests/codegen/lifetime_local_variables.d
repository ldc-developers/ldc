// RUN: %ldc -femit-local-var-lifetime -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

extern(C): // disable mangling for easier matching

void opaque(byte* i);

// CHECK-LABEL: define void @foo_array_foo()
void foo_array_foo() {
    // CHECK: %arr = alloca [400 x i8]
    // CHECK: %arr1 = alloca [800 x i8]
    {
        // CHECK: call void @llvm.lifetime.start.p0{{(i8)?}}({{.*}} %arr)
        byte[400] arr = void;
        // CHECK: call void @opaque
        opaque(&arr[0]);
        // CHECK: call void @llvm.lifetime.end.p0{{(i8)?}}({{.*}} %arr)
    }

    {
        // CHECK: call void @llvm.lifetime.start.p0{{(i8)?}}({{.*}} %arr1)
        byte[800] arr = void;
        // CHECK: call void @opaque
        opaque(&arr[0]);
        // CHECK: call void @llvm.lifetime.end.p0{{(i8)?}}({{.*}} %arr1)
    }

    // CHECK-LABEL: ret void
}

// CHECK-LABEL: define void @foo_forloop_foo()
void foo_forloop_foo() {
    byte i;
    // CHECK: call void @opaque
    // This call should appear before lifetime start of while-loop variable.
    opaque(&i);
    for (byte[13] d; d[0] < 2; d[0]++) {
        // CHECK: call void @llvm.lifetime.start.p0{{(i8)?}}({{.*}} %d)
        // Lifetime should start before initializing the variable
        // CHECK: call void @llvm.memset.p0{{(i8)?}}.i{{.*}}13
        // CHECK: call void @llvm.lifetime.start.p0{{(i8)?}}({{.*}} %arr)
        byte[44] arr = void;
        // CHECK: call void @opaque
        opaque(&arr[0]);
        // CHECK: call void @llvm.lifetime.end.p0{{(i8)?}}({{.*}} %arr)
        // CHECK: endfor:
        // CHECK: call void @llvm.lifetime.end.p0{{(i8)?}}({{.*}} %d)
    }

    // CHECK-LABEL: ret void
}

// CHECK-LABEL: define void @foo_whileloop_foo()
void foo_whileloop_foo() {
    byte i;
    // CHECK: call void @opaque
    // This call should appear before lifetime start of while-loop variable.
    opaque(&i);
    while (ulong d = 131) {
        // CHECK: call void @llvm.lifetime.start.p0{{(i8)?}}({{.*}} %d)
        // Lifetime should start before initializing the variable
        // CHECK: store i64 131
        // CHECK: call void @llvm.lifetime.start.p0{{(i8)?}}({{.*}} %arr)
        byte[33] arr = void;
        // CHECK: call void @opaque
        opaque(&arr[0]);
        // CHECK: call void @llvm.lifetime.end.p0{{(i8)?}}({{.*}} %arr)
        // CHECK: call void @llvm.lifetime.end.p0{{(i8)?}}({{.*}} %d)
    }

    // CHECK-LABEL: ret void
}

// CHECK-LABEL: define void @foo_if_foo()
void foo_if_foo() {
    byte i;
    // CHECK: call void @opaque
    // This call should appear before lifetime start of if-statement condition variable.
    opaque(&i);
    // CHECK: call void @llvm.lifetime.start.p0{{(i8)?}}({{.*}} %d)
    // Lifetime should start before initializing the variable
    // CHECK: store i64 565
    if (ulong d = 565) {
        // CHECK: call void @llvm.lifetime.start.p0{{(i8)?}}({{.*}} %arr)
        byte[72] arr = void;
        // CHECK: call void @opaque
        opaque(&arr[0]);
        // CHECK: call void @llvm.lifetime.end.p0{{(i8)?}}({{.*}} %arr)
    } else {
        // d is out of scope here.
        // CHECK: call void @llvm.lifetime.start.p0{{(i8)?}}({{.*}} %arr1)
        byte[51] arr = void;
        // CHECK: call void @opaque
        opaque(&arr[0]);
        // CHECK: call void @llvm.lifetime.end.p0{{(i8)?}}({{.*}} %arr1)
    }
    // CHECK: endif:
    // CHECK: call void @llvm.lifetime.end.p0{{(i8)?}}({{.*}} %d)

    // CHECK-LABEL: ret void
}

struct S {
    byte[123] a;
    ~this() {
        opaque(&a[1]);
    }
}

void opaque_S(S* i);

// CHECK-LABEL: define void @foo_struct_foo()
void foo_struct_foo() {
    {
        // CHECK: call void @llvm.lifetime.start.p0{{(i8)?}}({{.*}} %s)
        S s;
        // CHECK: invoke void @opaque_S
        opaque_S(&s);
    }

    // CHECK: call void @llvm.lifetime.end.p0{{(i8)?}}({{.*}} %s)
    // CHECK-NEXT: ret void
}
