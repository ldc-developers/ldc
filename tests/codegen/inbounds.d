// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

struct S {
    float x, y;
};

extern(C):  // Avoid name mangling

// IndexExp in array with const exp
// CHECK-LABEL: @foo1
int foo1(int[3] a) {
    // CHECK: getelementptr inbounds [3 x i32]
    return a[1];
}

// IndexExp in pointer
// CHECK-LABEL: @foo2
int foo2(int* p, int i) {
    // CHECK: getelementptr inbounds
    return p[i];
}

// PostExp in pointer
// CHECK-LABEL: @foo3
int foo3(int* p) {
    // CHECK: getelementptr inbounds
    return *p++;
}

// PreExp in pointer
// CHECK-LABEL: @foo4
int foo4(int* p) {
    // CHECK: getelementptr inbounds
    return *++p;
}

// Add offset to pointer
// CHECK-LABEL: @foo5
int foo5(int* p, int i) {
    // CHECK: getelementptr inbounds
    return *(p + i);
}

// Subtract offset from pointer
// CHECK-LABEL: @foo6
int foo6(int* p, int i) {
    // CHECK: getelementptr inbounds
    return *(p - i);
}

// Struct field
// CHECK-LABEL: @foo7
float foo7(S s) {
    // CHECK: getelementptr inbounds
    return s.y;
}
