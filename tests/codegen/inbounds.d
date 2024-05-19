// RUN: %ldc -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

struct S {
    float x, y;
};

extern(C):  // Avoid name mangling

// IndexExp in static array with const exp
// CHECK-LABEL: @foo1
int foo1(int[3] a) {
    // CHECK: getelementptr inbounds [3 x i32]
    return a[1];
}

// IndexExp in static array with variable exp
// CHECK-LABEL: @foo2
int foo2(int[3] a, int i) {
    // CHECK: getelementptr inbounds [3 x i32]
    return a[i];
}

// IndexExp in pointer
// CHECK-LABEL: @foo3
int foo3(int* p, int i) {
    // CHECK: getelementptr inbounds
    return p[i];
}

// PostExp in pointer
// CHECK-LABEL: @foo4
int foo4(int* p) {
    // CHECK: getelementptr inbounds
    return *p++;
}

// PreExp in pointer
// CHECK-LABEL: @foo5
int foo5(int* p) {
    // CHECK: getelementptr inbounds
    return *++p;
}

// Add offset to pointer
// CHECK-LABEL: @foo6
int foo6(int* p, int i) {
    // CHECK: getelementptr inbounds
    return *(p + i);
}

// Subtract offset from pointer
// CHECK-LABEL: @foo7
int foo7(int* p, int i) {
    // CHECK: getelementptr inbounds
    return *(p - i);
}

// Struct field
// CHECK-LABEL: @foo8
float foo8(S s) {
    // CHECK: getelementptr inbounds
    return s.y;
}

// IndexExp in dynamic array with const exp
// CHECK-LABEL: @foo9
int foo9(int[] a) {
    // CHECK: getelementptr inbounds i32, ptr
    return a[1];
}

// IndexExp in dynamic array with variable exp
// CHECK-LABEL: @foo10
int foo10(int[] a, int i) {
    // CHECK: getelementptr inbounds i32, ptr
    return a[i];
}

// SliceExp for static array with const lower bound
// CHECK-LABEL: @foo11
int[] foo11(ref int[3] a) {
    // CHECK: getelementptr inbounds i32, ptr
    return a[1 .. $];
}

// SliceExp for dynamic array with variable lower bound
// CHECK-LABEL: @foo12
int[] foo12(int[] a, int i) {
    // CHECK: getelementptr inbounds i32, ptr
    return a[i .. $];
}
