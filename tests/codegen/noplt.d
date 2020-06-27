// RUN: %ldc --output-ll --fno-plt -of=%t.ll %s && FileCheck %s -check-prefix=CHECK-NOPLT < %t.ll

// CHECK-NOPLT: Function Attrs: nonlazybind
// CHECK-NOPLT-NEXT: declare {{.*}} @{{.*}}3foo
int foo();

int bar() {
    return foo();
}
