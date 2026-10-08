// RUN: %ldc --output-ll --fno-plt -of=%t.ll %s && FileCheck %s -check-prefixes=CHECK-NOPLT,CHECK-NOPLT-METADATA < %t.ll

// CHECK-NOPLT: Function Attrs: nonlazybind
// CHECK-NOPLT-NEXT: declare {{.*}} @{{.*}}3foo
// CHECK-NOPLT-METADATA: !"RtLibUseGOT"
int foo();

int bar() {
    return foo();
}
