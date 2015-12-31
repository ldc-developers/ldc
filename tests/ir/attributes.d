// Tests LDC-specific attributes

// RUN: %ldc -O -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import ldc.attributes;

//---- @(section) -----------------------------------------------------

// CHECK-DAG: @{{.*}}mySectionedGlobali ={{.*}} section ".mySection"
@(section(".mySection")) int mySectionedGlobal;

// TODO: Specifying section for functions is not implemented yet
// TODO: C HECK-DAG: .... section "funcSection"
@(section("funcSection")) void foo() {}

//---------------------------------------------------------------------


// CHECK-LABEL: define i32 @_Dmain
void main() {
  foo();
}
