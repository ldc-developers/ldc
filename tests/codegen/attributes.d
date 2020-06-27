// Tests LDC-specific attributes

// RUN: %ldc -O -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import ldc.attributes;

//---- @(section) -----------------------------------------------------

// CHECK-DAG: @{{.*}}mySectionedGlobali{{.*}} section ".mySection"
@(section(".mySection")) int mySectionedGlobal;

// CHECK-DAG: define{{.*}} void @{{.*}}sectionedfoo{{.*}} section "funcSection"
@(section("funcSection")) void sectionedfoo() {}

//---------------------------------------------------------------------

//---------------------------------------------------------------------
//---- @(weak) --------------------------------------------------------

// CHECK-DAG: @{{.*}}myWeakGlobali{{\"?}} = weak
@(ldc.attributes.weak) int myWeakGlobal;

// CHECK-DAG: define{{.*}} {{(weak .*void @.*_D)|(void @.*_D6__weak)}}10attributes8weakFuncFZv
@weak void weakFunc() {}

//---------------------------------------------------------------------


// CHECK-LABEL: define{{.*}} i32 @_Dmain
void main() {
  sectionedfoo();
}
