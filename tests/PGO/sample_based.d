// Test basic use of sample-based PGO profile

// REQUIRES: atleast_llvm1500

// RUN: split-file %s %t
// RUN: %ldc -O2 -c -gline-tables-only -output-ll -of=%t.ll -fprofile-sample-use=%t/pgo-sample.prof %t/testcase.d && FileCheck %s < %t.ll

//--- pgo-sample.prof
foo:100:100
 1: 100

//--- testcase.d
// CHECK: define{{.*}} @foo{{.*}} #[[ATTRID:[0-9]+]]{{.*}} !prof ![[PROFID:[0-9]+]]
// CHECK: attributes #[[ATTRID]] = {{.*}} "use-sample-profile"
// CHECK-DAG: "ProfileFormat", !"SampleProfile"
// CHECK-DAG: "TotalCount", i64 100
// CHECK-DAG: ![[PROFID]] = !{!"function_entry_count", i64 101}

extern (C) int foo () {
    return 1;
}
