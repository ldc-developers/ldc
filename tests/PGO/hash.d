// A non-exhaustive test to check that old profile information is only partially
// applied when code has changed.
// The code changes are simulated by version(.) blocks.

// REQUIRES: PGO_RT

// RUN: %ldc -d-version=ProfData -fprofile-instr-generate=%t.profraw -run %s  \
// RUN:   &&  %profdata merge %t.profraw -o %t.profdata \
// RUN:   &&  %ldc -d-version=ProfData -c -output-ll -of=%t2.ll -fprofile-instr-use=%t.profdata %s \
// RUN:   &&  FileCheck -allow-deprecated-dag-overlap %s -check-prefix=PROFDATA < %t2.ll \
// RUN:   &&  %ldc -wi -c -output-ll -of=%t3.ll -fprofile-instr-use=%t.profdata %s \
// RUN:   &&  FileCheck -allow-deprecated-dag-overlap %s -check-prefix=NODATA < %t3.ll

extern(C):

// PROFDATA-LABEL: define {{.*}} @{{[A-Za-z0-9_]*}}same{{[A-Za-z0-9]*}}(
// PROFDATA-SAME: !prof ![[SAME0:[0-9]+]]
// NODATA-LABEL: define {{.*}} @{{[A-Za-z0-9_]*}}same{{[A-Za-z0-9]*}}(
// NODATA-SAME: !prof ![[SAME0:[0-9]+]]
void same(int i) {
  // PROFDATA: br {{.*}} !prof ![[SAME1:[0-9]+]]
  // NODATA: br {{.*}} !prof ![[SAME1:[0-9]+]]
  if (i % 3) {}
}


// PROFDATA-LABEL: define {{.*}} @{{[A-Za-z0-9_]*}}undetectedchange{{[A-Za-z0-9]*}}(
// PROFDATA-SAME: !prof ![[UNDTCTD0:[0-9]+]]
// NODATA-LABEL: define {{.*}} @{{[A-Za-z0-9_]*}}undetectedchange{{[A-Za-z0-9]*}}(
// NODATA-SAME: !prof ![[UNDTCTD0:[0-9]+]]
void undetectedchange(int i, int i2) {
  // PROFDATA: br {{.*}} !prof ![[UNDTCTD1:[0-9]+]]
  // NODATA: br {{.*}} !prof ![[UNDTCTD1:[0-9]+]]
version(ProfData) {
  if (i % 3) {}
} else {
  if (i2 % 2) {}
}
}

// PROFDATA-LABEL: define {{.*}} @{{[A-Za-z0-9_]*}}changedhash{{[A-Za-z0-9]*}}(
// PROFDATA-SAME: !prof ![[DIFF0:[0-9]+]]
// NODATA-LABEL: define {{.*}} @{{[A-Za-z0-9_]*}}changedhash{{[A-Za-z0-9]*}}(
// NODATA-NOT: !prof
void changedhash(int i, int i2) {
  // PROFDATA: br {{.*}} !prof ![[DIFF1:[0-9]+]]
version(ProfData) {
  if (i % 3) {}
} else {
  while (i++ < i2) { }
}
}

// PROFDATA-LABEL: define {{.*}} @_Dmain(
// NODATA-LABEL: define {{.*}} @_Dmain(
extern(D) void main() {
  foreach (int i; 0..10) {
    same(i);
    undetectedchange(i, i+2);
    changedhash(i, i+2);
  }
}

// PROFDATA-DAG: ![[SAME0]] = !{!"function_entry_count", i64 10}
// PROFDATA-DAG: ![[SAME1]] = !{!"branch_weights", i32 7, i32 5}
// NODATA-DAG: ![[SAME0]] = !{!"function_entry_count", i64 10}
// NODATA-DAG: ![[SAME1]] = !{!"branch_weights", i32 7, i32 5}

// PROFDATA-DAG: ![[UNDTCTD0]] = !{!"function_entry_count", i64 10}
// PROFDATA-DAG: ![[UNDTCTD1]] = !{!"branch_weights", i32 7, i32 5}
// NODATA-DAG: ![[UNDTCTD0]] = !{!"function_entry_count", i64 10}
// NODATA-DAG: ![[UNDTCTD1]] = !{!"branch_weights", i32 7, i32 5}

// PROFDATA-DAG: ![[DIFF0]] = !{!"function_entry_count", i64 10}
// PROFDATA-DAG: ![[DIFF1]] = !{!"branch_weights", i32 7, i32 5}
