// Test calculation of execution counts with loops with break-to-label and continue-to-label.

// REQUIRES: PGO_RT

// RUN: %ldc -c -output-ll -fprofile-instr-generate -of=%t.ll %s  \
// RUN:   &&  FileCheck -allow-deprecated-dag-overlap %s --check-prefix=PROFGEN < %t.ll

// RUN: %ldc -fprofile-instr-generate=%t.profraw -run %s  \
// RUN:   &&  %profdata merge %t.profraw -o %t.profdata \
// RUN:   &&  %ldc -boundscheck=off -c -output-ll -of=%t2.ll -fprofile-instr-use=%t.profdata %s \
// RUN:   &&  FileCheck -allow-deprecated-dag-overlap %s -check-prefix=PROFUSE < %t2.ll

extern(C):  // simplify name mangling for simpler string matching

// PROFGEN-DAG: @[[BREAK:__(llvm_profile_counters|profc)_testbreak]] ={{.*}} global [8 x i64] zeroinitializer
// PROFGEN-DAG: @[[CONT:__(llvm_profile_counters|profc)_testcontinue]] ={{.*}} global [8 x i64] zeroinitializer

// PROFGEN-LABEL: @testbreak({{.*}})
// PROFUSE-LABEL: @testbreak({{.*}})
// PROFGEN: store {{.*}} @[[BREAK]]
// PROFUSE-SAME: !prof ![[BREAK0:[0-9]+]]
void testbreak(bool a) {

  // PROFGEN: store {{.*}} @[[BREAK]], i{{32|64}} 0, i{{32|64}} 1
outer:
  // PROFGEN: store {{.*}} @[[BREAK]], i{{32|64}} 0, i{{32|64}} 2
  // PROFUSE: br {{.*}} !prof ![[BREAK2:[0-9]+]]
  foreach (i; 0..4) {
    // PROFGEN: store {{.*}} @[[BREAK]], i{{32|64}} 0, i{{32|64}} 3
    // PROFUSE: br {{.*}} !prof ![[BREAK3:[0-9]+]]
    foreach (j; 0..4) {
      // PROFGEN: store {{.*}} @[[BREAK]], i{{32|64}} 0, i{{32|64}} 4
      // PROFUSE: br {{.*}} !prof ![[BREAK4:[0-9]+]]
      if (i>0)
        break outer;

      // PROFGEN: store {{.*}} @[[BREAK]], i{{32|64}} 0, i{{32|64}} 5
      // PROFUSE: br {{.*}} !prof ![[BREAK5:[0-9]+]]
      if (a) {}
    }

    // PROFGEN: store {{.*}} @[[BREAK]], i{{32|64}} 0, i{{32|64}} 6
    // PROFUSE: br {{.*}} !prof ![[BREAK6:[0-9]+]]
    if (a) {}
  }

  // PROFGEN: store {{.*}} @[[BREAK]], i{{32|64}} 0, i{{32|64}} 7
  // PROFUSE: br {{.*}} !prof ![[BREAK7:[0-9]+]]
  if (a) {}
}

// PROFGEN-LABEL: @testcontinue({{.*}})
// PROFUSE-LABEL: @testcontinue({{.*}})
// PROFGEN: store {{.*}} @[[CONT]]
// PROFUSE-SAME: !prof ![[CONT0:[0-9]+]]
void testcontinue(bool a) {

  // PROFGEN: store {{.*}} @[[CONT]], i{{32|64}} 0, i{{32|64}} 1
outer:
  // PROFGEN: store {{.*}} @[[CONT]], i{{32|64}} 0, i{{32|64}} 2
  // PROFUSE: br {{.*}} !prof ![[CONT2:[0-9]+]]
  foreach (i; 0..4) {
    // PROFGEN: store {{.*}} @[[CONT]], i{{32|64}} 0, i{{32|64}} 3
    // PROFUSE: br {{.*}} !prof ![[CONT3:[0-9]+]]
    foreach (j; 0..4) {
      // PROFGEN: store {{.*}} @[[CONT]], i{{32|64}} 0, i{{32|64}} 4
      // PROFUSE: br {{.*}} !prof ![[CONT4:[0-9]+]]
      if (i>0)
        continue outer;

      // PROFGEN: store {{.*}} @[[CONT]], i{{32|64}} 0, i{{32|64}} 5
      // PROFUSE: br {{.*}} !prof ![[CONT5:[0-9]+]]
      if (a) {}
    }

    // PROFGEN: store {{.*}} @[[CONT]], i{{32|64}} 0, i{{32|64}} 6
    // PROFUSE: br {{.*}} !prof ![[CONT6:[0-9]+]]
    if (a) {}
  }

  // PROFGEN: store {{.*}} @[[CONT]], i{{32|64}} 0, i{{32|64}} 7
  // PROFUSE: br {{.*}} !prof ![[CONT7:[0-9]+]]
  if (a) {}
}

// PROFGEN-LABEL: @_Dmain(
// PROFUSE-LABEL: @_Dmain(
extern(D):
void main() {
  testbreak(false);
  testcontinue(false);
}


// PROFUSE-DAG: ![[BREAK0]] = !{!"function_entry_count", i64 1}
// PROFUSE-DAG: ![[BREAK2]] = !{!"branch_weights", i32 3, i32 1}
// PROFUSE-DAG: ![[BREAK3]] = !{!"branch_weights", i32 6, i32 2}
// PROFUSE-DAG: ![[BREAK4]] = !{!"branch_weights", i32 2, i32 5}
// PROFUSE-DAG: ![[BREAK5]] = !{!"branch_weights", i32 1, i32 5}
// PROFUSE-DAG: ![[BREAK6]] = !{!"branch_weights", i32 1, i32 2}
// PROFUSE-DAG: ![[BREAK7]] = !{!"branch_weights", i32 1, i32 2}

// PROFUSE-DAG: ![[CONT0]] = !{!"function_entry_count", i64 1}
// PROFUSE-DAG: ![[CONT2]] = !{!"branch_weights", i32 5, i32 2}
// PROFUSE-DAG: ![[CONT3]] = !{!"branch_weights", i32 8, i32 2}
// PROFUSE-DAG: ![[CONT4]] = !{!"branch_weights", i32 4, i32 5}
// PROFUSE-DAG: ![[CONT5]] = !{!"branch_weights", i32 1, i32 5}
// PROFUSE-DAG: ![[CONT6]] = !{!"branch_weights", i32 1, i32 2}
// PROFUSE-DAG: ![[CONT7]] = !{!"branch_weights", i32 1, i32 2}
