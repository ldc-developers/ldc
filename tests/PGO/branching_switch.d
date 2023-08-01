// Test instrumentation of switch with non-constant case expression.

// REQUIRES: PGO_RT

// RUN: %ldc -c -output-ll -fprofile-instr-generate -of=%t.ll %s  \
// RUN:   &&  FileCheck -allow-deprecated-dag-overlap %s --check-prefix=PROFGEN < %t.ll

// RUN: %ldc -fprofile-instr-generate=%t.profraw -run %s  \
// RUN:   &&  %profdata merge %t.profraw -o %t.profdata \
// RUN:   &&  %ldc -boundscheck=off -c -output-ll -of=%t2.ll -fprofile-instr-use=%t.profdata %s \
// RUN:   &&  FileCheck -allow-deprecated-dag-overlap %s -check-prefix=PROFUSE < %t2.ll

extern(C):  // simplify name mangling for simpler string matching

// PROFGEN-DAG: @[[BoB:__(llvm_profile_counters|profc)_bunch_of_branches]] ={{.*}} global [15 x i64] zeroinitializer

// PROFGEN-LABEL: @bunch_of_branches(i32
// PROFUSE-LABEL: @bunch_of_branches(i32
// PROFGEN: store {{.*}} @[[BoB]]
// PROFUSE-SAME: !prof ![[BoB0:[0-9]+]]
void bunch_of_branches(const uint two) {
  uint i;

  // PROFGEN: store {{.*}} @[[BoB]], i{{32|64}} 0, i{{32|64}} 1
  // PROFUSE: br {{.*}} !prof ![[BoB1:[0-9]+]]
  for (i = 1; i < 4; ++i) {

    switch (i) {
    // PROFGEN: store {{.*}} @[[BoB]], i{{32|64}} 0, i{{32|64}} 3
    case 1: // 1 + 1*gototarget
      // PROFGEN: store {{.*}} @[[BoB]], i{{32|64}} 0, i{{32|64}} 4

      // PROFGEN: store {{.*}} @[[BoB]], i{{32|64}} 0, i{{32|64}} 5
      // PROFUSE: br {{.*}} !prof ![[BoB5:[0-9]+]]
      if (i != 1) {}
      goto default;

    // PROFGEN: store {{.*}} @[[BoB]], i{{32|64}} 0, i{{32|64}} 6
    case 11: // 0x

      // PROFGEN: store {{.*}} @[[BoB]], i{{32|64}} 0, i{{32|64}} 7
      // never reached, no branch weights
      if (i != 11) {}
      goto case;

    // PROFGEN: store {{.*}} @[[BoB]], i{{32|64}} 0, i{{32|64}} 8
    case two: // 1x

      // PROFGEN: store {{.*}} @[[BoB]], i{{32|64}} 0, i{{32|64}} 9
      // PROFUSE: br {{.*}} !prof ![[BoB9:[0-9]+]]
      if (i != 2) {}
      goto case 1;

    // PROFGEN: store {{.*}} @[[BoB]], i{{32|64}} 0, i{{32|64}} 10
    default: // 1 + 2*gototarget
      // PROFGEN: store {{.*}} @[[BoB]], i{{32|64}} 0, i{{32|64}} 11
      goto case;

    // PROFGEN: store {{.*}} @[[BoB]], i{{32|64}} 0, i{{32|64}} 12
    case 5: // 0 + 3*fallthrough

      // PROFGEN: store {{.*}} @[[BoB]], i{{32|64}} 0, i{{32|64}} 13
      // PROFUSE: br {{.*}} !prof ![[BoB13:[0-9]+]]
      if (i != 5) {}
      break;
    }
    // PROFGEN: store {{.*}} @[[BoB]], i{{32|64}} 0, i{{32|64}} 2

    // Bunch of compares and branches is put at the end in IR
    // PROFUSE: br {{.*}} !prof ![[BoB3:[0-9]+]]
    // PROFUSE: br {{.*}} !prof ![[BoB6:[0-9]+]]
    // PROFUSE: br {{.*}} !prof ![[BoB8:[0-9]+]]
    // PROFUSE: br {{.*}} !prof ![[BoB12:[0-9]+]]
  }

  // PROFGEN: store {{.*}} @[[BoB]], i{{32|64}} 0, i{{32|64}} 14
  // PROFUSE: br {{.*}} !prof ![[BoB14:[0-9]+]]
  if (i) {}
}



// PROFGEN-LABEL: @_Dmain(
// PROFUSE-LABEL: @_Dmain(
extern(D):
void main() {
  bunch_of_branches(2);
}


// PROFUSE-DAG: ![[BoB0]] = !{!"function_entry_count", i64 1}
// PROFUSE-DAG: ![[BoB1]] = !{!"branch_weights", i32 4, i32 2}

// PROFUSE-DAG: ![[BoB3]] = !{!"branch_weights", i32 2, i32 3}
// PROFUSE-DAG: ![[BoB6]] = !{!"branch_weights", i32 1, i32 3}
// PROFUSE-DAG: ![[BoB8]] = !{!"branch_weights", i32 2, i32 2}
// PROFUSE-DAG: ![[BoB12]] = !{!"branch_weights", i32 1, i32 2}

// PROFUSE-DAG: ![[BoB5]] = !{!"branch_weights", i32 2, i32 2}
// PROFUSE-DAG: ![[BoB9]] = !{!"branch_weights", i32 1, i32 2}
// PROFUSE-DAG: ![[BoB13]] = !{!"branch_weights", i32 1, i32 -1}
