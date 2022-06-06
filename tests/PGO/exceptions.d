// Test instrumentation of statement types involving exceptions.

// REQUIRES: PGO_RT

// XFAIL: Windows

// RUN: %ldc -c -output-ll -fprofile-instr-generate -of=%t.ll %s && FileCheck %s --check-prefix=PROFGEN < %t.ll

// RUN: %ldc -fprofile-instr-generate=%t.profraw -run %s  \
// RUN:   &&  %profdata merge %t.profraw -o %t.profdata \
// RUN:   &&  %ldc -boundscheck=off -c -output-ll -of=%t2.ll -fprofile-instr-use=%t.profdata %s \
// RUN:   &&  FileCheck %s -check-prefix=PROFUSE < %t2.ll

extern(C):  // simplify name mangling for simpler string matching

// PROFGEN-DAG: @[[TC:__(llvm_profile_counters|profc)_try_catch]] ={{.*}} global [9 x i64] zeroinitializer
// FIXME: fix the nr of counter entries:
// PROFGEN-DAG: @[[TF:__(llvm_profile_counters|profc)_try_finally]] ={{.*}} global [{{[0-9]+}} x i64] zeroinitializer
// PROFGEN-DAG: @[[SCP:__(llvm_profile_counters|profc)_scope_stmts]] ={{.*}} global [{{[0-9]+}} x i64] zeroinitializer

// Simple classes to disambiguate different throw statements
class ExceptionTwo : Exception {
  this(string s) { super(s); }
}
class ExceptionThree : Exception {
  this(string s) { super(s); }
}

// PROFGEN-LABEL: @scope_stmts(
// PROFUSE-LABEL: @scope_stmts(
// PROFGEN: store {{.*}} @[[SCP]], i{{32|64}} 0, i{{32|64}} 0
// PROFUSE-SAME: !prof ![[SCP0:[0-9]+]]
void scope_stmts(bool fail) {
  int i;
  scope(failure) i += 890;
  scope(exit) i += 234;
  scope(success) i += 456;
  if (fail) throw new Exception(i ? "true string" : "fail scope_stmts");

  // OnScope statements are lowered into TryCatch statements.
  // Simplify the tests to only make sure that all conditional branches have
  // branch weights,
  // PROFUSE-NOT: {{br i1 %[0-9]+, label %[A-Za-z0-9\.]+, label %[A-Za-z0-9\.]+$}}
  // and test that switches have profiling information too
  // P ROFUSE-NOT: {{^[ ]+\]$}}

  // Detect function end:
  // PROFUSE:      ret void
}

// PROFGEN-LABEL: @try_catch()
// PROFUSE-LABEL: @try_catch()
// PROFGEN: store {{.*}} @[[TC]], i{{32|64}} 0, i{{32|64}} 0
// PROFUSE-SAME: !prof ![[TC0:[0-9]+]]
void try_catch() {
  // PROFGEN: store {{.*}} @[[TC]], i{{32|64}} 0, i{{32|64}} 1
  // PROFUSE: br {{.*}} !prof ![[TC1:[0-9]+]]
  for (int i = 0; i < 6; ++i) { // 6 : 0 (branch taken)

    // The IR for TryStatements has a very unintuitive ordering.
    // The IR block ordering needs to be improved, but for now we have to
    // carefully read the generated IR and put the checks here in the according
    // order.
    // The try statement and exception catches receive their counter numbers
    // before recursing into the try body and exception handlers.

    try {
      if (i < 2) { // 2 : 4 (branch taken)
        throw new ExceptionThree("first"); // 2x
      } else if (i < 5) { // 3 : 1 (branch taken)
        throw new ExceptionTwo("two"); // 3x
      }
      throw new Exception("Uncaught exception");
    }
    catch (ExceptionTwo e) { // 3 : 3 (match : pass)
      if (i) {}  // 3 : 0 (branch taken)
    }
    catch (ExceptionThree e) { // 2 : 1 (match : pass)
      if (i) {}  // 1 : 1 (branch taken)
    }

    // Try body:  if(i < 2)
    // PROFGEN: store {{.*}} @[[TC]], i{{32|64}} 0, i{{32|64}} 5
    // More try body:  if(i < 5)
    // PROFGEN: store {{.*}} @[[TC]], i{{32|64}} 0, i{{32|64}} 6
    // ExceptionTwo body:
    // PROFGEN: store {{.*}} @[[TC]], i{{32|64}} 0, i{{32|64}} 3
    // More ExceptionTwo body:  if(i)
    // PROFGEN: store {{.*}} @[[TC]], i{{32|64}} 0, i{{32|64}} 7
    // ExceptionThree body:
    // PROFGEN: store {{.*}} @[[TC]], i{{32|64}} 0, i{{32|64}} 4
    // More ExceptionThree body:  if(i)
    // PROFGEN: store {{.*}} @[[TC]], i{{32|64}} 0, i{{32|64}} 8
    // Try end:
    // PROFGEN: store {{.*}} @[[TC]], i{{32|64}} 0, i{{32|64}} 2

    // Try body:  if(i < 2)
    // PROFUSE: br {{.*}} !prof ![[TC5:[0-9]+]]
    // More try body:  if(i < 5)
    // PROFUSE: br {{.*}} !prof ![[TC6:[0-9]+]]
    // Catch bodies:  if(i)
    // PROFUSE: br {{.*}} !prof ![[TC7:[0-9]+]]
    // PROFUSE: br {{.*}} !prof ![[TC8:[0-9]+]]
    // Landing pad - match ExceptionTwo:
    // PROFUSE: br {{.*}} !prof ![[TC3:[0-9]+]]
    // Landing pad - match ExceptionThree:
    // PROFUSE: br {{.*}} !prof ![[TC4:[0-9]+]]
  }
}

// PROFGEN-LABEL: @try_finally()
// PROFUSE-LABEL: @try_finally()
// PROFGEN: store {{.*}} @[[TF]], i{{32|64}} 0, i{{32|64}} 0
// PROFUSE-SAME: !prof ![[TF0:[0-9]+]]
void try_finally() {
  int i;
    try {
      i+=512;
      if (!i) throw new Exception("first");
    } finally {
      i+=765;
      //throw new Exception("second");
    }
//  } catch (Exception e) {
//    i+=91;//if (true) {}
  //}
}

// PROFGEN-LABEL: @_Dmain(
// PROFUSE-LABEL: @_Dmain(
extern(D):
void main() {
  try { // for testing an escaping Exception
    try_catch();
  } catch (Exception e) {}
  try_finally();
  scope_stmts(false);
  try {
    scope_stmts(true);
  } catch (Exception e) {}

  // Detect function end:
  // PROFUSE:      ret i32 0
}

// PROFUSE-DAG: ![[TC0]] = !{!"function_entry_count", i64 1}
// PROFUSE-DAG: ![[TC1]] = !{!"branch_weights", i32 7, i32 1}
// PROFUSE-DAG: ![[TC5]] = !{!"branch_weights", i32 3, i32 5}
// PROFUSE-DAG: ![[TC6]] = !{!"branch_weights", i32 4, i32 2}
// PROFUSE-DAG: ![[TC7]] = !{!"branch_weights", i32 4, i32 1}
// PROFUSE-DAG: ![[TC3]] = !{!"branch_weights", i32 4, i32 4}
// PROFUSE-DAG: ![[TC4]] = !{!"branch_weights", i32 3, i32 2}
// PROFUSE-DAG: ![[TC8]] = !{!"branch_weights", i32 2, i32 2}

// PROFUSE-DAG: ![[SCP0]] = !{!"function_entry_count", i64 2}
