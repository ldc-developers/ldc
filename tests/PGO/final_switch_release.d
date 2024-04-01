// See GH issue 3375

// REQUIRES: PGO_RT

// Test instrumentation for final switches without default case.
// The frontend creates hidden default case to catch runtime errors. We disable
// that using `-release`.

// RUN: %ldc -release -fprofile-instr-generate=%t.profraw -run %s  \
// RUN:   &&  %profdata merge %t.profraw -o %t.profdata \
// RUN:   &&  %ldc -release -c -output-ll -of=%t.ll -fprofile-instr-use=%t.profdata %s \
// RUN:   &&  FileCheck %s < %t.ll

extern (C): // Simplify matching by disabling function name mangling

enum A
{
    Start,
    End
}

// CHECK-LABEL: @final_switch(
// CHECK-SAME: !prof ![[SW0:[0-9]+]]
void final_switch(A state)
{
    // CHECK: switch {{.*}} [
    // CHECK: ], !prof ![[SW1:[0-9]+]]
    final switch (state)
    {
    case A.Start:
        break;
    case A.End:
        break;
    }
}

void main()
{
    final_switch(A.Start);
    final_switch(A.End);
    final_switch(A.Start);
}

// CHECK-DAG: ![[SW0]] = !{!"function_entry_count", i64 3}
// CHECK-DAG: ![[SW1]] = !{!"branch_weights", i32 1, i32 3, i32 2}
