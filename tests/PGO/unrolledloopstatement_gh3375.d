// Test PGO instrumentation and profile use for front-end-unrolled loops.

// REQUIRES: PGO_RT

// RUN: %ldc -fprofile-instr-generate=%t.profraw -run %s
// RUN: %profdata merge %t.profraw -o %t.profdata
// RUN: %ldc -c -output-ll -of=%t2.ll -fprofile-instr-use=%t.profdata %s
// RUN: FileCheck -allow-deprecated-dag-overlap %s -check-prefix=PROFUSE < %t2.ll

alias AliasSeq(TList...) = TList;

void main() {
    foreach (i; 0..400)
        foofoofoo(i);
}

// PROFUSE-LABEL: define void @foofoofoo(
// PROFUSE-SAME: !prof ![[FUNCENTRY:[0-9]+]]
extern(C) void foofoofoo(int i)
{
    alias R = AliasSeq!(char, int);
    foreach (j, r; R)
    {
        if (i + 125*j > 200)
            continue;

        if (i + 125*j > 150)
            break;

        if (i-j == 0)
            goto function_exit;
    }
    /* The loop will be unrolled to:
    {
        // Here: i in [0..399] = 400 counts
        // PROFUSE: br {{.*}} !prof ![[IF1_1:[0-9]+]]
        if (i + 0 > 200)
            continue; // [201..399] = 199 counts

        // [0..200] = 201 counts
        // PROFUSE: br {{.*}} !prof ![[IF1_2:[0-9]+]]
        if (i + 0 > 150)
            break; // [151..200] = 50 counts

        // [0..150] = 151 counts
        // PROFUSE: br {{.*}} !prof ![[IF1_3:[0-9]+]]
        if (i-0 == 0)
            goto function_exit;
    }
    {
        // [1..150] U [201..399] = 150+199 = 349 counts
        // PROFUSE: br {{.*}} !prof ![[IF2_1:[0-9]+]]
        if (i + 125 > 200)
            continue; // [76..150] U [201..399] = 75+199 = 274 counts

        // [1..75] = 75 counts
        // PROFUSE: br {{.*}} !prof ![[IF2_2:[0-9]+]]
        if (i + 125 > 150)
            break; // [26..75] = 50 counts

        // [1..25] = 25 counts
        // PROFUSE: br {{.*}} !prof ![[IF2_3:[0-9]+]]
        if (i-1 == 0)
            goto function_exit;
    }
    */

    // [2..400] = 398 counts
    // PROFUSE: br {{.*}} !prof ![[IFEXIT:[0-9]+]]
    if (i) {} // always true

    // 400 counts
    function_exit:
}

// PROFUSE-DAG: ![[FUNCENTRY]] = !{!"function_entry_count", i64 400}
// PROFUSE-DAG: ![[IF1_1]] = !{!"branch_weights", i32 200, i32 202}
// PROFUSE-DAG: ![[IF1_2]] = !{!"branch_weights", i32 51, i32 152}
// PROFUSE-DAG: ![[IF1_3]] = !{!"branch_weights", i32 2, i32 151}
// PROFUSE-DAG: ![[IF2_1]] = !{!"branch_weights", i32 275, i32 76}
// PROFUSE-DAG: ![[IF2_2]] = !{!"branch_weights", i32 51, i32 26}
// PROFUSE-DAG: ![[IF2_3]] = !{!"branch_weights", i32 2, i32 25}
// PROFUSE-DAG: ![[IFEXIT]] = !{!"branch_weights", i32 399, i32 1}

