// REQUIRES: Windows

void foo(int a)
{
    a += 3;
}

// RUN: %ldc -g -output-ll -of=%t.ll %s               && FileCheck --check-prefix=NOCOL %s < %t.ll
// NOCOL-NOT: column:
// NOCOL: !DILocation(line:
// NOCOL-NOT: column:

// RUN: %ldc -g -output-ll -of=%t.ll %s -gcolumn-info && FileCheck --check-prefix=WITHCOL %s < %t.ll
// WITHCOL: !DILocation(line:
// WITHCOL-SAME: column:
