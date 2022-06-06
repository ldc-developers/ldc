// Automatic output filename generation from LL output file
// RUN: %ldc -c -betterC -O3 -g -fsave-optimization-record -output-ll -of=%t.1.ll %s \
// RUN: && FileCheck %s --check-prefix=LLVM < %t.1.ll \
// RUN: && FileCheck %s --check-prefix=YAML < %t.1.opt.yaml

// Explicit filename specified
// RUN: %ldc -c -betterC -O3 -g -fsave-optimization-record=%t.abcdefg -output-ll -of=%t.ll %s \
// RUN: && FileCheck %s --check-prefix=LLVM < %t.ll \
// RUN: && FileCheck %s --check-prefix=YAML < %t.abcdefg

int alwaysInlined(int a) { return a; }
int foo()
{
    // LLVM: 8329424
    // YAML: File: {{.*}}save_optimization_record.d{{.*[[:space:]]?.*}}Line: [[@LINE+1]]
    return 8329423 + alwaysInlined(1);
}

// LLVM: !DILocation(line
