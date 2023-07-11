// Test -fcf-protection

// REQUIRES: target_X86

// RUN: %ldc -mtriple=x86_64-linux-gnu -output-ll -of=%t.ll        %s                         && FileCheck %s --check-prefix=NOTHING < %t.ll

// RUN: %ldc -mtriple=x86_64-linux-gnu -output-ll -of=%t_branch.ll %s --fcf-protection=branch -d-version=BRANCH && FileCheck %s --check-prefix=BRANCH < %t_branch.ll
// RUN: %ldc -mtriple=x86_64-linux-gnu -output-ll -of=%t_return.ll %s --fcf-protection=return -d-version=RETURN && FileCheck %s --check-prefix=RETURN < %t_return.ll
// RUN: %ldc -mtriple=x86_64-linux-gnu -output-ll -of=%t_full.ll   %s --fcf-protection=full   -d-version=FULL   && FileCheck %s --check-prefix=FULL   < %t_full.ll
// RUN: %ldc -mtriple=x86_64-linux-gnu -output-ll -of=%t_noarg.ll  %s --fcf-protection        -d-version=FULL   && FileCheck %s --check-prefix=FULL   < %t_noarg.ll

// NOTHING-NOT: cf-prot
// BRANCH-DAG: "cf-protection-branch", i32 1
// RETURN-DAG: "cf-protection-return", i32 1
// FULL-DAG: "cf-protection-branch", i32 1
// FULL-DAG: "cf-protection-return", i32 1

void foo() {}

version(BRANCH) {
    version(__CET_1__) {} else { static assert(false); };
}
version(RETURN) {
    version(__CET_2__) {} else { static assert(false); };
}
version(FULL) {
    version(__CET_3__) {} else { static assert(false); };
}
