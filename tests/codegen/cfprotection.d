// REQUIRES: target_X86

// RUN: %ldc -mtriple=x86_64 -output-ll -of=%t_undefined.ll %s
// RUN : FileCheck %s --check-prefix=UNDEFINED < %t_undefined.ll

// RUN: %ldc -mtriple=x86_64 -fcf-protection=none -output-ll -of=%t_none.ll %s
// RUN: FileCheck %s --check-prefix=NONE < %t_none.ll

// RUN: %ldc -mtriple=x86_64 -fcf-protection=branch -output-ll -of=%t_branch.ll %s
// RUN: FileCheck %s --check-prefix=BRANCH < %t_branch.ll

// RUN: %ldc -mtriple=x86_64 -fcf-protection=return -output-ll -of=%t_return.ll %s
// RUN: FileCheck %s --check-prefix=RETURN < %t_return.ll

// RUN: %ldc -mtriple=x86_64 -fcf-protection=full -output-ll -of=%t_full.ll %s
// RUN: FileCheck %s --check-prefix=FULL < %t_full.ll

void foo() {}

// UNDEFINED-NOT: !"cf-protection-branch"
// NONE-NOT: !"cf-protection-branch"
// BRANCH: !"cf-protection-branch", i32 1
// RETURN-NOT: !"cf-protection-branch"
// FULL: !"cf-protection-branch", i32 1

// UNDEFINED-NOT: !"cf-protection-return"
// NONE-NOT: !"cf-protection-return"
// BRANCH-NOT: !"cf-protection-return"
// RETURN: !"cf-protection-return", i32 1
// FULL: !"cf-protection-return", i32 1
