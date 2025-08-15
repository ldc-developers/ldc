// RUN: %ldc -lib %s -of=%t%lib
// RUN: %ldc -lib %t%lib -of=%t_merged%lib
// RUN: %diff_binary %t%lib %t_merged%lib

void foo() {}
