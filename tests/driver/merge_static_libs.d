// RUN: %ldc -lib %s -of=%t_1%lib
// RUN: %ldc -lib %s -of=%t_2%lib
// RUN: %ldc -lib %t_1%lib %t_2%lib -of=%t_merged%lib

void foo() {}
