// RUN: %ldc -c -output-ll -of=%t1.ll                  %s && FileCheck %s --check-prefix=NO_FP   < %t1.ll
// RUN: %ldc -c -output-ll -of=%t2.ll -g               %s && FileCheck %s --check-prefix=WITH_FP < %t2.ll
// RUN: %ldc -c -output-ll -of=%t3.ll -g -O3           %s && FileCheck %s --check-prefix=NO_FP   < %t3.ll
// RUN: %ldc -c -output-ll -of=%t4.ll -disable-fp-elim %s && FileCheck %s --check-prefix=WITH_FP < %t4.ll

int foo(int a)
{
    int x = a * a;
    return x;
}

// WITH_FP: "no-frame-pointer-elim"="true"
// NO_FP:   "no-frame-pointer-elim"="false"
