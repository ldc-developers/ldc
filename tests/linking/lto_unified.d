// Test unified lto commandline flag

// REQUIRES: LTO
// REQUIRES: atleast_llvm1800

// RUN: split-file %s %t

// RUN: %ldc %t/second.d -of=%t/second_thin%obj -c -flto=thin
// RUN: %ldc %t/third.d -of=%t/third_full%obj -c -flto=full
// RUN: %ldc -I%t %t/main.d %t/second_thin%obj %t/third_full%obj -flto=full -O

//--- main.d
import second;
import third;
void main()
{
    foo();
    g();
}

//--- second.d
void foo() {}

//--- third.d
void g() {}
