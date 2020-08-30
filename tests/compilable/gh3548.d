// Test inner struct initZ linking for a pragma(inline) function and separate compilation.

// RUN: %ldc -c %S/inputs/gh3548a.d -of=%t.a.o
// RUN: %ldc -I%S/inputs %s %t.a.o

module b;

import gh3548a;

void main() {
    S s;
    s.innerStruct();
    s.innerInnerStruct();
    s.innerClass();
    s.innerInnerClass();
}
