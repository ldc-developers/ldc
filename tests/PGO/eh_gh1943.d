// RUN: %ldc -c -fprofile-instr-generate=%t.profraw %s

void foo();

void bar()
{
    try { foo(); }
    catch (Throwable) {}
}
