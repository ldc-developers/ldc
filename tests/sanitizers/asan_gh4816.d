// Test extern(C) structs (zero sized structs) and zero-sized arrays. Github #4816

// REQUIRES: ASan

// RUN: %ldc -g -fsanitize=address %s -of=%t%exe
// RUN: %t%exe

auto foo(void[0] bar) { }

extern(C) struct S {}
auto foo(S s) { }

void main()
{
    void[0] bar;
    foo(bar);

    S s;
    foo(s);
}
