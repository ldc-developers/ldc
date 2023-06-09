// RUN: %ldc -c -singleobj %s %S/inputs/gh2782b.d

struct S {}
extern(C) void foo(S);
