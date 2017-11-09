
// llvm generates overlapped simd reads and writes to init these
// structs but fails on win32 for some reason
// XFAIL: Windows_x86
// RUN: %ldc -enable-dynamic-compile -run %s

import ldc.dynamic_compile;
import ldc.attributes;
import std.stdio;

struct Foo1
{
  int[4 + 2] arr = [ 0 , 1, 2, 3, 4, 5];
}

struct Foo2
{
  int[8 + 4] arr = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11];
}

struct Foo3
{
  int[16 + 8] arr = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,
                     12,13,14,15,16,17,18,19,20,21,22,23];
}

Foo1 foo1()
{
  auto f = Foo1();
  return f;
}

Foo2 foo2()
{
  auto f = Foo2();
  return f;
}

Foo3 foo3()
{
  auto f = Foo3();
  return f;
}

@dynamicCompile Foo1 bar1()
{
  auto f = Foo1();
  return f;
}

@dynamicCompile Foo2 bar2()
{
  auto f = Foo2();
  return f;
}

@dynamicCompile Foo3 bar3()
{
  auto f = Foo3();
  return f;
}

void main(string[] args)
{
  compileDynamicCode();
  stdout.flush();
  auto f1 = foo1();
  auto f2 = foo2();
  auto f3 = foo3();
  auto b1 = bar1();
  auto b2 = bar2();
  auto b3 = bar3();
  assert(f1 == b1);
  assert(f2 == b2);
  assert(f3 == b3);
}
