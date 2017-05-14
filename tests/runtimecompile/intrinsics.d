
// RUN: %ldc -enable-runtime-compile -run %s

import std.exception;
import ldc.runtimecompile;

__gshared int[555] arr1 = 42;
__gshared int[555] arr2 = 42;

@runtimeCompile int foo()
{
  int[555] a = arr1;
  return a[3];
}

@runtimeCompile int bar()
{
  arr2 = 0;
  return arr2[3];
}

void main(string[] args)
{
  rtCompileProcess();
  assert(42 == foo());
  assert(0  == bar());
}
