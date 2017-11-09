
// RUN: %ldc -enable-dynamic-compile -run %s

import std.exception;
import ldc.attributes;
import ldc.runtimecompile;

__gshared int[555] arr1 = 42;
__gshared int[555] arr2 = 42;

@dynamicCompile int foo()
{
  int[555] a = arr1;
  return a[3];
}

@dynamicCompile int bar()
{
  arr2 = 0;
  return arr2[3];
}

void main(string[] args)
{
  compileDynamicCode();
  assert(42 == foo());
  assert(0  == bar());
}
