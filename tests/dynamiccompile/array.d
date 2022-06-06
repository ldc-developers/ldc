
// RUN: %ldc -enable-dynamic-compile -run %s

import std.exception;
import ldc.attributes;
import ldc.dynamic_compile;

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

@dynamicCompile int baz()
{
  // Large stack array to force compiler to emit __chkstk call
  int[4 * 1024] a;
  return a[3 * 1024];
}

void main(string[] args)
{
  compileDynamicCode();
  assert(42 == foo());
  assert(0  == bar());
  assert(0  == baz());
}
