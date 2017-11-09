
// RUN: %ldc -enable-dynamic-compile -run %s

import ldc.attributes;
import ldc.dynamic_compile;

@dynamicCompileConst __gshared int foovar = 0;

@dynamicCompile int foo()
{
  return foovar;
}

@dynamicCompileConst __gshared int barvar = 5;

@dynamicCompile int bar()
{
  return barvar;
}

void main(string[] args)
{
  compileDynamicCode();
  assert(0 == foo());
  assert(5 == bar());
  foovar = 42;
  barvar = 43;
  assert(0 == foo());
  assert(5 == bar());
  compileDynamicCode();
  assert(42 == foo());
  assert(43 == bar());
}
