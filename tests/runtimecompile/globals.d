
// RUN: %ldc -enable-runtime-compile -run %s

import ldc.attributes;
import ldc.runtimecompile;

@runtimeCompile __gshared int foovar = 0;

@runtimeCompile int foo()
{
  return foovar;
}

void main(string[] args)
{
  compileDynamicCode();
  assert(0 == foo());
  foovar = 42;
  assert(0 == foo());
  compileDynamicCode();
  assert(42 == foo());
}
