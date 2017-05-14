
// RUN: %ldc -enable-runtime-compile -run %s

import ldc.runtimecompile;

@runtimeCompile __gshared int foovar = 0;

@runtimeCompile int foo()
{
  return foovar;
}

void main(string[] args)
{
  rtCompileProcess();
  assert(0 == foo());
  foovar = 42;
  assert(0 == foo());
  rtCompileProcess();
  assert(42 == foo());
}
