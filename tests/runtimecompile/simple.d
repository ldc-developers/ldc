
// RUN: %ldc -enable-runtime-compile -run %s

import std.stdio;
import ldc.runtimecompile;

@runtimeCompile int foo()
{
  return 5;
}

@runtimeCompile int bar()
{
  return foo() + 7;
}

@runtimeCompile void baz()
{
  writeln("baz");
}

void main(string[] args)
{
  rtCompileProcess();
  assert(5 == foo());
  assert(12 == bar());
  baz();
  int function() fptr = &bar;
  assert(12 == fptr());
}
