
// RUN: %ldc -enable-runtime-compile -run %s

import std.stdio;
import ldc.attributes;
import ldc.runtimecompile;

@runtimeCompile int foo()
{
  return 5;
}

void main(string[] args)
{
  bool dumpHandlerCalled = false;
  CompilerSettings settings;
  settings.dumpHandler = ((a) { dumpHandlerCalled = true; });
  rtCompileProcess(settings);
  assert(5 == foo());
  assert(dumpHandlerCalled);
}
