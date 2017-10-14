
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
  bool progressHandlerCalled = false;
  CompilerSettings settings;

  settings.dumpHandler = (in char[] str)
  {
    stderr.write(str);
    stderr.flush();
    dumpHandlerCalled = true;
  };
  settings.progressHandler = (in char[] desc, in char[] object)
  {
    stderr.writefln("Progress: %s -- %s", desc, object);
    stderr.flush();
    progressHandlerCalled = true;
  };
  compileDynamicCode(settings);
  assert(5 == foo());
  assert(dumpHandlerCalled);
  assert(progressHandlerCalled);
}
