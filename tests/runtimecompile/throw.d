
// exceptions is broken on windows
// win64 issue https://bugs.llvm.org//show_bug.cgi?id=24233
// XFAIL: Windows
// RUN: %ldc -enable-runtime-compile -run %s

import std.exception;
import ldc.attributes;
import ldc.runtimecompile;

@dynamicCompile void foo()
{
  throw new Exception("foo");
}

@dynamicCompile int bar()
{
  try
  {
    throw new Exception("foo");
  }
  catch(Exception e)
  {
    return 42;
  }
  return 0;
}

@dynamicCompile int baz()
{
  try
  {
    foo();
  }
  catch(Exception e)
  {
    return 42;
  }
  return 0;
}

void main(string[] args)
{
  compileDynamicCode();
  assert(collectExceptionMsg(foo()) == "foo");
  assert(42 == bar());
  assert(42 == baz());
}
