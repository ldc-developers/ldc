
// RUN: %ldc -enable-runtime-compile -run %s

import std.exception;
import ldc.runtimecompile;

@runtimeCompile void foo()
{
  throw new Exception("foo");
}

@runtimeCompile int bar()
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

@runtimeCompile int baz()
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
  rtCompileProcess();
  assert(collectExceptionMsg(foo()) == "foo");
  assert(42 == bar());
  assert(42 == baz());
}
