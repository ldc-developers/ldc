
// RUN: %ldc -enable-runtime-compile -run %s

import ldc.attributes;
import ldc.runtimecompile;

@dynamicCompile int foo(int i)
{
  if (i > 0)
  {
    return foo(i - 1) + 1;
  }
  return 0;
}

void main(string[] args)
{
  compileDynamicCode();
  assert(15 == foo(15));
}
