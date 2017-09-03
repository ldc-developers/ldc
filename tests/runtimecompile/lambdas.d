
// RUN: %ldc -enable-runtime-compile -run %s

import ldc.attributes;
import ldc.runtimecompile;

void main(string[] args)
{
  @runtimeCompile int foo()
  {
    return 42;
  }
  int val1 = 5;
  int val2 = 3;
  @runtimeCompile int bar()
  {
    val2 += 5;
    return val1;
  }

  rtCompileProcess();
  assert(42 == foo());
  assert(5 == bar());
  assert(8 == val2);
}
