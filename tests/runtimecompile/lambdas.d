
// RUN: %ldc -enable-dynamic-compile -run %s

import ldc.attributes;
import ldc.dynamic_compile;

void main(string[] args)
{
  @dynamicCompile int foo()
  {
    return 42;
  }
  int val1 = 5;
  int val2 = 3;
  @dynamicCompile int bar()
  {
    val2 += 5;
    return val1;
  }

  compileDynamicCode();
  assert(42 == foo());
  assert(5 == bar());
  assert(8 == val2);
}
