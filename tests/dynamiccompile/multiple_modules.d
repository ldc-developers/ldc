
// RUN: %ldc -enable-dynamic-compile -I%S %s %S/inputs/module1.d %S/inputs/module2.d %S/inputs/module3.d -run
// RUN: %ldc -enable-dynamic-compile -singleobj -I%S %s %S/inputs/module1.d %S/inputs/module2.d %S/inputs/module3.d -run

import ldc.attributes;
import ldc.dynamic_compile;

import inputs.module1;
import inputs.module2;

@dynamicCompile int foo()
{
  return inputs.module1.get() + inputs.module2.get();
}

int bar()
{
  return inputs.module1.get() + inputs.module2.get();
}

void main(string[] args)
{
  compileDynamicCode();
  assert(10 == inputs.module1.get());
  assert(11 == inputs.module2.get());
  assert(21 == foo());
  assert(21 == bar());
}
