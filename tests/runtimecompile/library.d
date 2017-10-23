
// RUN: %ldc -enable-runtime-compile -lib -I%S %S/inputs/module1.d %S/inputs/module2.d %S/inputs/module3.d -of=%t1%lib
// RUN: %ldc -enable-runtime-compile -I%S %s %t1%lib -run

import ldc.attributes;
import ldc.runtimecompile;

import inputs.module1;
import inputs.module2;

@runtimeCompile int foo()
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
