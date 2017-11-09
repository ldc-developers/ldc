
// RUN: %ldc -enable-dynamic-compile -run %s

import ldc.attributes;
import ldc.dynamic_compile;

interface IFoo
{
  int foo();
}

class Foo : IFoo
{
  int val = 0;

  @dynamicCompile int foo()
  {
    return val;
  }
}

void main(string[] args)
{
  auto f1 = new Foo;
  auto f2 = cast(IFoo)f1;
  auto fun = &f1.foo;
  f1.val = 42;

  compileDynamicCode();
  assert(42 == f1.foo());
  assert(42 == f2.foo());
  assert(42 == fun());
}
