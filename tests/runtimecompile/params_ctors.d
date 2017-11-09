
// RUN: %ldc -enable-dynamic-compile -run %s

import ldc.attributes;
import ldc.dynamic_compile;

__gshared int ctorsCalled = 0;
__gshared int dtorsCalled = 0;
__gshared int fooCalled = 0;

struct Foo
{
  this(this)
  {
    ++ctorsCalled;
  }
  
  ~this()
  {
    ++dtorsCalled;
  }
  
  void foo()
  {
    ++fooCalled;
  }
}

void func1(Foo f)
{
  f.foo();
}

@dynamicCompile void func2(Foo f)
{
  f.foo();
}

void main(string[] args)
{
  compileDynamicCode();
  Foo f;
  func1(f);
  assert(1 == ctorsCalled);
  assert(1 == dtorsCalled);
  assert(1 == fooCalled);
  ctorsCalled = 0;
  dtorsCalled = 0;
  fooCalled = 0;
  func2(f);
  assert(1 == ctorsCalled);
  assert(1 == dtorsCalled);
  assert(1 == fooCalled);
}
