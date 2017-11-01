
// RUN: %ldc -enable-runtime-compile -run %s

import std.stdio;
import ldc.attributes;
import ldc.runtimecompile;

__gshared int ctorsCalled = 0; 
__gshared int dtorsCalled = 0;

struct Foo
{
  this(this)
  {
    ++ctorsCalled;
    writefln("ctor %s", &this);
  }
  
  ~this()
  {
    ++dtorsCalled;
    writefln("dtor %s", &this);
  }
  
  void foo()
  {
    writefln("foo %s", &this);
  }
}

void func1(Foo f)
{
  f.foo();
}

@runtimeCompile void func2(Foo f)
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
  ctorsCalled = 0;
  dtorsCalled = 0;
  func2(f);
  assert(1 == ctorsCalled);
  assert(1 == dtorsCalled);
}
