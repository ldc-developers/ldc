
// RUN: %ldc -enable-dynamic-compile -run %s

import ldc.attributes;
import ldc.dynamic_compile;

@dynamicCompile int foo(int a, int b, int c)
{
  return a + b + c;
}

int bar(int a, int b, int c)
{
  return a + b + c;
}

void main(string[] args)
{
  foreach (i; 0..4)
  {
    CompilerSettings settings;
    settings.optLevel = i;

    auto f1 = ldc.dynamic_compile.bind(&foo, placeholder, placeholder, placeholder);
    auto f2 = ldc.dynamic_compile.bind(&foo, 1, placeholder, 3);
    auto f3 = ldc.dynamic_compile.bind(&foo, 1, 2, 3);
    auto f4 = f3;

    int delegate(int,int,int) fd1 = f1.toDelegate();
    int delegate(int)         fd2 = f2.toDelegate();
    int delegate()            fd3 = f3.toDelegate();
    int delegate()            fd4 = f4.toDelegate();

    auto b1 = ldc.dynamic_compile.bind(&bar, placeholder, placeholder, placeholder);
    auto b2 = ldc.dynamic_compile.bind(&bar, 1, placeholder, 3);
    auto b3 = ldc.dynamic_compile.bind(&bar, 1, 2, 3);
    auto b4 = b3;

    compileDynamicCode(settings);
    assert(6 == f1(1,2,3));
    assert(6 == f2(2));
    assert(6 == f3());
    assert(6 == f4());

    assert(6 == fd1(1,2,3));
    assert(6 == fd2(2));
    assert(6 == fd3());
    assert(6 == fd4());

    assert(!b1.isCallable());
    assert(!b2.isCallable());
    assert(!b3.isCallable());
    assert(!b4.isCallable());
  }
}
