
// RUN: %ldc -enable-dynamic-compile -run %s

import ldc.attributes;
import ldc.dynamic_compile;

@dynamicCompile
int foo(int a, int b, int c)
{
  return a + b * 10 + c * 100;
}

int bar(int a, int b, int c)
{
  return a + b * 10 + c * 100;
}

struct Foo
{
  int i;
  int j;
}

struct Bar
{
  Foo f;
  int k;

  @dynamicCompileEmit int get(int val)
  {
    return f.i + f.j * 10 + k * 100 + val * 1000;
  }
}

@dynamicCompile
int baz1(Bar b)
{
  return b.f.i + b.f.j * 10 + b.k * 100;
}

@dynamicCompile
int baz2(Bar* b)
{
  return b.f.i + b.f.j * 10 + b.k * 100;
}

@dynamicCompile
int zzz1(int function() f)
{
  return f();
}

@dynamicCompile
int zzz2(int delegate() f)
{
  return f();
}

int get6()
{
  return 6;
}

@dynamicCompile
int yyy(int i, int delegate() j)
{
  return i + j();
}

@dynamicCompile
int ccc(int i, const int j, immutable int k)
{
  return i + j + k;
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
    assert(!f1.isNull());
    assert(!f2.isNull());
    assert(!f3.isNull());
    assert(!f4.isNull());

    int delegate(int,int,int) fd1 = f1.toDelegate();
    int delegate(int)         fd2 = f2.toDelegate();
    int delegate()            fd3 = f3.toDelegate();
    int delegate()            fd4 = f4.toDelegate();

    auto b1 = ldc.dynamic_compile.bind(&bar, placeholder, placeholder, placeholder);
    auto b2 = ldc.dynamic_compile.bind(&bar, 1, placeholder, 3);
    auto b3 = ldc.dynamic_compile.bind(&bar, 1, 2, 3);
    auto b4 = b3;

    auto bz1 = ldc.dynamic_compile.bind(&baz1, Bar(Foo(1,2),3));
    auto b = Bar(Foo(1,2),3);
    auto bz2 = ldc.dynamic_compile.bind(&baz2, &b);

    int delegate() bzd1 = bz1.toDelegate();
    int delegate() bzd2 = bz2.toDelegate();

    auto zz1 = ldc.dynamic_compile.bind(&zzz1, &get6);
    import std.functional;
    auto zz2 = ldc.dynamic_compile.bind(&zzz2, toDelegate(&get6));

    int delegate() zzd1 = zz1.toDelegate();
    int delegate() zzd2 = zz2.toDelegate();

    auto elem = Bar(Foo(1,3),5);
    auto yy = ldc.dynamic_compile.bind(&elem.get, 7);
    auto yyd = yy.toDelegate();

    int dget6()
    {
      return 6;
    }

    auto p = ldc.dynamic_compile.bind(&yyy, placeholder, &dget6);

    auto c = ldc.dynamic_compile.bind(&ccc, 1, 10, 100);

    compileDynamicCode(settings);
    assert(f1.isCallable());
    assert(f2.isCallable());
    assert(f3.isCallable());
    assert(f4.isCallable());

    assert(321 == f1(1,2,3));
    assert(321 == f2(2));
    assert(321 == f3());
    assert(321 == f4());

    assert(321 == fd1(1,2,3));
    assert(321 == fd2(2));
    assert(321 == fd3());
    assert(321 == fd4());

    f3 = null;
    assert(f3.isNull());
    assert(!f4.isNull());
    assert(!f3.isCallable());
    assert(f4.isCallable());

    assert(b1.isCallable());
    assert(b2.isCallable());
    assert(b3.isCallable());
    assert(b4.isCallable());

    assert(321 == b1(1,2,3));
    assert(321 == b2(2));
    assert(321 == b3());
    assert(321 == b4());

    assert(321 == bz1());
    assert(321 == bz2());

    assert(321 == bzd1());
    assert(321 == bzd2());

    assert(6 == zz1());
    assert(6 == zz2());

    assert(6 == zzd1());
    assert(6 == zzd2());

    assert(yy.isCallable());
    assert(7531 == yy());
    assert(7531 == yyd());

    assert(10 == p(4));
    assert(111 == c());
  }
}
