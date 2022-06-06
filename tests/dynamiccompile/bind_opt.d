// RUN: %ldc -enable-dynamic-compile -run %s

import std.array;
import std.stdio;
import std.string;
import ldc.attributes;
import ldc.dynamic_compile;

@dynamicCompile
int foo(int a, int b, int c)
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

  @dynamicCompile int get(int val)
  {
    return f.i + f.j * 10 + k * 100 + val * 1000;
  }
}

@dynamicCompile
int bar(Bar b)
{
  return b.f.i + b.f.j * 10 + b.k * 100;
}

@dynamicCompileEmit
int zzz(int a, int b, int c)
{
  return a + b * 10 + c * 100;
}

@dynamicCompileEmit
int baz(int[4] a)
{
    return a[0] + a[1] * 10 + a[2] * 100 + a[3] * 1000;
}

void main(string[] args)
{
  auto dump = appender!string();
  CompilerSettings settings;
  settings.optLevel = 3;
  settings.dumpHandler = (DumpStage stage, in char[] str)
  {
    if (DumpStage.FinalAsm == stage)
    {
      write(str);
      dump.put(str);
    }
  };
  writeln("===========================================");
  compileDynamicCode(settings);
  writeln();
  writeln("===========================================");
  stdout.flush();

  auto elem = Bar(Foo(1,3),5);
  int[4] arr = [2,6,8,9];

  auto f = ldc.dynamic_compile.bind(&foo, 1, 2, 3);
  auto b = ldc.dynamic_compile.bind(&bar, Bar(Foo(4,5),6));
  auto z = ldc.dynamic_compile.bind(&foo, 7, 8, 9);
  auto e = ldc.dynamic_compile.bind(&elem.get, 7);
  auto a = ldc.dynamic_compile.bind(&baz, arr);

  compileDynamicCode(settings);

  assert(321 == f());
  assert(654 == b());
  assert(987 == z());
  assert(7531 == e());
  version (Win64)
  {
    // TODO: fix https://github.com/ldc-developers/ldc/issues/3695
  }
  else
  {
    assert(9862 == a());
  }
  assert(indexOf(dump.data, "321") != -1);
  assert(indexOf(dump.data, "654") != -1);
  assert(indexOf(dump.data, "987") != -1);
  //assert(indexOf(dump.data, "7531") != -1); // TODO: doesn't properly optimized yet
  version (Win64) { /* ditto */ } else assert(indexOf(dump.data, "9862") != -1);
}
