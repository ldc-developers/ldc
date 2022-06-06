
// RUN: %ldc -enable-dynamic-compile -run %s

import std.array;
import std.stdio;
import std.string;
import ldc.attributes;
import ldc.dynamic_compile;

@dynamicCompile
{

int foo(int delegate() a, int delegate() b)
{
  return a() + b();
}

int bar(int delegate() a, int delegate() b)
{
  return a() + b();
}

int getVal(int val)
{
  return val;
}
}

void main(string[] args)
{
  auto dump = appender!string();
  CompilerSettings settings;
  settings.optLevel = 3;
  settings.dumpHandler = (DumpStage stage, in char[] str)
  {
    if (DumpStage.FinalAsm == stage ||
          DumpStage.MergedModule == stage ||
          DumpStage.OptimizedModule == stage)
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

  auto v1 = ldc.dynamic_compile.bind(&getVal, 1001);
  auto v2 = ldc.dynamic_compile.bind(&getVal, 1002);
  auto v3 = ldc.dynamic_compile.bind(&getVal, 1003);

  auto d1 = v1.toDelegate();
  auto d2 = v2.toDelegate();
  auto d3 = v3.toDelegate();
  auto f1 = ldc.dynamic_compile.bind(&foo, d1, d2);
  
  auto d4 = f1.toDelegate();
  
  auto f2 = ldc.dynamic_compile.bind(&bar, d3, d4);

  compileDynamicCode(settings);

  assert(2003 == f1());
  assert(3006 == f2());
  assert(indexOf(dump.data, "2003") != -1);
  assert(indexOf(dump.data, "3006") != -1);
}
