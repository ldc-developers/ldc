
// RUN: %ldc -enable-dynamic-compile -run %s

import std.array;
import std.stdio;
import std.string;
import ldc.attributes;
import ldc.dynamic_compile;

@dynamicCompile
{
int foo(int function() a, int function() b, int function() c)
{
  return a() + b() + c();
}

int bar(int delegate() a, int delegate() b, int delegate() c)
{
  return a() + b() + c();
}

int get1001()
{
  return 1001;
}

int get1002()
{
  return 1002;
}

int get1003()
{
  return 1003;
}
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

  @dynamicCompile
  int get1001d()
  {
    return 1001;
  }

  @dynamicCompile
  int get1002d()
  {
    return 1002;
  }

  @dynamicCompile
  int get1004d()
  {
    return 1004;
  }

  auto f = ldc.dynamic_compile.bind(&foo, &get1001, &get1002, &get1003);
  auto b = ldc.dynamic_compile.bind(&bar, &get1001d, &get1002d, &get1004d);

  compileDynamicCode(settings);

  assert(3006 == f());
  assert(3007 == b());
  assert(indexOf(dump.data, "3006") != -1);
  assert(indexOf(dump.data, "3007") != -1);
}
