
// RUN: %ldc -enable-dynamic-compile -run %s

import std.stdio;
import std.array;
import std.string;
import ldc.attributes;
import ldc.dynamic_compile;

__gshared int value = 32;


@dynamicCompile int foo()
{
  return 42;
}

@dynamicCompile int bar()
{
  return foo();
}

void fun(int function())
{
}

@dynamicCompile void baz()
{
  fun(&foo);
}

@dynamicCompile int bzz()
{
  return value;
}

class Foo()
{
  @dynamicCompile int foo()
  {
    return 43;
  }
}

struct UnusedStruct1
{
  int i;
  int j;
};

struct UnusedStruct2
{
  UnusedStruct1 f;
  int k;
}

void unusedfunc1(UnusedStruct2)
{
}

void unusedfunc2(int function())
{
}

__gshared int unusedvalue = 32;

void main(string[] args)
{
  auto dump = appender!string();
  CompilerSettings settings;
  settings.dumpHandler = (DumpStage stage, in char[] str)
  {
    if (DumpStage.OriginalModule == stage)
    {
      write(str);
      dump.put(str);
    }
  };
  writeln("===========================================");
  compileDynamicCode(settings);
  writeln("===========================================");

  assert(0 == count(dump.data, "thunk"));
  assert(0 == count(dump.data, "unusedvalue"));
  assert(0 == count(dump.data, "unusedfunc1"));
  assert(0 == count(dump.data, "unusedfunc2"));

  // TODO: these symbols is pulled by llvm.ldc.typeinfo
  //assert(0 == count(dump.data, "UnusedStruct1"));
  //assert(0 == count(dump.data, "UnusedStruct2"));
}
