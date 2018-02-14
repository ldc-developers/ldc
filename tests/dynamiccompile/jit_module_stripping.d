
// RUN: %ldc -enable-dynamic-compile -run %s

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

class Foo()
{
  @dynamicCompile int foo()
  {
    return 43;
  }
}


void main(string[] args)
{
  auto dump = appender!string();
  CompilerSettings settings;
  settings.dumpHandler = (DumpStage stage, in char[] str)
  {
    if (DumpStage.OriginalModule == stage)
    {
      dump.put(str);
    }
  };
  compileDynamicCode(settings);

  assert(0 == count(dump.data, "thunk"));
}
