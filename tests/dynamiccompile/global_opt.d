
// RUN: %ldc -enable-dynamic-compile -run %s

import std.array;
import std.string;
import ldc.attributes;
import ldc.dynamic_compile;

@dynamicCompileConst __gshared int value = 0;

@dynamicCompile int foo()
{
  return (value + 5) / 2;
}

void main(string[] args)
{
  auto dump = appender!(char[])();
  CompilerSettings settings;
  settings.optLevel = 2;
  settings.dumpHandler = (DumpStage stage, in char[] str)
  {
    if (DumpStage.FinalAsm == stage)
    {
      dump.put(str);
    }
  };
  value = 7;
  compileDynamicCode(settings);

  // Function return value will be reduced to constant
  // search for this value in asm
  assert(indexOf(dump.data, "6") != -1);
  assert(foo() == 6);

  dump.clear();

  value = 3;
  compileDynamicCode(settings);
  assert(indexOf(dump.data, "4") != -1);
  assert(foo() == 4);
}
