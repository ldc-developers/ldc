
// REQUIRES: atleast_llvm500
// RUN: %ldc -enable-dynamic-compile -run %s

import std.array;
import std.string;
import ldc.attributes;
import ldc.dynamic_compile;

__gshared int value = 32;

@dynamicCompile int foo()
{
  return value;
}

void main(string[] args)
{
  auto dump = appender!string();
  CompilerSettings settings;
  settings.dumpHandler = (DumpStage stage, in char[] str)
  {
    if (DumpStage.FinalAsm == stage)
    {
      dump.put(str);
    }
  };
  compileDynamicCode(settings);

  // Check function and variables names in asm
  assert(-1 != indexOf(dump.data, foo.mangleof));
  assert(-1 != indexOf(dump.data, value.mangleof));
}
