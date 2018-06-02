
// RUN: %ldc -enable-dynamic-compile -run %s

import std.stdio;
import std.array;
import std.string;
import ldc.attributes;
import ldc.dynamic_compile;

__gshared int value = 32;

@dynamicCompile int foo()
{
  return value;
}

@dynamicCompile int bar()
{
  return 7;
}

@dynamicCompile int baz()
{
  return 8;
}

void main(string[] args)
{
  auto dump = appender!string();
  CompilerSettings settings;
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

  // Check function and variables names in asm
  assert(1 == count(dump.data, foo.mangleof));
  assert(1 == count(dump.data, bar.mangleof));
  assert(1 == count(dump.data, baz.mangleof));
  assert(count(dump.data, value.mangleof) > 0);
  assert(1 == count(dump.data, "7"));
  assert(1 == count(dump.data, "8"));
}
