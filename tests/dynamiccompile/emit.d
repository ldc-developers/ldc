
// RUN: %ldc -enable-dynamic-compile -run %s

import std.stdio;
import std.array;
import std.string;
import ldc.attributes;
import ldc.dynamic_compile;

int foo()
{
  return 42;
}

@dynamicCompileEmit int bar()
{
  return 43;
}

@dynamicCompileEmit @dynamicCompile int baz()
{
  return 44;
}

void main(string[] args)
{
  assert(43 == bar());

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
  writeln();
  writeln("===========================================");
  stdout.flush();

  assert(44 == baz());

  // Check function name in original IR
  assert(count(dump.data, foo.mangleof) == 0);
  assert(count(dump.data, bar.mangleof) > 0);
  assert(count(dump.data, baz.mangleof) > 0);
}
