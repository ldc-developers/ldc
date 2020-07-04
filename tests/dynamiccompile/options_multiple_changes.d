
// RUN: %ldc -enable-dynamic-compile -run %s

import std.stdio;
import std.array;
import std.string;
import ldc.attributes;
import ldc.dynamic_compile;

@dynamicCompile int foo()
{
  int* i = new int;
  *i = 42;
  return *i;
}

void main(string[] args)
{
  foreach (bool add_opt; [false, true, false, true, false])
  {
    auto dump = appender!string();
    CompilerSettings settings;
    settings.optLevel = 3;
    settings.dumpHandler = (DumpStage stage, in char[] str)
    {
      if (DumpStage.OptimizedModule == stage)
      {
        write(str);
        dump.put(str);
      }
    };
    writeln("===========================================");
    if (add_opt)
    {
      auto res = setDynamicCompilerOptions(["-disable-gc2stack"]);
      assert(res);
    }
    else
    {
      auto res = setDynamicCompilerOptions([]);
      assert(res);
    }
    compileDynamicCode(settings);
    writeln();
    writeln("===========================================");
    stdout.flush();

    if (add_opt)
    {
      assert(count(dump.data, "_d_allocmemoryT") > 0);
    }
    else
    {
      assert(count(dump.data, "_d_allocmemoryT") == 0);
    }
  }
}
