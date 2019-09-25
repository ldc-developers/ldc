
// RUN: %ldc -enable-dynamic-compile -run %s

import std.array;
import std.string;
import std.stdio;
import ldc.attributes;
import ldc.dynamic_compile;

@dynamicCompileEmit int foo(int a, int b, bool flag)
{
  if (flag)
  {
    return a + b;
  }
  else
  {
    return a - b;
  }
}

void main(string[] args)
{
  foreach (val; [false, true])
  {
    auto dump = appender!(char[])();
    CompilerSettings settings;
    settings.optLevel = 3;
    settings.dumpHandler = (DumpStage stage, in char[] str)
    {
      if (DumpStage.FinalAsm == stage)
      {
        dump.put(str);
        write(str);
      }
    };
    auto f = ldc.dynamic_compile.bind(&foo, 123110, 3, val);
    auto d = f.toDelegate();

    compileDynamicCode(settings);

    if (val)
    {
      assert(indexOf(dump.data, "123113") != -1);
      assert(indexOf(dump.data, "123107") == -1);
      assert(f() == 123113);
      assert(d() == 123113);
    }
    else
    {
      assert(indexOf(dump.data, "123113") == -1);
      assert(indexOf(dump.data, "123107") != -1);
      assert(f() == 123107);
      assert(d() == 123107);
    }
  }
}
