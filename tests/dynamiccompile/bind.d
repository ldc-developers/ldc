
// RUN: %ldc -enable-dynamic-compile -run %s

import ldc.attributes;
import ldc.dynamic_compile;

@dynamicCompile int foo(int a, int b, int c)
{
  return a + b + c;
}

void main(string[] args)
{
  CompilerSettings settings;
  settings.dumpHandler = (DumpStage stage, in char[] str)
  {
    if (DumpStage.OriginalModule == stage)
    {
      import std.stdio;
      //write(str);
      //stdout.flush();
    }
  };

  auto f1 = ldc.dynamic_compile.bind(&foo, placeholder, placeholder, placeholder);
  auto f2 = ldc.dynamic_compile.bind(&foo, 1, placeholder, 3);
  auto f3 = ldc.dynamic_compile.bind(&foo, 1, 2, 3);
  compileDynamicCode(settings);
  //h(2);
  assert(false);
}
