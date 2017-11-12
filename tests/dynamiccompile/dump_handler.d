
// RUN: %ldc -enable-dynamic-compile -run %s

import ldc.attributes;
import ldc.dynamic_compile;

@dynamicCompile int foo()
{
  return 5;
}

@dynamicCompile int bar(int i = 5)
{
  if(i > 0)
  {
    return bar(i - 1) + 1;
  }
  return 1;
}

@dynamicCompile int baz()
{
  int i = 0;
  foreach(j;1..4)
  {
    i += j * j;
  }
  return i;
}

void main(string[] args)
{
  bool[4] dumpHandlerCalled = false;
  bool progressHandlerCalled = false;
  CompilerSettings settings;

  settings.dumpHandler = (DumpStage stage, in char[] str)
  {
    dumpHandlerCalled[stage] = true;
  };
  settings.progressHandler = (in char[] desc, in char[] object)
  {
    progressHandlerCalled = true;
  };
  compileDynamicCode(settings);
  assert(5 == foo());
  assert(6 == bar());
  assert(14 == baz());
  assert(dumpHandlerCalled[DumpStage.OriginalModule]);
  assert(dumpHandlerCalled[DumpStage.MergedModule]);
  assert(dumpHandlerCalled[DumpStage.OptimizedModule]);
  assert(dumpHandlerCalled[DumpStage.FinalAsm]);
  assert(progressHandlerCalled);
}
