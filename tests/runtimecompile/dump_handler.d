
// RUN: %ldc -enable-runtime-compile -run %s

import ldc.attributes;
import ldc.runtimecompile;

@dynamicCompile int foo()
{
  return 5;
}

void main(string[] args)
{
  bool dumpHandlerCalled[4] = false;
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
  assert(dumpHandlerCalled[DumpStage.OriginalModule]);
  assert(dumpHandlerCalled[DumpStage.MergedModule]);
  assert(dumpHandlerCalled[DumpStage.OptimizedModule]);
  // asm dump is disabled for now
  //assert(dumpHandlerCalled[DumpStage.FinalAsm]);
  assert(progressHandlerCalled);
}
