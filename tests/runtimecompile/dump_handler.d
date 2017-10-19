
// RUN: %ldc -enable-runtime-compile -run %s

import ldc.attributes;
import ldc.runtimecompile;

@runtimeCompile int foo()
{
  return 5;
}

void main(string[] args)
{
  bool dumpHandlerCalled[3] = false;
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
  assert(dumpHandlerCalled[DumpStage.OriginalIR]);
  assert(dumpHandlerCalled[DumpStage.OptimizedIR]);
  assert(dumpHandlerCalled[DumpStage.FinalAsm]);
  assert(progressHandlerCalled);
}
