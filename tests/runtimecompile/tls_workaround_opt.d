
// tls without workaround broken on all platforms
// just test ldc accept this option
// RUN: %ldc -enable-runtime-compile -runtime-compile-tls-workaround=0 -run %s
// RUN: %ldc -enable-runtime-compile -runtime-compile-tls-workaround=1 -run %s

import ldc.attributes;
import ldc.runtimecompile;

@dynamicCompile void foo()
{
}

void main(string[] args)
{
  compileDynamicCode();
  foo();
}
