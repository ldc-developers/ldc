
// RUN: %ldc -enable-runtime-compile -run %s

import ldc.runtimecompile;

void main(string[] args)
{
  compileDynamicCode();
}
