
// RUN: %ldc -enable-dynamic-compile -run %s

import ldc.runtimecompile;

void main(string[] args)
{
  compileDynamicCode();
}
