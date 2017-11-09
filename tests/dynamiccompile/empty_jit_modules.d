
// RUN: %ldc -enable-dynamic-compile -run %s

import ldc.dynamic_compile;

void main(string[] args)
{
  compileDynamicCode();
}
