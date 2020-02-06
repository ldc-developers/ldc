
// RUN: %ldc -enable-dynamic-compile -run %s

import std.array;
import std.string;
import ldc.attributes;
import ldc.dynamic_compile;

@dynamicCompile int foo()
{
  return 42;
}

void main(string[] args)
{
  auto dump = appender!string();
  auto res = setDynamicCompilerOptions(["-invalid_option"], (in char[] str)
  {
    dump.put(str);
  });
  assert(!res);
  assert(dump.data.length > 0);
}
