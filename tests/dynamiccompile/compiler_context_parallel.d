
// RUN: %ldc -enable-dynamic-compile -run %s

import std.parallelism;
import std.range;
import ldc.attributes;
import ldc.dynamic_compile;

@dynamicCompile
{
int foo(int a, int b, int c)
{
  return a + b + c;
}

int bar(int a, int b, int c)
{
  return a + b + c;
}
}

void main(string[] args)
{
  foreach (i; parallel(iota(16)))
  {
    auto context = createCompilerContext();
    assert(context !is null);
    scope(exit) destroyCompilerContext(context);

    auto f = ldc.dynamic_compile.bind(context, &foo, 1,2,3);
    auto b = ldc.dynamic_compile.bind(context, &bar, 4,5,6);

    CompilerSettings settings;
    settings.optLevel = 3;
    compileDynamicCode(context, settings);

    assert(6 == f());
    assert(15 == b());
  }
}
