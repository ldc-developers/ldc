
// RUN: %ldc -enable-dynamic-compile -run %s

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
  auto context1 = createCompilerContext();
  assert(context1 !is null);
  scope(exit) destroyCompilerContext(context1);

  auto context2 = createCompilerContext();
  assert(context2 !is null);
  scope(exit) destroyCompilerContext(context2);

  auto f = ldc.dynamic_compile.bind(context1, &foo, 1,2,3);
  auto b = ldc.dynamic_compile.bind(context2, &bar, 4,5,6);

  CompilerSettings settings;
  settings.optLevel = 3;

  compileDynamicCode(context1, settings);

  assert(f.isCallable());
  assert(!b.isCallable());
  assert(6 == f());

  compileDynamicCode(context2, settings);

  assert(f.isCallable());
  assert(b.isCallable());
  assert(6 == f());
  assert(15 == b());
}
