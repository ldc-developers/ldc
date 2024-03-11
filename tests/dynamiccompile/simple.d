
// RUN: %ldc -enable-dynamic-compile -run %s

import std.stdio;
import ldc.attributes;
import ldc.dynamic_compile;

version(LDC_DynamicCompilation)
{
}
else
{
static assert(false, "LDC_DynamicCompilation is not defined");
}

@dynamicCompile int foo()
{
  return 5;
}

@dynamicCompile int bar()
{
  return foo() + 7;
}

@dynamicCompile void baz()
{
  // has regressed with Phobos v2.108, dragging in unsupported inline asm
  //writeln("baz");
}

@dynamicCompile int bzz(int a, int b)
{
  return a + b;
}

void main(string[] args)
{
  void run(CompilerSettings settings)
  {
    compileDynamicCode(settings);
    assert(5 == foo());
    assert(12 == bar());
    baz();
    int function() fptr = &bar;
    assert(12 == fptr());
    assert(15 == bzz(7, 8));
  }

  foreach(i;0..4)
  {
    CompilerSettings settings;
    settings.optLevel = i;
    run(settings);
  }
  foreach(i;0..3)
  {
    CompilerSettings settings;
    settings.sizeLevel = i;
    run(settings);
  }
}
