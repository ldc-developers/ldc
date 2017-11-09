
// RUN: %ldc -enable-dynamic-compile -run %s

import ldc.attributes;
import ldc.dynamic_compile;

@dynamicCompile int foo()
{
  return 5;
}
 
int bar()
{
  return 7;
}

@dynamicCompile int baz()
{
  return foo() + bar();
}

alias fptr = int function();

@dynamicCompile
{
fptr get_foo_ptr()
{
  return &foo;
}

fptr get_bar_ptr()
{
  return &bar;
}

fptr get_baz_ptr()
{
  return &baz;
}
}

void main(string[] args)
{
  compileDynamicCode();
  assert(5  == foo());
  assert(7  == bar());
  assert(12 == baz());
  assert(5  == get_foo_ptr()());
  assert(7  == get_bar_ptr()());
  assert(12 == get_baz_ptr()());
}
