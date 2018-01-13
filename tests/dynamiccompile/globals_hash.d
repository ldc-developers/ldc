
// RUN: %ldc -enable-dynamic-compile -run %s

import ldc.attributes;
import ldc.dynamic_compile;

@dynamicCompileConst __gshared int value = 0;

@dynamicCompile int foo()
{
  return value;
}

void main(string[] args)
{
  CompilerSettings settings;
  const(char)[][] ids;
  settings.saveCache = (in char[] id, in void[] data)
  {
    assert(data.length != 0);
    ids ~= id;
  };
  compileDynamicCode(settings);
  value = 3;
  compileDynamicCode(settings);
  value = 5;
  compileDynamicCode(settings);

  assert(ids.length == 3);
  assert(ids[0] != ids[1]);
  assert(ids[0] != ids[2]);
  assert(ids[1] != ids[2]);
}
