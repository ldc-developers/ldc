
// RUN: %ldc -enable-dynamic-compile -run %s save %t
// RUN: %ldc -enable-dynamic-compile -run %s load %t

import ldc.attributes;
import ldc.dynamic_compile;
import std.file;

@dynamicCompile int foo()
{
  return 42;
}

void main(string[] args)
{
  assert(args.length == 3);
  auto prefix = args[2];
  if (args[1] == "save")
  {
    CompilerSettings settings;
    settings.saveCache = (in char[] id, in void[] data)
    {
      assert(id.length != 0);
      assert(data.length != 0);
      std.file.write(prefix ~ id, data);
    };
    compileDynamicCode(settings);
    assert(foo() == 42);
  }
  else if (args[1] == "load")
  {
    CompilerSettings settings;
    settings.loadCache = (in char[] id, in void delegate(in void[]) sink)
    {
      assert(id.length != 0);
      assert(std.file.exists(prefix ~ id));
      auto data = std.file.read(prefix ~ id);
      assert(data.length != 0);
      sink(data);
    };
    compileDynamicCode(settings);
    assert(foo() == 42);
  }
  else
  {
    assert(false);
  }
  
}
