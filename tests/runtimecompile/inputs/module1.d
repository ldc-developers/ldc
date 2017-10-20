module inputs.module1;

import inputs.module3;

import ldc.attributes;

@runtimeCompile int get()
{
  return 3 + inputs.module3.get1() + inputs.module3.get2();
}
