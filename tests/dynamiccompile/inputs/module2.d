module inputs.module2;

import inputs.module3;

import ldc.attributes;

@dynamicCompile int get()
{
  return 4 + inputs.module3.get1() + inputs.module3.get2();
}
