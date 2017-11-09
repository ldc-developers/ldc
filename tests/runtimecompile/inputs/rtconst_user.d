module inputs.rtconst_user;

import inputs.rtconst_owner;

import ldc.attributes;

@dynamicCompile int getValue()
{
  return 10 + value;
}
