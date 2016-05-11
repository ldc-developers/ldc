module inputs.link_bitcode_input;

extern(C) int return_seven() {
  return 7;
}

import inputs.link_bitcode_import;
void bar()
{
    SomeStrukt r = {1};
    takeStrukt(&r);
}
