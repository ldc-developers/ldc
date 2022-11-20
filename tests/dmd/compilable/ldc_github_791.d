// DISABLED: LDC_not_x86

int crash()
{
  asm
  {
    naked;
    ret;
  }
}
