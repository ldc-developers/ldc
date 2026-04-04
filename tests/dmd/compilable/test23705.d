// https://issues.dlang.org/show_bug.cgi?id=23705

// DISABLED: win32

void main ()
{
  version (LDC_LLVM_22)
  {
    // regressed for 32-bit x86 with LLVM 22: 'error: 64-bit offset calculated but target is 32-bit'
  }
  else
  {
    ubyte [0x7fff_fffe] x;
  }
}
