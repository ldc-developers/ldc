module bug44;

pragma(LLVM_internal, "notypeinfo")
struct rgb
{
    long l;
}

void foo()
{
}

rgb test() {
  scope(exit) foo();
  return rgb();
}

void main()
{
    rgb r = test();
}
