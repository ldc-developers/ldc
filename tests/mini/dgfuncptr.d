void main()
{
  void foo() {}

  auto dg = &foo;

  if(dg.funcptr is null)
  { assert(0); }
}
