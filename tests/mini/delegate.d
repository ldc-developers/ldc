class C { void foo() {} }

void main()
{
  C c = new C;
  void delegate() dlg = &c.foo;

  assert(dlg);
}
