
static foocalled = false;
static barcalled = false;
void foo() { foocalled = true; }
void bar() { barcalled = true; }

void f(bool b)
{
  return b ? foo() : bar();
}

void main()
{
  f(true);
  assert(foocalled && !barcalled);
  f(false);
  assert(foocalled && barcalled);
}
