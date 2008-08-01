module mini.nested14;

extern(C) int printf(char*, ...);

class C
{
  void foo()
  {
    void bar()
    {
       car();
    }

    bar();
  }

  void car()
  {
    printf("great\n");
  }
}

void main()
{
  scope c = new C;
  c.foo();
}
