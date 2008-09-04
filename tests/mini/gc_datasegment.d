extern(C) void gc_collect();

class C
{
  int i = 42;
}

C data;

void main()
{
  data = new C;
  gc_collect();
  assert(data.i == 42);
}
