module bug9;
struct rgb
{
  ubyte[3] values;
  rgb average(rgb other)
  {
    rgb res;
    foreach (id, ref v; res.values) v=(values[id]+other.values[id])/2;
    return res;
  }
  void print()
  {
    printf("[%d,%d,%d]\n", values[0], values[1], values[2]);
  }
}

void main()
{
    rgb a,b;
    a.values[0] = 10;
    a.values[1] = 20;
    a.values[2] = 30;
    b.values[0] = 30;
    b.values[1] = 20;
    b.values[2] = 10;
    rgb avg = a.average(b);
    avg.print();
    assert(avg.values[0] == 20);
    assert(avg.values[1] == 20);
    assert(avg.values[2] == 20);
}

