// bug #191

int[3] a = [0: 0, 2: 42, 1: 1];

void main()
{
  assert(a[0] == 0);
  assert(a[1] == 1); // fails!
  assert(a[2] == 42);
}
