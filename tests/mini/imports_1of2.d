module imports_1of2;

import imports_2of2;

void main()
{
    assert(func() == 42);
    S s;
    s.l = 32;
    assert(s.l == 32);
}
