int[10] test;
int* t = &test[3];

void main()
{
    assert(t is &test[3]);
}
