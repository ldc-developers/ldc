module b;

struct S
{
    int i;
    float[4] f;
}

void main()
{
    S s;
    int i = s.i;
    /*int* p = &s.i;
    *p = 42;
    printf("%d == %d\n", *p, s.i);

    float* f = &s.f[0];
    printf("%f == %f\n", *f, s.f[0]);
    *f = 3.1415;
    printf("%f == %f\n", *f, s.f[0]);
    s.f[0] = 123.456;
    printf("%f == %f\n", *f, s.f[0]);*/
}
