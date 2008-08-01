module mini.real1;

extern(C)
{
real tanl(real x);
int printf(char*, ...);
}

void main()
{
    real ans = tanl(0.785398163398);
    printf("%llf\n", ans);
    assert(ans > 0.999 && ans < 1.001);
}
