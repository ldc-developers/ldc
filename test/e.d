module e;

struct C
{
    float x=0,y=0;

    float dot(ref C b)
    {
        return x*b.x + y*b.y;
    }
}

void main()
{
    C a,b;
    a.x = 2;
    a.y = 6;
    b.x = 3;
    b.y = 5;
    float f = a.dot(b);
    printf("%f\n", f);
    assert(f == 36);
}
