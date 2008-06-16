extern(C) int printf(char*, ...);

struct V2D(T)
{
    T x,y;

    T dot(ref V2D v)
    {
        return x*v.x + y*v.y;
    }

    V2D opAdd(ref V2D v)
    {
        return V2D(x+v.x, y+v.y);
    }
}

alias V2D!(float) V2Df;

void main()
{
    printf("V2D test\n");
    auto up = V2Df(0.0f, 1.0f);
    auto right = V2Df(1.0f, 0.0f);
    assert(up.dot(right) == 0.0f);
    auto upright = up + right;
    assert(upright.x == 1.0f && upright.y == 1.0f);
    auto copy = upright;
    copy.x++;
    assert(copy.x > upright.x);
    printf("  SUCCESS\n");
}
