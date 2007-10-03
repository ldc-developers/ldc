module switch1;

void main()
{
    int i = 2;
    int r;
    switch (i)
    {
    case 1: r+=1; break;
    case 2: r-=2;
    case 3: r=3; break;
    default: r=-1;
    }
    assert(r == 3);
}
