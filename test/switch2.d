module switch2;

void main()
{
    int i = 2;
    switch(i)
    {
    case 0: assert(0);
    case 1,2: printf("hello world\n"); break;
    default: assert(0);
    }
}
