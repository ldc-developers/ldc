module tangotests.debug1;

void main()
{
    int* ptr;

    // all these should be inspectable
    int i = 1;
    int j = 2;
    long k = 3;
    float l = 4.521;
    ubyte m = 5;

    *ptr = 0;//cast(int)(i+j+k+l+m);
}

extern(C) int printf(char*, ...);
