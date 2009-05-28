module mini.norun_debug5;

void main()
{
    int i = 32;
    real r = 3.1415;
    real* p = &r;
    func(i,r,p);
}

void func(int i, real r, real* p)
{
    int* fail = cast(int*) 1;
    *fail = 666;
}
