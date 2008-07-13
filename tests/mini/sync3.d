module tangotests.sync3;

void main()
{
    int i = foo();
}

int foo()
{
    int res;
    synchronized
    {
        scope(exit) res--;
        res++;
        return res;
    }
}
