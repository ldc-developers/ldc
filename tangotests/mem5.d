module tangotests.mem5;

class SC
{
    int* ip;
    this()
    {
        ip = new int;
    }
    ~this()
    {
        delete ip;
    }
    void check()
    {
        assert(ip !is null);
    }
}

void main()
{
    scope sc = new SC;
    sc.check();
}
