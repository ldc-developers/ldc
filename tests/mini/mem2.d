module tangotests.mem2;

void main()
{
    int* ip = new int;
    assert(*ip == 0);
    *ip = 4;
    assert(*ip == 4);
    delete ip;
    assert(ip is null);
}
