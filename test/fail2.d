module fail2;

void a()
{
    b();
}

void b()
{
    c();
}

void c()
{
    d();
}

void d()
{
    int* ip;
    int i = *ip;
}

void main()
{
    a();
}
