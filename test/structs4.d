module structs4;

struct S{
    int a;
    T t;
}

struct T{
    int b;
    U u;
}

struct U{
    int c;
}

void main()
{
    S s;
    s.a = 3;
    s.t = T.init;
    s.t.b = 4;
    s.t.u = U.init;
    s.t.u.c = 5;
    {assert(s.t.u.c == 5);}
}
