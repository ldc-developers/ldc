struct V(T...) {
    T v;
}

alias V!(float, int) MyV;

void main()
{
    assert(MyV.sizeof == float.sizeof + int.sizeof);
    auto f = 3.75f;
    auto v = MyV(f, 3);
    assert(v.v[0] == 3.75f);
    assert(v.v[1] == 3);
}

