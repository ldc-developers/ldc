struct V(T...) {
    T v;
}

alias V!(Object, int) MyV;

void main()
{
    assert(MyV.sizeof == Object.sizeof + int.sizeof);
    auto o = new Object;
    auto v = MyV(o, 3);
    assert(v.v[0] is o);
    assert(v.v[1] == 3);
}

