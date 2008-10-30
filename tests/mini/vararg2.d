module vararg2;

void func(...)
{
    assert(_arguments.length == 2);
    assert(_arguments[0] is typeid(int));
    int a = *cast(int*)_argptr;
    _argptr += size_t.sizeof;
    assert(_arguments[1] is typeid(int));
    a += *cast(int*)_argptr;
    _argptr += int.sizeof;
    assert(a == 3);
}

void main()
{
    func(1,2);
}
