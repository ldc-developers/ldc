// RUN: %ldc -run %s

struct S
{
    int x;
    S* self;
    this(int x)
    {
        this.x = x;
        self = &this;
    }
    ~this()
    {
        assert(self is &this);
    }
}

S makeS()
{
    return S(2); // one form of RVO: <temporary>.ctor(2)
}

S nrvo()
{
    S ret = makeS();
    return ret;
}

S rvo()
{
    return makeS();
}

void main()
{
    auto s = makeS();
    auto nrvo = nrvo();
    auto rvo = rvo();
}
