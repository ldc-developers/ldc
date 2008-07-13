module structinit;

import structinit2;

struct S
{
    uint ui;
    float f;
    long l;
    real r;
}

S adef;

S a = { 1, 2.0f };
S b = { f:2.0f, l:42 };

Imp imp;

void main()
{
    //assert(a == S.init);
    //assert(b == S(0,3.14f,0,real.init));
}

