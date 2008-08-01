module mini.sextzext;

void main()
{
    byte sb  = sextreturn1();
    short ss = sextreturn2();
    assert(ss == -2);
    assert(sb == -2);
    assert(sextparam1(-42) == -42);
    assert(sextparam2(-42) == -42);

    ubyte ub  = zextreturn1();
    ushort us = zextreturn2();
    assert(ub == 2);
    assert(us == 2);
    assert(zextparam1(42) == 42);
    assert(zextparam2(42) == 42);

    assert(getchar() == 'a');
    assert(passchar('z') == 'z');

}

byte sextreturn1()
{
    return -2;
}
short sextreturn2()
{
    return -2;
}

ubyte zextreturn1()
{
    return 2;
}
ushort zextreturn2()
{
    return 2;
}

byte sextparam1(byte b)
{
    return b;
}
short sextparam2(short s)
{
    return s;
}

ubyte zextparam1(ubyte b)
{
    return b;
}
ushort zextparam2(ushort s)
{
    return s;
}

char getchar()
{
    return 'a';
}

char passchar(char ch)
{
    return ch;
}
