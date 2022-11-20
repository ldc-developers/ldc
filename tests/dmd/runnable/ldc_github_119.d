alias ubyte[2] buf;

buf initUsingValue() { buf x = 0; return x; }
buf initDefault() { return buf.init; } // can just as easily replace buf for typeof(return)
buf initUsingValue2() { buf x = 42; return x; }
buf initUsingLiteral() { return [ 4, 8 ]; }

void main()
{
    buf x = initUsingValue();
    assert(x[0] == 0 && x[1] == 0);

    x = initDefault();
    assert(x[0] == 0 && x[1] == 0);

    x = initUsingValue2();
    assert(x[0] == 42 && x[1] == 42);

    x = initUsingLiteral();
    assert(x[0] == 4 && x[1] == 8);
}