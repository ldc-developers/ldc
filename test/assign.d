module assign;

// this is taken from std.intrinsic from gdc

int mybtc(uint *p, uint bitnum)
{
    uint * q = p + (bitnum / (uint.sizeof*8));
    uint mask = 1 << (bitnum & ((uint.sizeof*8) - 1));
    int result = *q & mask;
    *q ^= mask;
    return result ? -1 : 0;
}

void main()
{
    uint i = 0xFFFF_FFFF;
    int r = mybtc(&i, 31);
    assert(r);
    assert(i == 0x7FFF_FFFF);
    r = mybtc(&i, 31);
    assert(!r);
    assert(i == 0xFFFF_FFFF);
}
