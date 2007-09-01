void main()
{
    printf("Bitwise operations test\n");
    {   ushort a = 0xFFF0;
        ushort b = 0x0FFF;
        assert((a&b) == 0x0FF0);
        assert((a|b) == 0xFFFF);
        assert((a^b) == 0xF00F);
    }
    {   ubyte a = 0xFF;
        ulong b = 0xFFFF_FFFF_FFFF_FFF0;
        assert((a&b) == 0xF0);
    }
    {   ushort s = 0xFF;
        assert((s<<1) == s*2);
        assert((s>>>1) == s/2);
    }
    {   int s = -10;
        assert((s>>1) == -5);
        assert((s>>>1) != -5);
    }
    
    {   ushort a = 0xFFF0;
        ushort b = 0x0FFF;
        auto t = a;
        t &= b;
        assert(t == 0x0FF0);
        t = a;
        t |= b;
        assert(t == 0xFFFF);
        t = a;
        t ^= b;
        assert(t == 0xF00F);
    }
    {   ubyte a = 0xFF;
        ulong b = 0xFFFF_FFFF_FFFF_FFF0;
        a &= b;
        assert(a == 0xF0);
    }
    {   ushort s = 0xFF;
        auto t = s;
        t <<= 1;
        assert(t == (s*2));
        t = s;
        t >>>= 1;
        assert(t == s/2);
    }
    {   int s = -10;
        auto t = s;
        t >>= 1;
        assert(t == -5);
        t = s;
        t >>>= 1;
        assert(t != -5);
    }
    {   struct S
        {
            uint i;
            ulong l;
        };
        S s = S(1,4);
        auto a = s.i | s.l;
        assert(a == 5);
        s.i = 0xFFFF_00FF;
        s.l = 0xFFFF_FFFF_0000_FF00;
        s.l ^= s.i;
        assert(s.l == ulong.max);
        s.i = 0x__FFFF_FF00;
        s.l = 0xFF00FF_FF00;
        s.i &= s.l;
        assert(s.i == 0x00FF_FF00);
    }
        
    printf("  SUCCESS\n");
}
