module floatcmp;
extern(C) int printf(char*, ...);

void eq()
{
    float _3 = 3;
    assert(!(_3 == 4));
    assert(!(_3 == 2));
    assert(_3 == 3);
    assert(!(_3 == float.nan));
}

void neq()
{
    float _3 = 3;
    assert(_3 != 4);
    assert(_3 != 2);
    assert(!(_3 != 3));
    assert(_3 != float.nan);
}

void gt()
{
    float _3 = 3;
    assert(_3 > 2);
    assert(!(_3 > 4));
    assert(!(_3 > 3));
    assert(!(_3 > float.nan));
}

void ge()
{
    float _3 = 3;
    assert(_3 >= 2);
    assert(!(_3 >= 4));
    assert(_3 >= 3);
    assert(!(_3 >= float.nan));
}

void lt()
{
    float _3 = 3;
    assert(_3 < 4);
    assert(!(_3 < 2));
    assert(!(_3 < 3));
    assert(!(_3 < float.nan));
}

void le()
{
    float _3 = 3;
    assert(_3 <= 4);
    assert(!(_3 <= 2));
    assert(_3 <= 3);
    assert(!(_3 <= float.nan));
}

void uno()
{
    float _3 = 3;
    assert(!(_3 !<>= 2));
    assert(!(_3 !<>= 3));
    assert(!(_3 !<>= 4));
    assert(_3 !<>= float.nan);
}

void lg()
{
    float _3 = 3;
    assert(_3 <> 4);
    assert(_3 <> 2);
    assert(!(_3 <> 3));
    assert(!(_3 <> float.nan));
}

void lge()
{
    float _3 = 3;
    assert(_3 <>= 4);
    assert(_3 <>= 2);
    assert(_3 <>= 3);
    assert(!(_3 <>= float.nan));
}

void ugt()
{
    float _3 = 3;
    assert(_3 !<= 2);
    assert(!(_3 !<= 4));
    assert(!(_3 !<= 3));
    assert(_3 !<= float.nan);
}

void uge()
{
    float _3 = 3;
    assert(_3 !< 2);
    assert(!(_3 !< 4));
    assert(_3 !< 3);
    assert(_3 !< float.nan);
}

void ult()
{
    float _3 = 3;
    assert(_3 !>= 4);
    assert(!(_3 !>= 2));
    assert(!(_3 !>= 3));
    assert(_3 !>= float.nan);
}

void ule()
{
    float _3 = 3;
    assert(_3 !> 4);
    assert(!(_3 !> 2));
    assert(_3 !> 3);
    assert(_3 !> float.nan);
}

void ueq()
{
    float _3 = 3;
    assert(!(_3 !<> 2));
    assert(!(_3 !<> 4));
    assert(_3 !<> 3);
    assert(_3 !<> float.nan);
}

void main()
{
    printf("floating point comparison test\n");
    
    eq();
    neq();
    gt();
    ge();
    lt();
    le();
    uno();
    lg();
    lge();
    ugt();
    uge();
    ult();
    ule();
    ueq();
    
    printf("  SUCCESS\n");
}

/+
void gt()
{
    float _3 = 3;
    assert();
    assert();
    assert();
    assert();
}
+/
