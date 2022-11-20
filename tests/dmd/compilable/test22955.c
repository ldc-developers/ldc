// https://issues.dlang.org/show_bug.cgi?id=22955

_Static_assert( sizeof(void*) < 8 || _Alignof(__uint128_t) == 16, "" );
