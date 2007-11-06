module bug51;

import std.stdint;

union in6_addr
{
    private union _in6_u_t
    {
        uint8_t[16] u6_addr8;
        uint16_t[8] u6_addr16;
        uint32_t[4] u6_addr32;
    }
    _in6_u_t in6_u;

    uint8_t[16] s6_addr8;
    uint16_t[8] s6_addr16;
    uint32_t[4] s6_addr32;
}


const in6_addr IN6ADDR_ANY = { s6_addr8: [0] };
const in6_addr IN6ADDR_LOOPBACK = { s6_addr8: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] };

void main()
{
}
