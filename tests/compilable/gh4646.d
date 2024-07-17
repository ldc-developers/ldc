// RUN: %ldc -c -preview=bitfields %s

struct BitField {
    uint _0 : 1;
    uint _1 : 1;
    uint _2 : 1;
    uint _3 : 1;
    uint _4 : 1;
    uint _5 : 1;
    uint _6 : 1;
    uint _7 : 1;
    uint _8 : 1;
    uint _9 : 1;
    uint _10 : 1;
    uint _11 : 1;
    uint _12 : 1;
    uint _13 : 1;
    uint _14 : 1;
    uint _15 : 1;
    uint _16 : 1;
}

static assert(BitField.sizeof == 4);
static assert(BitField.alignof == 4);

struct Foo {
    BitField bf;
}

static assert(Foo.sizeof == 4);
static assert(Foo.alignof == 4);
