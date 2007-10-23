module enum1;

void main()
{
    enum {
        HELLO,
        WORLD
    }

    assert(HELLO == 0);
    assert(WORLD == 1);
}
