module bug21;
extern(C) int printf(char*, ...);

void main()
{
    int i = 42;
    auto a = {
        int j = i*2;
        auto b = {
            return j;
        };
        return b();
    };
    int i2 = a();
    printf("%d\n", i2);
    assert(i2 == i*2);
}
