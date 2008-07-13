module bug80;
extern(C) int printf(char*, ...);

void main()
{
    byte b = 10;
    int i = b += 2;
    printf("byte=%d int=%d\n", b, i);
    assert(b == 12);
    assert(i == 12);
}
