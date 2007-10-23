module enum2;

void main()
{
    enum E {
        A,B
    }
    E e = E.B;
    assert(e == E.B);
}
