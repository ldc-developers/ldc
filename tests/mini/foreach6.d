module foreach6;

struct S
{
    long l;
    float f;
}

void main()
{
    S[4] arr = void;
    foreach(i,v;arr) {
        v = S(i, i*2.5);
    }
}
