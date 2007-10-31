module bug41;

void main()
{
    char[] a = "hello world";
    char* ap = a.ptr;
    size_t i = 5;
    char[] b = ap[0..i];
    assert(b == "hello");
}
