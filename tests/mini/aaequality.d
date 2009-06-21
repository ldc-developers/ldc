void test(K,V)(K k1, V v1, K k2, V v2, K k3, V v3)
{
    V[K] a, b;
    a[k1] = v1;
    a[k2] = v2;
    assert(a != b);
    assert(b != a);
    assert(a == a);
    assert(b == b);
    
    b[k1] = v1;
    assert(a != b);
    assert(b != a);

    b[k2] = v2;
    assert(a == b);
    assert(b == a);
    
    b[k1] = v2;
    assert(a != b);
    assert(b != a);
    
    b[k1] = v1;
    b[k2] = v3;
    assert(a != b);
    assert(b != a);
    
    b[k2] = v2;
    b[k3] = v3;
    assert(a != b);
    assert(b != a);
}

void main()
{
    test!(int,int)(1, 2, 3, 4, 5, 6);
    test!(char[],int)("abc", 2, "def", 4, "geh", 6);
    test!(int,char[])(1, "abc", 2, "def", 3, "geh");
    test!(char[],char[])("123", "abc", "456", "def", "789", "geh");
    
    Object a = new Object, b = new Object, c = new Object;
    test!(Object, Object)(a, a, b, b, c, c);
    
    int[int] a2 = [1:2, 2:3, 3:4];
    int[int] b2 = [1:2, 2:5, 3:4];
    int[int] c2 = [1:2, 2:3];
    test!(int,int[int])(1,a2, 2,b2, 3,c2);
}