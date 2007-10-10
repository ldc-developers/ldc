module arrays4;

void main()
{
    auto arr = new int[4];
    auto arrcat = arr ~ arr;
    assert(arrcat.length == arr.length * 2);
}

