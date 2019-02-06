// RUN: %ldc -c %s

struct Table
{
    RCArray _elems;

    void update(int x)
    {
        auto e = _elems[0 .. x][x == 0 ? x : $];
    }
}

struct RCArray
{
    int[] _arr;

    inout opSlice(size_t, size_t) { return _arr; }
    const length() { return _arr.length; }
    const opDollar() { return length; }
}
