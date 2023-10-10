// RUN: %ldc --O2 -run %s

void main()
{
    int count;
    foreach (i; 0..34)
    {
        auto flags = new bool[](1);
        if (flags[0] == false) count++;
            flags[] = true;
    }
    assert(count == 34);
}
