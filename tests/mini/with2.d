struct bar {
    int bar;
}

void main() {
    bar Bar;
    with (Bar)
    {
        assert(Bar.bar == 0);
        void test()
        {
            bar ++;
        }
        test();
    }
    assert(Bar.bar == 1);
}
