module mini.nested16;

void main()
{
    int idx = 123;
    int func(int* idp)
    {
        void foo()
        {
            void bar(int* idp)
            {
                auto c = new class
                {
                    void mem()
                    {
                        scope(exit) ++*idp;
                    }
                };
                auto dg = () {
                    c.mem();
                };
                dg();
            }
            bar(idp);
            ++*idp;
        }
        foo();
        return ++*idp;
    }
    assert(func(&idx) == 126);
}
