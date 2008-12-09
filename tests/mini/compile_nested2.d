void test(void delegate() spam)
{
    static void foo() // static is the problem
    {
        uint x;
        void peek() { x = 0; }
    }

    void bar()
    {
        spam();
    }
}
