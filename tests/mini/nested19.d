module mini.nested19;

void main()
{
    int i = 1;
    
    class C
    {
        int j = 2;
        void func()
        {
            int k = 3;
            
            void foo()
            {
                i = i+j+k;
            }
            
            foo();
        }
    }
    
    auto c = new C;
    c.func();
    
    assert(i == 6);
}
