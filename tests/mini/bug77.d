module bug77;
import tango.stdc.string;
void main()
{
    size_t len;
    void func2()
    {
        char* prefix = "";

        void func()
        {
        len = strlen(prefix);
        assert(len == 0);
        }

        func();
    }
    func2();
}
