module app;
import lib;

void func()
{
    int* ip;
    int i = lib_templ_func(ip);
}

int main(char[][] args)
{
    func();
    return 0;
}
