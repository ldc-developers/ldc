// Don't make any changes/additions to this file without consulting Github issue 1638 first.

module switch_ICE_gh1638_bar;

struct S(T)
{
    auto fun = (T a) {
        T r;
        switch (a)
        {
        case 1:
            r = 1;
            break;
        default:
            return 0;
        }
        return r * 2;
    };
}

alias T = S!int;

void f(int a)
{
    int r = T().fun(a);
}
