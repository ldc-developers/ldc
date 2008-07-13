module tangotests.aa2_1;

int main()
{
    int[cdouble] x;
    cdouble d=22.0+0.0i;
    x[d] = 44;
    if(44 != x[d]){
            assert(0);
    }
    return 0;
}
