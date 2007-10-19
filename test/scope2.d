module scope2;

void main()
{
    scope(exit) printf("exit\n");
    scope(failure) printf("failure\n");
    scope(success) printf("success\n");
}
