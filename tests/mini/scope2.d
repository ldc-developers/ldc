module scope2;
extern(C) int printf(char*, ...);
void main()
{
    scope(exit) printf("exit\n");
    scope(failure) printf("failure\n");
    scope(success) printf("success\n");
}
