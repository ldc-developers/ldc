extern(C):

void exit(int);
int printf(char*,...);

void _d_assert(bool cond, uint line, char[] msg)
{
    if (!cond) {
        printf("Aborted(%u): %.*s\n", line, msg.length, msg.ptr);
        exit(1);
    }
}
