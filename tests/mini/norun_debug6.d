module mini.norun_debug6;

void main()
{
    char[] str = "hello world :)";

    int* fail = cast(int*) 1;
    *fail = 32;
}
