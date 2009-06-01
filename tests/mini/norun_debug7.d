module mini.norun_debug7;

int gi;

void main()
{
    int* fail = cast(int*) 1;
    *fail = 0;
}
