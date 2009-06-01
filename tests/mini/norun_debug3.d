module mini.norun_debug3;

void main()
{
    int i = 42;
    int* ip = &i;

    int* fail = cast(int*) 1;
    *fail = 0;
}
