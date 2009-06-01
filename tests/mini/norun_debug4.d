module mini.norun_debug4;

void main()
{
    char c = 'c';
    wchar wc = 'w';
    dchar dc = 'd';

    int* fail = cast(int*) 1;
    *fail = 32;
}
