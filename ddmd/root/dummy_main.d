// A MASSIVE hack imho.
// See http://forum.dlang.org/post/caowrljxijchgmyyrtlr@forum.dlang.org
// This is needed on Windows to solve unresolved external symbol _deh_beg and _deh_end linker errors.
version (Windows)
{
    int main()
    {
        return 0;
    }
}
