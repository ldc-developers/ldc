module mini.ptrcond;

void main()
{
    char[4]* cp;
    void* vp = &cp;
    assert(cp < vp);
}
