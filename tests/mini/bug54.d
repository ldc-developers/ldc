module bug54;

extern(C) size_t strlen(char*);

// creating args for main
void d_main_args(size_t n, char** args, ref char[][] res)
{
    assert(res.length == n);
    foreach(i,v; args[0..n])
    {
        res[i] = v[0 .. strlen(v)];
    }
}

void main()
{
}
