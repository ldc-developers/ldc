extern(C):

void* realloc(void*,size_t);
void free(void*);

void* _d_realloc(void* ptr, size_t n)
{
    return realloc(ptr, n);
}

void _d_free(void* ptr)
{
    free(ptr);
}
