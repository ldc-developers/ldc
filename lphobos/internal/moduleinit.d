module internal.moduleinit;

private alias extern(C) void function() fptr_t;

extern(C):

fptr_t* _d_get_module_ctors();
fptr_t* _d_get_module_dtors();

void _d_run_module_ctors()
{
    auto p = _d_get_module_ctors();
    while(*p) {
        (*p++)();
    }
}

void _d_run_module_dtors()
{
    auto p = _d_get_module_dtors();
    while(*p) {
        (*p++)();
    }
}
