module mini.atomic1;

pragma(intrinsic, "llvm.atomic.swap.i#.p0i#")
    T atomic_swap(T)(T* ptr, T val);

void main()
{
    int i = 42;
    int j = atomic_swap(&i, 43);
    assert(j == 42);
    assert(i == 43);
}
