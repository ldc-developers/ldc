module bug60;
extern(C) int printf(char*, ...);

void func(T...)(T t)
{
    foreach(v;t) {
        if (v.length) {
            foreach(i;v) {
                printf("%d\n", i);
            }
        }
    }
}
void main()
{
    auto a = [1,2,3];
    func(a);
}
