module object;

extern(C) int _Dmain(char[][]);

extern(C) int main(int, char**)
{
    return _Dmain(null);
}
