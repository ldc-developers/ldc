module mini.norun_debug10;

struct Vec2
{
    float x,y;
}

void main()
{
    Vec2 v2;
    char[] str = "hello";
    int i = void;
    float f = 3.14;
    func(v2, v2, str, i, f);
}

void func(Vec2 v2, ref Vec2 rv2, char[] str, out int i, ref float f)
{
    int* fail;
    *fail = 0;
}
