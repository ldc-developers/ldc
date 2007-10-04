module foreach1;
import std.stdio;

void main()
{
    static arr = [1,2,3,4,5];

    writef("forward");
    foreach(v;arr) {
        writef(' ',v);
    }
    writef("\nreverse");
    foreach_reverse(v;arr) {
        writef(' ',v);
    }
    writef("\n");
}
