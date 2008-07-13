module stdiotest2;
import std.stdio;
void main()
{
    int[4] v = [1,2,3,4];
    {
        writefln("%s", v);
        {
            int[] dv = v;
            {writefln("%s", dv);}
        }
    }

    {
        writefln(v);
        {
            //int[] dv = v;
            //{writefln(dv);}
        }
    }
    //writefln(1,2,3,4.56,"hello",v);
}
