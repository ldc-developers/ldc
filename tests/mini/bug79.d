module bug79;
import std.c.linux.linux;
void main()
{
    timespec ts; 
    ts.tv_nsec -= 1;
    //auto t = ts.tv_nsec - 1;
    //t -= 1;
}
