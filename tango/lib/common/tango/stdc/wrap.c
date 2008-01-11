#include <errno.h>


int getErrno()
{
    return errno;
}


int setErrno( int val )
{
    errno = val;
    return val;
}
