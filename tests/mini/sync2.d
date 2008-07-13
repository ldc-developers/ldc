module tangotests.sync2;

class Lock
{
}

Lock lock;

void main()
{
    size_t id;
    synchronized(lock) id = 2;
}
