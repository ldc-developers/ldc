module tangotests.sync2;

class Lock
{
}

const Lock lock;

static this()
{
    lock = new Lock;
}

void main()
{
    size_t id;
    synchronized(lock) id = 2;
}
