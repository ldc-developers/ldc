import core.stdc.errno;

struct epoll_event  {
align(1):
    uint events;
    epoll_data_t data;
}

union epoll_data_t  {
    void *ptr;
    int fd;
    uint u32;
    ulong u64;
}

enum PollerEventType : int { a }

struct PollerEvent {
    PollerEventType type;
    int fd;
}

struct Epoll {
    ushort[1024] regFds;
    int epollFd = -1;
    epoll_event[32] events;
    PollerEvent[events.length * 4] pollerEvents;
}

struct Reactor {
    bool a = true;
    bool b;
    Epoll poll;
}

__gshared Reactor f;

Reactor* foo() {
    return &f;
}

void main() {}
