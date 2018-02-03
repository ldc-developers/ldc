import core.thread;
import core.sys.posix.sys.mman;
import ldc.attributes;

// this should be true for most architectures
// (taken from core.thread)
version = StackGrowsDown;

enum stackSize = 4096;

// Simple method that causes a stack overflow
@optStrategy("none")
void stackMethod()
{
    // Over the stack size, so it overflows the stack
    int[stackSize/int.sizeof+100] x;
}

void main()
{
    auto test_fiber = new Fiber(&stackMethod, stackSize);

    auto getPrivateFiberField(string id)()
    {
        static size_t getIndex(string id)()
        {
            static foreach (i, field; Fiber.tupleof)
            {
                static if (field.stringof == id)
                    return i;
            }
            assert(0);
        }

        enum i = getIndex!id();
        return test_fiber.tupleof[i];
    }

    // allocate a page below (above) the fiber's stack to make stack overflows possible (w/o segfaulting)
    version (StackGrowsDown)
    {
        auto stackBottom = getPrivateFiberField!"m_pmem"();
        auto p = mmap(stackBottom - 8 * stackSize, 8 * stackSize,
                      PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        assert(p !is null, "failed to allocate page");
    }
    else
    {
        auto m_sz = getPrivateFiberField!"m_sz"();
        auto m_pmem = getPrivateFiberField!"m_pmem"();

        auto stackTop = m_pmem + m_sz;
        auto p = mmap(stackTop, 8 * stackSize,
                      PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        assert(p !is null, "failed to allocate page");
    }

    // the guard page should prevent a mem corruption by stack
    // overflow and cause a segfault instead (or generate SIGBUS on *BSD flavors)
    test_fiber.call();
}
