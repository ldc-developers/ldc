/*******************************************************************************
  copyright:   Copyright (c) 2006 Juan Jose Comellas. All rights reserved
  license:     BSD style: $(LICENSE)
  author:      Juan Jose Comellas <juanjo@comellas.com.ar>
               Converted to use core.sync by Sean Kelly <sean@f4.ca>
*******************************************************************************/

private import tango.core.sync.Mutex;
private import tango.core.Exception;
private import tango.core.Thread;
private import tango.io.Stdout;
private import tango.text.convert.Integer;
debug (mutex)
{
    private import tango.util.log.Log;
    private import tango.util.log.ConsoleAppender;
    private import tango.util.log.DateLayout;
}


/**
 * Example program for the tango.core.sync.Mutex module.
 */
void main(char[][] args)
{
    debug (mutex)
    {
        Logger log = Log.getLogger("mutex");

        log.addAppender(new ConsoleAppender(new DateLayout()));

        log.info("Mutex test");
    }

    testNonRecursive();
    testLocking();
    testRecursive();
}

/**
 * Test that non-recursive mutexes actually do what they're supposed to do.
 *
 * Remarks:
 * Windows only supports recursive mutexes.
 */
void testNonRecursive()
{
    version (Posix)
    {
        debug (mutex)
        {
            Logger log = Log.getLogger("mutex.non-recursive");
        }

        Mutex   mutex = new Mutex(Mutex.Type.NonRecursive);
        bool    couldLock;

        try
        {
            mutex.lock();
            debug (mutex)
                log.trace("Acquired mutex");
            couldLock = mutex.tryLock();
            if (couldLock)
            {
                debug (mutex)
                {
                    log.trace("Re-acquired mutex");
                    log.trace("Releasing mutex");
                }
                mutex.unlock();
            }
            else
            {
                debug (mutex)
                    log.trace("Re-acquiring the mutex failed");
            }
            debug (mutex)
                log.trace("Releasing mutex");
            mutex.unlock();
        }
        catch (SyncException e)
        {
            Stderr.formatln("Sync exception caught when testing non-recursive mutexes:\n{0}\n", e.toString());
        }
        catch (Exception e)
        {
            Stderr.formatln("Unexpected exception caught when testing non-recursive mutexes:\n{0}\n", e.toString());
        }

        if (!couldLock)
        {
            debug (mutex)
                log.info("The non-recursive Mutex test was successful");
        }
        else
        {
            debug (mutex)
            {
                log.error("Non-recursive mutexes are not working: "
                          "Mutex.tryAcquire() did not fail on an already acquired mutex");
                assert(false);
            }
            else
            {
                assert(false, "Non-recursive mutexes are not working: "
                              "Mutex.tryAcquire() did not fail on an already acquired mutex");
            }
        }
    }
}

/**
 * Create several threads that acquire and release a mutex several times.
 */
void testLocking()
{
    const uint MaxThreadCount   = 10;
    const uint LoopsPerThread   = 1000;

    debug (mutex)
    {
        Logger log = Log.getLogger("mutex.locking");
    }

    Mutex   mutex = new Mutex();
    uint    lockCount = 0;

    void mutexLockingThread()
    {
        try
        {
            for (uint i; i < LoopsPerThread; i++)
            {
                synchronized (mutex)
                {
                    lockCount++;
                }
            }
        }
        catch (SyncException e)
        {
            Stderr.formatln("Sync exception caught inside mutex testing thread:\n{0}\n", e.toString());
        }
        catch (Exception e)
        {
            Stderr.formatln("Unexpected exception caught inside mutex testing thread:\n{0}\n", e.toString());
        }
    }

    ThreadGroup group = new ThreadGroup();
    Thread      thread;
    char[10]    tmp;

    for (uint i = 0; i < MaxThreadCount; i++)
    {
        thread = new Thread(&mutexLockingThread);
        thread.name = "thread-" ~ format(tmp, i);

        debug (mutex)
            log.trace("Created thread " ~ thread.name);
        thread.start();

        group.add(thread);
    }

    debug (mutex)
        log.trace("Waiting for threads to finish");
    group.joinAll();

    if (lockCount == MaxThreadCount * LoopsPerThread)
    {
        debug (mutex)
            log.info("The Mutex locking test was successful");
    }
    else
    {
        debug (mutex)
        {
            log.error("Mutex locking is not working properly: "
                      "the number of times the mutex was acquired is incorrect");
            assert(false);
        }
        else
        {
            assert(false,"Mutex locking is not working properly: "
                         "the number of times the mutex was acquired is incorrect");
        }
    }
}

/**
 * Test that recursive mutexes actually do what they're supposed to do.
 */
void testRecursive()
{
    const uint LoopsPerThread   = 1000;

    debug (mutex)
    {
        Logger log = Log.getLogger("mutex.recursive");
    }

    Mutex   mutex = new Mutex;
    uint    lockCount = 0;

    try
    {
        for (uint i = 0; i < LoopsPerThread; i++)
        {
            mutex.lock();
            lockCount++;
        }
    }
    catch (SyncException e)
    {
        Stderr.formatln("Sync exception caught in recursive mutex test:\n{0}\n", e.toString());
    }
    catch (Exception e)
    {
        Stderr.formatln("Unexpected exception caught in recursive mutex test:\n{0}\n", e.toString());
    }

    for (uint i = 0; i < lockCount; i++)
    {
        mutex.unlock();
    }

    if (lockCount == LoopsPerThread)
    {
        debug (mutex)
            log.info("The recursive Mutex test was successful");
    }
    else
    {
        debug (mutex)
        {
            log.error("Recursive mutexes are not working: "
                      "the number of times the mutex was acquired is incorrect");
            assert(false);
        }
        else
        {
            assert(false, "Recursive mutexes are not working: "
                          "the number of times the mutex was acquired is incorrect");
        }
    }
}
