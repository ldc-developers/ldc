/*******************************************************************************
  copyright:   Copyright (c) 2006 Juan Jose Comellas. All rights reserved
  license:     BSD style: $(LICENSE)
  author:      Juan Jose Comellas <juanjo@comellas.com.ar>
               Converted to use core.sync by Sean Kelly <sean@f4.ca>
*******************************************************************************/

private import tango.core.sync.Condition;
private import tango.core.Exception;
private import tango.core.Thread;
private import tango.text.convert.Integer;
private import tango.io.Stdout;
debug (condition)
{
    private import tango.util.log.Log;
    private import tango.util.log.ConsoleAppender;
    private import tango.util.log.DateLayout;
}


void main(char[][] args)
{
    debug (condition)
    {
        Logger log = Log.getLogger("condition");

        log.addAppender(new ConsoleAppender(new DateLayout()));

        log.info("Condition test");
    }

    testNotifyOne();
    testNotifyAll();
}


/**
 * Test for Condition.notify().
 */
void testNotifyOne()
{
    debug (condition)
    {
        Logger log = Log.getLogger("condition.notify-one");
    }

    scope Mutex     mutex   = new Mutex();
    scope Condition cond    = new Condition(mutex);
    int             waiting = 0;
    Thread          thread;

    void notifyOneTestThread()
    {
        debug (condition)
        {
            Logger log = Log.getLogger("condition.notify-one." ~ Thread.getThis().name());

            log.trace("Starting thread");
        }

        try
        {
            synchronized (mutex)
            {
                debug (condition)
                    log.trace("Acquired mutex");

                scope(exit)
                {
                    debug (condition)
                        log.trace("Releasing mutex");
                }

                waiting++;

                while (waiting != 2)
                {
                    debug (condition)
                        log.trace("Waiting on condition variable");
                    cond.wait();
                }

                debug (condition)
                    log.trace("Condition variable was signaled");
            }
        }
        catch (SyncException e)
        {
            Stderr.formatln("Sync exception caught in Condition test thread {0}:\n{1}",
                            Thread.getThis().name(), e.toString());
        }
        catch (Exception e)
        {
            Stderr.formatln("Unexpected exception caught in Condition test thread {0}:\n{1}",
                            Thread.getThis().name(), e.toString());
        }
        debug (condition)
            log.trace("Exiting thread");
    }

    thread = new Thread(&notifyOneTestThread);
    thread.name = "thread-1";

    debug (condition)
        log.trace("Created thread " ~ thread.name);
    thread.start();

    try
    {
        // Poor man's barrier: wait until the other thread is waiting.
        while (true)
        {
            synchronized (mutex)
            {
                if (waiting != 1)
                {
                    Thread.yield();
                }
                else
                {
                    break;
                }
            }
        }

        synchronized (mutex)
        {
            debug (condition)
                log.trace("Acquired mutex");

            waiting++;

            debug (condition)
                log.trace("Notifying test thread");
            cond.notify();

            debug (condition)
                log.trace("Releasing mutex");
        }

        thread.join();

        if (waiting == 2)
        {
            debug (condition)
                log.info("The Condition notification test to one thread was successful");
        }
        else
        {
            debug (condition)
            {
                log.error("The condition variable notification to one thread is not working");
                assert(false);
            }
            else
            {
                assert(false, "The condition variable notification to one thread is not working");
            }
        }
    }
    catch (SyncException e)
    {
        Stderr.formatln("Sync exception caught in main thread:\n{0}", e.toString());
    }
}


/**
 * Test for Condition.notifyAll().
 */
void testNotifyAll()
{
    const uint MaxThreadCount = 10;

    debug (condition)
    {
        Logger log = Log.getLogger("condition.notify-all");
    }

    scope Mutex     mutex   = new Mutex();
    scope Condition cond    = new Condition(mutex);
    int             waiting = 0;

    /**
     * This thread waits for a notification from the main thread.
     */
    void notifyAllTestThread()
    {
        debug (condition)
        {
            Logger log = Log.getLogger("condition.notify-all." ~ Thread.getThis().name());

            log.trace("Starting thread");
        }

        try
        {
            synchronized (mutex)
            {
                debug (condition)
                    log.trace("Acquired mutex");

                waiting++;

                while (waiting != MaxThreadCount + 1)
                {
                    debug (condition)
                        log.trace("Waiting on condition variable");
                    cond.wait();
                }

                debug (condition)
                    log.trace("Condition variable was signaled");

                debug (condition)
                    log.trace("Releasing mutex");
            }
        }
        catch (SyncException e)
        {
            Stderr.formatln("Sync exception caught in Condition test thread {0}:\n{1}",
                            Thread.getThis().name(), e.toString());
        }
        catch (Exception e)
        {
            Stderr.formatln("Unexpected exception caught in Condition test thread {0}:\n{1}",
                            Thread.getThis().name(), e.toString());
        }
        debug (condition)
            log.trace("Exiting thread");
    }

    ThreadGroup group = new ThreadGroup();
    Thread      thread;
    char[10]    tmp;

    for (uint i = 0; i < MaxThreadCount; ++i)
    {
        thread = new Thread(&notifyAllTestThread);
        thread.name = "thread-" ~ format(tmp, i);

        group.add(thread);
        debug (condition)
            log.trace("Created thread " ~ thread.name);
        thread.start();
    }

    try
    {
        // Poor man's barrier: wait until all the threads are waiting.
        while (true)
        {
            synchronized (mutex)
            {
                if (waiting != MaxThreadCount)
                {
                    Thread.yield();
                }
                else
                {
                    break;
                }
            }
        }

        synchronized (mutex)
        {
            debug (condition)
                log.trace("Acquired mutex");

            waiting++;

            debug (condition)
                log.trace("Notifying all threads");
            cond.notifyAll();

            debug (condition)
                log.trace("Releasing mutex");
        }

        debug (condition)
            log.trace("Waiting for threads to finish");
        group.joinAll();

        if (waiting == MaxThreadCount + 1)
        {
            debug (condition)
                log.info("The Condition notification test to many threads was successful");
        }
        else
        {
            debug (condition)
            {
                log.error("The condition variable notification to many threads is not working");
                assert(false);
            }
            else
            {
                assert(false, "The condition variable notification to many threads is not working");
            }
        }
    }
    catch (SyncException e)
    {
        Stderr.formatln("Sync exception caught in main thread:\n{0}", e.toString());
    }
}
