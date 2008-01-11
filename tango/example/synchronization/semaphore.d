/*******************************************************************************
  copyright:   Copyright (c) 2007 Juan Jose Comellas. All rights reserved
  license:     BSD style: $(LICENSE)
  author:      Juan Jose Comellas <juanjo@comellas.com.ar>
               Converted to use core.sync by Sean Kelly <sean@f4.ca>
*******************************************************************************/

module semaphore;

private import tango.core.sync.Semaphore;
private import tango.core.sync.Mutex;
private import tango.core.Exception;
private import tango.core.Exception;
private import tango.core.Thread;
private import tango.io.Console;
private import tango.text.stream.LineIterator;
private import tango.text.convert.Integer;
private import tango.sys.Process;

debug (semaphore)
{
    private import tango.util.log.Log;
    private import tango.util.log.ConsoleAppender;
    private import tango.util.log.DateLayout;
}

const char[] SemaphoreName = "TestProcessSemaphore";


/**
 * Example program for the tango.core.sync.Barrier module.
 */
int main(char[][] args)
{
    if (args.length == 1)
    {
        debug (semaphore)
        {
            Logger log = Log.getLogger("semaphore");

            log.addAppender(new ConsoleAppender(new DateLayout()));

            log.info("Semaphore test");
        }

        testSemaphore();
        testProcessSemaphore(args[0]);

        return 0;
    }
    else
    {
        return testSecondProcessSemaphore();
    }
}

/**
 * Test for single-process (multi-threaded) semaphores.
 */
void testSemaphore()
{
    const uint MaxThreadCount   = 10;

    // Semaphore used in the tests.  Start it "locked" (i.e., its initial
    // count is 0).
    Semaphore   sem = new Semaphore(MaxThreadCount - 1);
    Mutex       mutex = new Mutex();
    uint        count = 0;
    bool        passed = false;

    void semaphoreTestThread()
    {
        debug (semaphore)
        {
            Logger log = Log.getLogger("semaphore.single." ~ Thread.getThis().name());

            log.trace("Starting thread");
        }

        try
        {
            uint threadNumber;

            // 'count' is a resource shared by multiple threads, so we must
            // acquire the mutex before modifying it.
            synchronized (mutex)
            {
                // debug (semaphore)
                //     log.trace("Acquired mutex");
                threadNumber = ++count;
                // debug (semaphore)
                //     log.trace("Releasing mutex");
            }

            // We wait for all the threads to finish counting.
            if (threadNumber < MaxThreadCount)
            {
                sem.wait();
                debug (semaphore)
                    log.trace("Acquired semaphore");

                while (true)
                {
                    synchronized (mutex)
                    {
                        if (count >= MaxThreadCount + 1)
                            break;
                    }
                    Thread.yield();
                }

                debug (semaphore)
                    log.trace("Releasing semaphore");
                sem.notify();
            }
            else
            {
                passed = !sem.tryWait();
                if (passed)
                {
                    debug (semaphore)
                        log.trace("Tried to acquire the semaphore too many times and failed: OK");
                }
                else
                {
                    debug (semaphore)
                        log.error("Tried to acquire the semaphore too may times and succeeded: FAILED");

                    debug (semaphore)
                        log.trace("Releasing semaphore");
                    sem.notify();
                }
                synchronized (mutex)
                {
                    count++;
                }
            }
        }
        catch (SyncException e)
        {
            Cerr("Sync exception caught in Semaphore test thread " ~ Thread.getThis().name ~
                 ":\n" ~ e.toString()).newline;
        }
        catch (Exception e)
        {
            Cerr("Unexpected exception caught in Semaphore test thread " ~ Thread.getThis().name ~
                 ":\n" ~ e.toString()).newline;
        }
    }

    debug (semaphore)
    {
        Logger log = Log.getLogger("semaphore.single");
    }

    ThreadGroup group = new ThreadGroup();
    Thread      thread;
    char[10]    tmp;

    for (uint i = 0; i < MaxThreadCount; ++i)
    {
        thread = new Thread(&semaphoreTestThread);
        thread.name = "thread-" ~ tango.text.convert.Integer.format(tmp, i);

        group.add(thread);
        debug (semaphore)
            log.trace("Created thread " ~ thread.name);
        thread.start();
    }

    debug (semaphore)
        log.trace("Waiting for threads to finish");
    group.joinAll();

    if (passed)
    {
        debug (semaphore)
            log.info("The Semaphore test was successful");
    }
    else
    {
        debug (semaphore)
        {
            log.error("The Semaphore is not working properly: it allowed "
                      "to be acquired more than it should have done");
            assert(false);
        }
        else
        {
            assert(false, "The Semaphore is not working properly: it allowed "
                          "to be acquired more than it should have done");
        }
    }
}

/**
 * Test for multi-process semaphores: this test works by creating a copy of
 * this process that tries to acquire the ProcessSemaphore that was created
 * in this function. If everything works as expected, the attempt should fail,
 * as the count of the semaphore is set to 1.
 */
void testProcessSemaphore(char[] programName)
{
    /+
    bool success = false;

    debug (semaphore)
    {
        Logger log = Log.getLogger("semaphore.multi");
        Logger childLog = Log.getLogger("semaphore.multi.child");

        log.info("ProcessSemaphore test");
    }

    try
    {
        scope ProcessSemaphore sem = new ProcessSemaphore(SemaphoreName, 1);
        Process proc = new Process(programName, "2");

        debug (semaphore)
            log.trace("Created ProcessSemaphore('" ~ SemaphoreName ~ "')'");

        sem.wait();
        debug (semaphore)
            log.trace("Acquired semaphore in main process");

        debug (semaphore)
            log.trace("Executing child test process: " ~ proc.toString());
        proc.execute();

        debug (semaphore)
        {
            foreach (line; new LineIterator!(char)(proc.stdout))
            {
                childLog.trace(line);
            }
        }
        foreach (line; new LineIterator!(char)(proc.stderr))
        {
            Cerr(line).newline;
        }

        debug (semaphore)
            log.trace("Waiting for child process to finish");
        auto result = proc.wait();

        success = (result.reason == Process.Result.Exit && result.status == 2);

        debug (semaphore)
            log.trace("Releasing semaphore in main process");
        sem.notify();
    }
    catch (SyncException e)
    {
        Cerr("Sync exception caught in ProcessSemaphore main test process:\n" ~ e.toString()).newline;
    }
    catch (ProcessException e)
    {
        Cerr("Process exception caught in ProcessSemaphore main test process:\n" ~ e.toString()).newline;
    }
    catch (Exception e)
    {
        Cerr("Unexpected exception caught in ProcessSemaphore main test process:\n" ~ e.toString()).newline;
    }

    if (success)
    {
        debug (semaphore)
            log.info("The ProcessSemaphore test was successful");
    }
    else
    {
        debug (semaphore)
        {
            log.error("The multi-process semaphore is not working");
            assert(false);
        }
        else
        {
            assert(false, "The multi-process semaphore is not working");
        }
    }
    +/
}

/**
 * Test for multi-process semaphores (second process).
 */
int testSecondProcessSemaphore()
{
    int rc = 0;

    /+
    debug (semaphore)
    {
        Cout("Starting child process\n");
    }

    try
    {
        scope ProcessSemaphore sem = new ProcessSemaphore(SemaphoreName);
        bool success;

        success = !sem.tryAcquire();
        if (success)
        {
            debug (semaphore)
                Cout("Tried to acquire semaphore in child process and failed: OK\n");
            rc = 2;
        }
        else
        {
            debug (semaphore)
            {
                Cout("Acquired semaphore in child process: this should not have happened\n");
                Cout("Releasing semaphore in child process\n");
            }
            sem.notify();
            rc = 1;
        }
    }
    catch (SyncException e)
    {
        Cerr("Sync exception caught in ProcessSemaphore child test process:\n" ~ e.toString()).newline;
    }
    catch (ProcessException e)
    {
        Cerr("Process exception caught in ProcessSemaphore child test process:\n" ~ e.toString()).newline;
    }
    catch (Exception e)
    {
        Cerr("Unexpected exception caught in ProcessSemaphore child test process:\n" ~ e.toString()).newline;
    }

    debug (semaphore)
        Cout("Leaving child process\n");

    +/
    return rc;
}
