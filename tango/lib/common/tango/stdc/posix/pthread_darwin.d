/**
 * D header file for POSIX.
 *
 * Copyright: Public Domain
 * License:   Public Domain
 * Authors:   Sean Kelly
 * Standards: The Open Group Base Specifications Issue 6, IEEE Std 1003.1, 2004 Edition
 */
module tango.stdc.posix.pthread;

public import tango.stdc.posix.sys.types;
public import tango.stdc.posix.sched;
public import tango.stdc.posix.time;
private import tango.stdc.stdlib;

extern (C):

//
// Required
//

version( darwin )
{
    int pthread_cond_broadcast(pthread_cond_t*);
    int pthread_cond_destroy(pthread_cond_t*);
    int pthread_cond_init(pthread_cond_t*, pthread_condattr_t*);
    //int pthread_cond_signal(pthread_cond_t*);
    //int pthread_cond_timedwait(pthread_cond_t*, pthread_mutex_t*, timespec*);
    int pthread_cond_wait(pthread_cond_t*, pthread_mutex_t*);

    int pthread_mutex_destroy(pthread_mutex_t*);
    int pthread_mutex_init(pthread_mutex_t*, pthread_mutexattr_t*);
    int pthread_mutex_lock(pthread_mutex_t*);
    int pthread_mutex_trylock(pthread_mutex_t*);
    int pthread_mutex_unlock(pthread_mutex_t*);

    //int pthread_rwlock_destroy(pthread_rwlock_t*);
    //int pthread_rwlock_init(pthread_rwlock_t*, pthread_rwlockattr_t*);
    //int pthread_rwlock_rdlock(pthread_rwlock_t*);
    int pthread_rwlock_tryrdlock(pthread_rwlock_t*);
    int pthread_rwlock_trywrlock(pthread_rwlock_t*);
    //int pthread_rwlock_unlock(pthread_rwlock_t*);
    //int pthread_rwlock_wrlock(pthread_rwlock_t*);
}

//
// Barrier (BAR)
//
/*
PTHREAD_BARRIER_SERIAL_THREAD

int pthread_barrier_destroy(pthread_barrier_t*);
int pthread_barrier_init(pthread_barrier_t*, pthread_barrierattr_t*, uint);
int pthread_barrier_wait(pthread_barrier_t*);
int pthread_barrierattr_destroy(pthread_barrierattr_t*);
int pthread_barrierattr_getpshared(pthread_barrierattr_t*, int*); (BAR|TSH)
int pthread_barrierattr_init(pthread_barrierattr_t*);
int pthread_barrierattr_setpshared(pthread_barrierattr_t*, int); (BAR|TSH)
*/

version( darwin )
{
    const PTHREAD_BARRIER_SERIAL_THREAD = -1;

    // defined in tango.stdc.posix.pthread and redefined here
    enum
    {
        PTHREAD_PROCESS_PRIVATE,
        PTHREAD_PROCESS_SHARED
    }

    int pthread_barrier_destroy( pthread_barrier_t* barrier )
    {
        if( barrier is null )
            return EINVAL;
        if( barrier.b_waiters > 0 )
            return EBUSY;
        int mret = pthread_mutex_destroy( &barrier.b_lock );
        int cret = pthread_cond_destroy( &barrier.b_cond );
        free( barrier );
        return mret ? mret : cret;
    }

    int pthread_barrier_init( pthread_barrier_t* barrier,
                              pthread_barrierattr_t* attr,
                              uint count )
    {
        if( barrier is null || count <= 0 )
            return EINVAL;

        pthread_barrier_t* newbarrier = cast(pthread_barrier_t*)
                                            malloc( pthread_barrier_t.sizeof );
        if( newbarrier is null )
            return ENOMEM;

        int   ret;
        if( ( ret = pthread_mutex_init( &newbarrier.b_lock, null ) ) != 0 )
        {
            free( newbarrier );
            return ret;
        }
        if( ( ret = pthread_cond_init( &newbarrier.b_cond, null ) ) != 0 )
        {
            pthread_mutex_destroy( &newbarrier.b_lock );
            free( newbarrier );
            return ret;
        }
        newbarrier.b_waiters    = 0;
        newbarrier.b_count      = count;
        newbarrier.b_generation = 0;
        *barrier                = *newbarrier;

        return 0;
    }

    int pthread_barrier_wait( pthread_barrier_t* barrier )
    {
        if( barrier is null )
            return EINVAL;

        int   ret;
        if( ( ret = pthread_mutex_lock( &barrier.b_lock ) ) != 0 )
            return ret;

        if( ++barrier.b_waiters == barrier.b_count )
        {
            // current thread is lastest thread
            barrier.b_generation++;
            barrier.b_waiters = 0;
            if( ( ret = pthread_cond_broadcast( &barrier.b_cond ) ) == 0 )
                ret = PTHREAD_BARRIER_SERIAL_THREAD;
        }
        else
        {
            int gen = barrier.b_generation;
            do
            {
                ret = pthread_cond_wait( &barrier.b_cond, &barrier.b_lock );
                // test generation to avoid bogus wakeup
            } while( ret == 0 && gen == barrier.b_generation );
        }
        pthread_mutex_unlock( &barrier.b_lock );
        return ret;
    }

    int pthread_barrierattr_destroy( pthread_barrierattr_t* attr )
    {
        if( attr is null )
            return EINVAL;
        free( attr );
        return 0;
    }

    int pthread_barrierattr_getpshared( pthread_barrierattr_t* attr, int* pshared )
    {
        if( attr is null )
            return EINVAL;
        *pshared = attr.pshared;
        return 0;
    }

    int pthread_barrierattr_init( pthread_barrierattr_t* attr )
    {
        if( attr is null )
            return EINVAL;
        if( ( attr = cast(pthread_barrierattr_t*)
                        malloc( pthread_barrierattr_t.sizeof ) ) is null )
            return ENOMEM;
        attr.pshared = PTHREAD_PROCESS_PRIVATE;
        return 0;
    }

    int pthread_barrierattr_setpshared( pthread_barrierattr_t* attr, int pshared )
    {
        if( attr is null )
            return EINVAL;
        // only PTHREAD_PROCESS_PRIVATE is supported
        if( pshared != PTHREAD_PROCESS_PRIVATE )
            return EINVAL;
        attr.pshared = pshared;
        return 0;
    }
}

//
// Timeouts (TMO)
//
/*
int pthread_mutex_timedlock(pthread_mutex_t*, timespec*);
int pthread_rwlock_timedrdlock(pthread_rwlock_t*, timespec*);
int pthread_rwlock_timedwrlock(pthread_rwlock_t*, timespec*);
*/

version( darwin )
{
    private
    {
        import tango.stdc.errno;
        import tango.stdc.posix.unistd;
        import tango.stdc.posix.sys.time;

        extern (D)
        {
            void timerclear( timeval* tvp )
            {
                tvp.tv_sec = tvp.tv_usec = 0;
            }

            bool timerisset( timeval* tvp )
            {
                return tvp.tv_sec || tvp.tv_usec;
            }

            bool timer_cmp_leq( timeval* tvp, timeval* uvp )
            {
                return tvp.tv_sec == uvp.tv_sec ?
                       tvp.tv_usec <= uvp.tv_usec :
                       tvp.tv_sec <= uvp.tv_sec;
            }

            void timeradd( timeval* tvp, timeval* uvp, timeval* vvp )
            {
                vvp.tv_sec = tvp.tv_sec + uvp.tv_sec;
                vvp.tv_usec = tvp.tv_usec + uvp.tv_usec;
                if( vvp.tv_usec >= 1000000 )
                {
                    vvp.tv_sec++;
                    vvp.tv_usec -= 1000000;
                }
            }

            void timersub( timeval* tvp, timeval* uvp, timeval* vvp )
            {
                vvp.tv_sec = tvp.tv_sec - uvp.tv_sec;
                vvp.tv_usec = tvp.tv_usec - uvp.tv_usec;
                if( vvp.tv_usec < 0 )
                {
                    vvp.tv_sec--;
                    vvp.tv_usec += 1000000;
                }
            }

            void TIMEVAL_TO_TIMESPEC( timeval* tv, timespec* ts )
            {
                ts.tv_sec = tv.tv_sec;
                ts.tv_nsec = tv.tv_usec * 1000;
            }

            void TIMESPEC_TO_TIMEVAL( timeval* tv, timespec* ts )
            {
                tv.tv_sec = ts.tv_sec;
                tv.tv_usec = ts.tv_nsec / 1000;
            }
        }
    }

    int pthread_mutex_timedlock( pthread_mutex_t* m, timespec* t )
    {
        timeval currtime;
        timeval maxwait;
        TIMESPEC_TO_TIMEVAL( &maxwait, t );
        timeval waittime;
        waittime.tv_usec = 100;

        while( timer_cmp_leq( &currtime, &maxwait ) )
        {
            int ret = pthread_mutex_trylock( m );
            switch( ret )
            {
            case 0:     // locked successfully
                return ret;
            case EBUSY: // waiting
                timeradd( &currtime, &waittime, &currtime );
                break;
            default:
                return ret;
            }
            usleep( waittime.tv_usec );
        }
        return ETIMEDOUT;
    }

    int pthread_rwlock_timedrdlock( pthread_rwlock_t *rwlock, timespec* t )
    {
        timeval currtime;
        timeval maxwait;
        TIMESPEC_TO_TIMEVAL( &maxwait, t );
        timeval waittime;
        waittime.tv_usec = 100;

        while( timer_cmp_leq( &currtime, &maxwait ) )
        {
            int ret = pthread_rwlock_tryrdlock( rwlock );
            switch( ret )
            {
            case 0:     // locked successfully
                return ret;
            case EBUSY: // waiting
                timeradd( &currtime, &waittime, &currtime );
                break;
            default:
                return ret;
            }
            usleep( waittime.tv_usec );
        }
        return ETIMEDOUT;
    }

    int pthread_rwlock_timedwrlock( pthread_rwlock_t* l, timespec* t )
    {
        timeval currtime;
        timeval maxwait;
        TIMESPEC_TO_TIMEVAL( &maxwait, t );
        timeval waittime;
        waittime.tv_usec = 100;

        while( timer_cmp_leq( &currtime, &maxwait ) )
        {
            int ret = pthread_rwlock_trywrlock( l );
            switch( ret )
            {
            case 0:     // locked successfully
                return ret;
            case EBUSY: // waiting
                timeradd( &currtime, &waittime, &currtime );
                break;
            default:
                return ret;
            }
            usleep( waittime.tv_usec );
        }
        return ETIMEDOUT;
    }
}
