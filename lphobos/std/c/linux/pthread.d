/* Written by Walter Bright, Christopher E. Miller, and many others.
 * www.digitalmars.com
 * Placed into public domain.
 */

module std.c.linux.pthread;

extern (C)
{
    /*  pthread declarations taken from pthread headers and
        http://svn.dsource.org/projects/bindings/trunk/pthreads.d
    */

    /* from bits/types.h
    */

    typedef int __time_t;

    /* from time.h
    */

    struct timespec
    {
        __time_t tv_sec;    /* seconds   */
        int tv_nsec;        /* nanosecs. */
    }

    /* from bits/pthreadtypes.h
    */

    alias uint pthread_t;
    alias uint pthread_key_t;
    alias int pthread_once_t;
    alias int clockid_t;
    alias int pthread_spinlock_t;	// volatile

    struct _pthread_descr_struct
    {
    /*  Not defined in the headers ???
        Just needed here to typedef
        the _pthread_descr pointer
    */
    }

    typedef _pthread_descr_struct* _pthread_descr;

    struct _pthread_fastlock
    {
        int __status;
        int __spinlock;
    }

    typedef long __pthread_cond_align_t;

    struct pthread_cond_t 
    {
        _pthread_fastlock __c_lock;
        _pthread_descr    __c_waiting;
        char[48
            - _pthread_fastlock.sizeof
            - _pthread_descr.sizeof
            - __pthread_cond_align_t.sizeof
            ] __padding;
        __pthread_cond_align_t __align;
    }

    struct pthread_condattr_t
    {
        int __dummy;
    }

    struct pthread_mutex_t
    {
        int         __m_reserved;
        int         __m_count;
        _pthread_descr  __m_owner;
        int         __m_kind;
        _pthread_fastlock __m_lock;
    }

    struct pthread_mutexattr_t
    {
        int __mutexkind;
    }

    /* from pthread.h
    */

    struct _pthread_cleanup_buffer
    {
	void function(void*) __routine;
	void* __arg;
	int __canceltype;
	_pthread_cleanup_buffer* __prev;
    }

    struct __sched_param		// bits/sched.h
    {
	int __sched_priority;
    }

    struct pthread_attr_t
    {
	int __detachstate;
	int __schedpolicy;
	__sched_param schedparam;
	int __inheritshed;
	int __scope;
	size_t __guardsize;
	int __stackaddr_set;
	void* __stackaddr;
	size_t __stacksize;
    }

    struct pthread_barrier_t
    {
	_pthread_fastlock __ba_lock;
	int __ba_required;
	int __ba_present;
	_pthread_descr __ba_waiting;
    }

    struct pthread_barrierattr_t
    {
	int __pshared;
    }

    struct pthread_rwlockattr_t
    {
	int __lockkind;
	int __pshared;
    }

    struct pthread_rwlock_t
    {
	_pthread_fastlock __rw_lock;
	int __rw_readers;
	_pthread_descr __rw_writer;
	_pthread_descr __rw_read_waiting;
	_pthread_descr __rw_write_waiting;
	int __rw_kind;
	int __rw_pshared;
    }

    int pthread_mutex_init(pthread_mutex_t*, pthread_mutexattr_t*);
    int pthread_mutex_destroy(pthread_mutex_t*);
    int pthread_mutex_trylock(pthread_mutex_t*);
    int pthread_mutex_lock(pthread_mutex_t*);
    int pthread_mutex_unlock(pthread_mutex_t*);

    int pthread_mutexattr_init(pthread_mutexattr_t*);
    int pthread_mutexattr_destroy(pthread_mutexattr_t*);

    int pthread_cond_init(pthread_cond_t*, pthread_condattr_t*);
    int pthread_cond_destroy(pthread_cond_t*);
    int pthread_cond_signal(pthread_cond_t*);
    int pthread_cond_wait(pthread_cond_t*, pthread_mutex_t*);
    int pthread_cond_timedwait(pthread_cond_t*, pthread_mutex_t*, timespec*);

    int pthread_attr_init(pthread_attr_t*);
    int pthread_attr_destroy(pthread_attr_t*);
    int pthread_attr_setdetachstate(pthread_attr_t*, int);
    int pthread_attr_getdetachstate(pthread_attr_t*, int*);
    int pthread_attr_setinheritsched(pthread_attr_t*, int);
    int pthread_attr_getinheritsched(pthread_attr_t*, int*);
    int pthread_attr_setschedparam(pthread_attr_t*, __sched_param*);
    int pthread_attr_getschedparam(pthread_attr_t*, __sched_param*);
    int pthread_attr_setschedpolicy(pthread_attr_t*, int);
    int pthread_attr_getschedpolicy(pthread_attr_t*, int*);
    int pthread_attr_setscope(pthread_attr_t*, int);
    int pthread_attr_getscope(pthread_attr_t*, int*);
    int pthread_attr_setguardsize(pthread_attr_t*, size_t);
    int pthread_attr_getguardsize(pthread_attr_t*, size_t*);
    int pthread_attr_setstack(pthread_attr_t*, void*, size_t);
    int pthread_attr_getstack(pthread_attr_t*, void**, size_t*);
    int pthread_attr_setstackaddr(pthread_attr_t*, void*);
    int pthread_attr_getstackaddr(pthread_attr_t*, void**);
    int pthread_attr_setstacksize(pthread_attr_t*, size_t);
    int pthread_attr_getstacksize(pthread_attr_t*, size_t*);

    int pthread_barrierattr_init(pthread_barrierattr_t*);
    int pthread_barrierattr_getpshared(pthread_barrierattr_t*, int*);
    int pthread_barrierattr_destroy(pthread_barrierattr_t*);
    int pthread_barrierattr_setpshared(pthread_barrierattr_t*, int);

    int pthread_barrier_init(pthread_barrier_t*, pthread_barrierattr_t*, uint);
    int pthread_barrier_destroy(pthread_barrier_t*);
    int pthread_barrier_wait(pthread_barrier_t*);

    int pthread_condattr_init(pthread_condattr_t*);
    int pthread_condattr_destroy(pthread_condattr_t*);
    int pthread_condattr_getpshared(pthread_condattr_t*, int*);
    int pthread_condattr_setpshared(pthread_condattr_t*, int);

    int pthread_detach(pthread_t);
    void pthread_exit(void*);
    int pthread_getattr_np(pthread_t, pthread_attr_t*);
    int pthread_getconcurrency();
    int pthread_getcpuclockid(pthread_t, clockid_t*);

    int pthread_mutexattr_getpshared(pthread_mutexattr_t*, int*);
    int pthread_mutexattr_setpshared(pthread_mutexattr_t*, int);
    int pthread_mutexattr_settype(pthread_mutexattr_t*, int);
    int pthread_mutexattr_gettype(pthread_mutexattr_t*, int*);
    int pthread_mutex_timedlock(pthread_mutex_t*, timespec*);
    int pthread_yield();

    int pthread_rwlock_init(pthread_rwlock_t*, pthread_rwlockattr_t*);
    int pthread_rwlock_destroy(pthread_rwlock_t*);
    int pthread_rwlock_rdlock(pthread_rwlock_t*);
    int pthread_rwlock_tryrdlock(pthread_rwlock_t*);
    int pthread_rwlock_timedrdlock(pthread_rwlock_t*, timespec*);
    int pthread_rwlock_wrlock(pthread_rwlock_t*);
    int pthread_rwlock_trywrlock(pthread_rwlock_t*);
    int pthread_rwlock_timedwrlock(pthread_rwlock_t*, timespec*);
    int pthread_rwlock_unlock(pthread_rwlock_t*);

    int pthread_rwlockattr_init(pthread_rwlockattr_t*);
    int pthread_rwlockattr_destroy(pthread_rwlockattr_t*);
    int pthread_rwlockattr_getpshared(pthread_rwlockattr_t*, int*);
    int pthread_rwlockattr_setpshared(pthread_rwlockattr_t*, int);
    int pthread_rwlockattr_getkind_np(pthread_rwlockattr_t*, int*);
    int pthread_rwlockattr_setkind_np(pthread_rwlockattr_t*, int);

    int pthread_spin_init(pthread_spinlock_t*, int);
    int pthread_spin_destroy(pthread_spinlock_t*);
    int pthread_spin_lock(pthread_spinlock_t*);
    int pthread_spin_trylock(pthread_spinlock_t*);
    int pthread_spin_unlock(pthread_spinlock_t*);

    int pthread_cancel(pthread_t);
    void pthread_testcancel();
    int pthread_once(pthread_once_t*, void function());

    int pthread_join(pthread_t, void**);
    int pthread_create(pthread_t*, pthread_attr_t*, void*function(void*), void*);
    pthread_t pthread_self();
    int pthread_equal(pthread_t, pthread_t);
    int pthread_atfork(void function(), void function(), void function());
    void pthread_kill_other_threads_np();
    int pthread_setschedparam(pthread_t, int, __sched_param*);
    int pthread_getschedparam(pthread_t, int*, __sched_param*);
    int pthread_cond_broadcast(pthread_cond_t*);
    int pthread_key_create(pthread_key_t*, void function(void*));
    int pthread_key_delete(pthread_key_t);
    int pthread_setconcurrency(int);
    int pthread_setspecific(pthread_key_t, void*);
    void* pthread_getspecific(pthread_key_t);
    int pthread_setcanceltype(int, int*);
    int pthread_setcancelstate(int, int*);

    void _pthread_cleanup_push(_pthread_cleanup_buffer*, void function(void*), void*);
    void _pthread_cleanup_push_defer(_pthread_cleanup_buffer*, void function(void*), void*);
    void _pthread_cleanup_pop(_pthread_cleanup_buffer*, int);
    void _pthread_cleanup_pop_restore(_pthread_cleanup_buffer*, int);
}

