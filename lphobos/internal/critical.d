module internal.critical;
extern(C):

import std.c.linux.linux, std.c.stdlib:ccalloc=calloc, cmalloc=malloc, cfree=free;

struct CritSec {
  pthread_mutex_t* p;
}

const PTHREAD_MUTEX_RECURSIVE = 1, PTHREAD_MUTEX_ERRORCHECK=2;

extern(C) int pthread_self();

void _d_criticalenter(CritSec* cs) {
  if (!cs.p) {
    auto newp = cast(pthread_mutex_t*) cmalloc(pthread_mutex_t.sizeof);
    auto cspp = &cs.p;
    pthread_mutexattr_t mt; pthread_mutexattr_init(&mt);
    pthread_mutexattr_settype(&mt, PTHREAD_MUTEX_RECURSIVE);
    printf("Create -> %i\n", pthread_mutex_init(newp, &mt));
    asm { xor EAX, EAX; mov ECX, newp; mov EDX, cspp; lock; cmpxchg int ptr [EDX], ECX; }
    if (cs.p != newp) pthread_mutex_destroy(newp);
  }
  auto count = (cast(uint*) cs.p)[1];
  // printf("%i ::%u\n", pthread_self(), count);
  //printf("%i: Lock %p -> %i\n", pthread_self(), cs.p,
    pthread_mutex_lock(cs.p);//);
}

void _d_criticalexit(CritSec* cs) {
  //printf("%i: Unlock %p -> %i\n", pthread_self(), cs.p,
    pthread_mutex_unlock(cs.p);//);
}

void _d_monitorenter(Object h)
{
    _d_criticalenter(cast(CritSec*) &h.__monitor);
}

void _d_monitorexit(Object h)
{
    _d_criticalexit(cast(CritSec*) &h.__monitor);
}

void _STI_monitor_staticctor() { }
void _STI_critical_init() { }
void _STI_critical_term() { }
void _STD_monitor_staticdtor() { }
void _STD_critical_term() { }
