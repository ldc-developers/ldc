/**
 * A fixed memory pool for fast allocation/deallocation of structs.
 *
 * Authors:   Mithun Hunsur
 */
module ldc.eh.fixedpool;

import core.stdc.stdlib : malloc, free;

/***********************************
 * A fixed pool of T instances, used to optimise allocation/deallocation
 * performance for up to N instances before falling back to the heap.
 *
 * Implemented as an intrusive free list; each instance not in use contains
 * a pointer to the next free instance. As a result, allocation in the common
 * case is O(1); the first free instance is initialised and returned to the user.
 *
 * When deallocating, the given pointer is checked against the pool; if it lies
 * within the pool's range of addresses, it gets given the address of the old
 * first free instance, and the first free instance is then set to point at the
 * newly freed pointer.
 */
struct FixedPool(T, int N)
{
    /** 
      * Disable copying: we use internal pointers, so copying would not work
      * as expected (postblit semantics are unsuitable for fixing the pointers) 
      */
    @disable this(this);

    /**
      * Allocate a new instance from the pool if available, or from the heap otherwise.
      */
    T* malloc()
    {
        // Initialize the free list if not already initialized
        if (!initialized)
            initialize();

        if (firstFreeInstance)
        {
            // Set the next free instance to the pointer stored in
            // the current free instance
            auto instance = &firstFreeInstance.value;
            firstFreeInstance = firstFreeInstance.next;
            // Initialize the newly-allocated instance and return it
            *instance = T.init;
            return instance;
        }

        // Allocate a new instance from the heap, initialize it, and return it
        auto instance = cast(T*).malloc(T.sizeof);
        *instance = T.init;
        return instance;
    }

    /**
      * Free an instance that was allocated with this pool's `malloc` method.
      *
      * Warning: Has undefined behaviour if an instance that did not originate 
      * from this `FixedPool` is passed in as the argument.
      */
    void free(T* ptr)
    {
        // Initialize the free list if not already initialized
        if (!initialized)
            initialize();

        // If the instance comes from our pool, add it to the linked list and return
        if (isInstanceInPool(ptr))
        {
            // Overwrite the instance's first few bytes with a pointer to the next entry
            auto instanceBlock = cast(PoolBlock*)ptr;
            instanceBlock.next = firstFreeInstance;
            // Set the first free instance to the newly-freed instance
            firstFreeInstance = instanceBlock;

            return;
        }

        .free(ptr);
    }

    /**
      * Returns whether the given instance belongs to the instance pool.
      *
      * Warning: Does not return whether a given heap-allocated instance
      * originated from this `FixedPool`.
      */
    bool isInstanceInPool(T* ptr) const
    {
        return ptr >= &instances[0].value && ptr <= &instances[$-1].value;
    }

private:
    // We can't default construct the free list, so we initialize it the first time
    // we allocate or deallocate.
    bool initialized = false;
    void initialize()
    {
        // Initialize the free list
        firstFreeInstance = &instances[0];
        // Set each instance to point to the next one in the list
        foreach (i; 0..N-1)
            instances[i].next = &instances[i+1];
        // Set the last instance to point at null
        instances[$-1].next = null;
        initialized = true;
    }
    
    union PoolBlock
    {
        T value;
        PoolBlock* next;
    }

    PoolBlock[N] instances;
    PoolBlock* firstFreeInstance;
}

unittest
{
    struct Test
    {
        int a = 5;
        int b = 6;
    }

    FixedPool!(Test, 8) testPool;
    Test*[] ptrs;

    // Allocate 10 instances and store them in an array for simulated use
    foreach (i; 0..10)
        ptrs ~= testPool.malloc();

    // Check whether the first and last pointers allocated come from
    // the pool and the heap, respectively
    assert(testPool.isInstanceInPool(ptrs[0]));
    assert(!testPool.isInstanceInPool(ptrs[$-1]));

    foreach (ptr; ptrs)
        testPool.free(ptr);

    // After returning all the pointers to the heap, the first free pointer
    // should be the last instance (as it was the last pool instance to be freed.)
    auto ptr = testPool.malloc();
    scope (exit) testPool.free(ptr);
    assert(ptr == &testPool.instances[$-1].value);

    // Verify that the newly-returned instance has been initialized correctly
    assert(ptr.a == 5);
    assert(ptr.b == 6);
}