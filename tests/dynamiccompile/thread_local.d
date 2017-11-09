
// RUN: %ldc -enable-dynamic-compile -run %s

import core.thread;
import ldc.attributes;
import ldc.dynamic_compile;

ThreadID threadId; //thread local

@dynamicCompile void set_val()
{
  threadId = Thread.getThis().id();
}

@dynamicCompile ThreadID get_val()
{
  return threadId;
}

@dynamicCompile ThreadID* get_ptr()
{
  auto ptr = &threadId;
  return ptr;
}

void bar()
{
  set_val();
  auto id = Thread.getThis().id();
  assert(id == threadId);
  assert(id == get_val());
  assert(&threadId is get_ptr());
  assert(id == *get_ptr());
}

void main(string[] args)
{
  compileDynamicCode();
  bar();
  Thread[] threads = [new Thread(&bar),new Thread(&bar),new Thread(&bar)];
  foreach(t;threads[])
  {
    t.start();
  }
  foreach(t;threads[])
  {
    t.join();
  }
}
