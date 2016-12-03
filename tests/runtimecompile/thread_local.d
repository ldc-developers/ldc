
// RUN: %ldc -enable-runtime-compile -run %s

import core.thread;
import ldc.attributes;
import ldc.runtimecompile;

ThreadID threadId; //thread local

@runtimeCompile void foo()
{
  threadId = Thread.getThis().id();
}

void bar()
{
  foo();
  assert(threadId == Thread.getThis().id());
}

void main(string[] args)
{
  rtCompileProcess();
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
