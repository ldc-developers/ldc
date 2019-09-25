/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Operation Interface Definitions                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

CallInterfaceCallable CallOpInterface::getCallableForCallee() {
      return getImpl()->getCallableForCallee(getOperation());
  }
Region * CallableOpInterface::getCallableRegion(CallInterfaceCallable callable) {
      return getImpl()->getCallableRegion(getOperation(), callable);
  }
void CallableOpInterface::getCallableRegions(SmallVectorImpl<Region *> & callables) {
      return getImpl()->getCallableRegions(getOperation(), callables);
  }
