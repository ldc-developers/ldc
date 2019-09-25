/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Operation Interface Declarations                                           *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace detail {
struct CallOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual CallInterfaceCallable getCallableForCallee(Operation *tablegen_opaque_op) = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
public:
    CallInterfaceCallable getCallableForCallee(Operation *tablegen_opaque_op) final {
      auto op = llvm::cast<ConcreteOp>(tablegen_opaque_op);
      (void)op;
      return op.getCallableForCallee();
    }
  };
};
} // end namespace detail
class CallOpInterface : public OpInterface<CallOpInterface, detail::CallOpInterfaceInterfaceTraits> {
public:
  using OpInterface<CallOpInterface, detail::CallOpInterfaceInterfaceTraits>::OpInterface;
  CallInterfaceCallable getCallableForCallee();
};
namespace detail {
struct CallableOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual Region * getCallableRegion(Operation *tablegen_opaque_op, CallInterfaceCallable callable) = 0;
    virtual void getCallableRegions(Operation *tablegen_opaque_op, SmallVectorImpl<Region *> & callables) = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
public:
    Region * getCallableRegion(Operation *tablegen_opaque_op, CallInterfaceCallable callable) final {
      auto op = llvm::cast<ConcreteOp>(tablegen_opaque_op);
      (void)op;
      return op.getCallableRegion(callable);
    }
    void getCallableRegions(Operation *tablegen_opaque_op, SmallVectorImpl<Region *> & callables) final {
      auto op = llvm::cast<ConcreteOp>(tablegen_opaque_op);
      (void)op;
      return op.getCallableRegions(callables);
    }
  };
};
} // end namespace detail
class CallableOpInterface : public OpInterface<CallableOpInterface, detail::CallableOpInterfaceInterfaceTraits> {
public:
  using OpInterface<CallableOpInterface, detail::CallableOpInterfaceInterfaceTraits>::OpInterface;
  Region * getCallableRegion(CallInterfaceCallable callable);
  void getCallableRegions(SmallVectorImpl<Region *> & callables);
};
