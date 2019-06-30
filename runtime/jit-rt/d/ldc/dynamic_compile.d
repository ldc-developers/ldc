/**
 * Contains dynamic compilation API.
 *
 * Copyright: the LDC team
 * License:   $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 */

module ldc.dynamic_compile;

version (LDC_DynamicCompilation):

import ldc.attributes;

/// Dump handler stage
enum DumpStage : int
{
  OriginalModule = 0,
  MergedModule = 1,
  OptimizedModule = 2,
  FinalAsm = 3
}

/// Dynamic compiler settings
struct CompilerSettings
{
  /// The Optimization Level - Specify the basic optimization level.
  ///    0 = -O0, 1 = -O1, 2 = -O2, 3 = -O3
  uint optLevel = 0;

  /// SizeLevel - How much we're optimizing for size.
  ///    0 = none, 1 = -Os, 2 = -Oz
  uint sizeLevel = 0;

  /// Optional progress handler, dynamic compiler will report compilation stages through it
  /// Signature is (in char[] action, in char[] object)
  /// Actual format of reports is not specified and must be used for debugging
  /// purposes only
  void delegate(in char[], in char[]) progressHandler = null;

  /// Optional dump handler, dynamic compiler will report module contents through it
  /// This function will be called multiple times during compilation and user must concatenate all
  /// reported parts manually
  /// Actual format of dump is not specified and must be used for debugging
  /// purposes only
  void delegate(DumpStage, in char[]) dumpHandler = null;
}

/++
 + Compile all dynamic code.
 + This function must be called before any calls to @dynamicCompile functions and
 + after any changes to @dynamicCompileConst variables
 +
 + Consecutive calls to this function do nothing
 +
 + This function is not thread-safe
 +
 + Example:
 + ---
 + import ldc.attributes, ldc.dynamic_compile, std.stdio;
 +
 + @dynamicCompile int foo() { return value * 42; }
 +
 + void main() {
 +   compileDynamicCode();
 +   writeln(foo());
 + }
 +/
void compileDynamicCode(in CompilerSettings settings = CompilerSettings.init)
{
  Context context;
  context.optLevel = settings.optLevel;
  context.sizeLevel = settings.sizeLevel;

  if (settings.progressHandler !is null)
  {
    context.interruptPointHandler = &progressHandlerWrapper;
    context.interruptPointHandlerData = cast(void*)&settings.progressHandler;
  }

  if (settings.dumpHandler !is null)
  {
    context.dumpHandler = &dumpHandlerWrapper;
    context.dumpHandlerData = cast(void*)&settings.dumpHandler;
  }
  rtCompileProcessImpl(context, context.sizeof);
}

/++
 + Returns a reference-counted functional object based on a function or delegate
 + with values bound to some parameters.
 + Each argument in `args` must be either a value, convertible to function parameter,
 + or `ldc.dynamic_compile.placeholder`.
 + `func` must be a pointer to a function or to a delegate.
 + The JIT runtime will generate an efficient function specialization based on `args`.
 + The passed function (or delegate) must be marked `@dynamicCompile` or
 + `@dynamicCompileEmit` to be efficiently optimized.
 +
 + `compileDynamicCode()` must be called before making calls to the returned
 + functional object.
 +
 + `toDelegate()` can be called on the returned object to get a callable delegate.
 + The returned delegate does not prolong the lifetime of the original object (and
 + thus a copy of the original object must be kept as long as this delegate is alive).
 +
 + Example:
 + ---
 + @dynamicCompile int foo(int a, int b)
 + {
 +   return a + b;
 + }
 +
 + auto f = bind(&foo, 40, placeholder);
 + int delegate(int) d = f.toDelegate();
 +
 + compileDynamicCode();
 +
 + assert(f(2) == 42);
 + assert(d(2) == 42);
 +/
auto bind(F, Args...)(F func, Args args) if (isFunctionPointer!F || isDelegate!F)
{
  assert(func !is null);
  import std.format;
  alias FuncParams = Parameters!F;
  enum ParametersCount = FuncParams.length;
  static assert(ParametersCount == Args.length, format("Invalid bind parameters count: %s, expected %s", Args.length, ParametersCount));
  struct Context
  {
    F saved_func = null;
  }
  @dynamicCompileEmit static auto wrapper(Context context, FuncParams wrapperArgs)
  {
    return context.saved_func(wrapperArgs);
  }
  return bindImpl(&wrapper, Context(func), args);
}

/++
 + Placeholder object to be used with bind.
 +/
immutable placeholder = _placeholder();
private struct _placeholder
{
}

/+
 + Reference-counted object which wraps ldc.dynamic_compile.bind result
 +/
struct BindPtr(F)
{
package:
  static assert(isFunctionPointer!F);
  alias FuncParams = Parameters!(F);
  alias Ret = ReturnType!F;
  alias Payload = BindPayloadBase!(F);
  import core.memory : pureMalloc;
  extern(C) private pure nothrow @nogc static
  {
    pragma(mangle, "free") void pureFree( void *ptr );
  }

  Payload* _payload = null;

  static auto make(int[] Index, OF, Args...)(OF func, Args args)
  {
    import core.exception : onOutOfMemoryError;
    import std.conv : emplace;
    alias PayloadImpl = BindPayload!(OF, F, Index, Args);
    auto payload = cast(PayloadImpl*) pureMalloc(PayloadImpl.sizeof);
    if (payload is null)
    {
        onOutOfMemoryError();
    }
    scope(failure)
    {
      pureFree(payload);
    }

    emplace(payload, func, args);
    payload.register();
    BindPtr!F ret;
    ret._payload = cast(Payload*)payload;
    return ret;
  }

  void decPayload()
  {
    if (_payload !is null)
    {
      auto res = --_payload.counter;
      assert(res >= 0);
      if (res == 0)
      {
        _payload.dtor(*_payload);
        pureFree(_payload);
      }
      _payload = null;
    }
  }

  void incPayload()
  {
    if (_payload !is null)
    {
      ++_payload.counter;
    }
  }

public:
  this(this)
  {
    incPayload();
  }
  ~this()
  {
    decPayload();
  }

  void opAssign(typeof(this) rhs)
  {
    import std.algorithm.mutation : swap;
    decPayload();
    _payload = rhs._payload;
    incPayload();
  }

  void opAssign(typeof(null))
  {
    decPayload();
  }

  bool isNull() const
  {
    return _payload is null;
  }

  bool isCallable() const pure nothrow @safe @nogc
  {
    return _payload !is null && _payload.func !is null;
  }

  auto opCall(FuncParams args)
  {
    assert(isCallable());
    return _payload.func(args);
  }

  @dynamicCompileEmit auto toDelegate() @nogc
  {
    assert(_payload !is null);
    return _payload.toDelegate();
  }
}

private:
auto bindImpl(F, Args...)(F func, Args args)
{
  import std.format;
  static assert(isFunctionPointer!F, "Function pointer expected as first parameter");
  alias FuncParams = Parameters!(F);
  enum ParametersCount = FuncParams.length;
  static assert(ParametersCount == Args.length, format("Invalid bind parameters count: %s, expected %s", Args.length, ParametersCount));
  assert(func !is null);
  enum Index = bindParamsInd!(0, 0, Args)();
  alias PartialF = ReturnType!F function(UnbindTypes!(Index, FuncParams));
  alias BindPtrType = BindPtr!PartialF;
  return BindPtrType.make!Index(func, mapBindParams!(F, 0)(args).expand);
}

import std.meta;
import std.traits;
import std.typecons;

int[] bindParamsInd(int I, int Off, Args...)()
{
  static if (Args.length == 0)
  {
    return [];
  }
  else static if (is(Unqual!(Args[0]) == _placeholder))
  {
    return [-1] ~ bindParamsInd!(I + 1, Off, Args[1..$])();
  }
  else
  {
    return [Off] ~ bindParamsInd!(I + 1, Off + 1, Args[1..$])();
  }
}

auto convert(Dst, Src)(Src src)
{
  Dst ret = src;
  return ret;
}

auto mapBindParams(F, size_t I, Args...)(Args args)
{
  static if (Args.length == 0)
  {
    return tuple();
  }
  else static if (is(Unqual!(Args[0]) == _placeholder))
  {
    return mapBindParams!(F, I + 1)(args[1..$]);
  }
  else
  {
    alias T = Parameters!(F)[I];
    return tuple(convert!T(args[0]), mapBindParams!(F, I + 1)(args[1..$]).expand);
  }
}

template UnbindTypes(int[] Index, Args...)
{
  static assert(Index.length == Args.length);
  static if(Args.length == 0)
  {
    alias UnbindTypes = AliasSeq!();
  }
  else static if (-1 == Index[0])
  {
    alias UnbindTypes = AliasSeq!(Args[0], UnbindTypes!(Index[1..$], Args[1..$]));
  }
  else
  {
    alias UnbindTypes = UnbindTypes!(Index[1..$], Args[1..$]);
  }
}

struct BindPayloadBase(F)
{
  alias FuncParams = Parameters!(F);
  alias Ret = ReturnType!F;
  F func = null;
  static assert(func.offsetof == 0, "func must be first");
  void function(ref BindPayloadBase!F) dtor;
  int counter = 1;

  auto isCallable() const
  {
    return func !is null;
  }

  auto opCall(FuncParams args)
  {
    assert(isCallable());
    return func(args);
  }

  auto toDelegate() @nogc
  {
    return &opCall;
  }
}

struct BindPayload(OF, F, int[] Index, Args...)
{
  enum InvalidIndex = -1;
  alias Base = BindPayloadBase!F;
  static assert(isFunctionPointer!OF);
  static assert(isFunctionPointer!F);
  alias FuncParams = Parameters!(OF);
  enum ParametersCount = FuncParams.length;
  static assert(Index.length == ParametersCount, "Invalid index size");
  extern(C) private pure nothrow @nogc static
  {
      pragma(mangle, "gc_addRange") void pureGcAddRange( in void* p, size_t sz, const TypeInfo ti = null );
      pragma(mangle, "gc_removeRange") void pureGcRemoveRange( in void* p );
  }

  Base base;
  OF originalFunc = null;
  struct ArgStore
  {
    import std.meta: staticMap;
    import std.traits: Unqual;
    alias UArgs = staticMap!(Unqual, Args);
    UArgs args;
  }
  ArgStore argStore;
  bool registered = false;

  this(OF orFunc, Args a)
  {
    assert(orFunc !is null);
    originalFunc = orFunc;
    static if (hasIndirections!(ArgStore))
    {
        pureGcAddRange(&argStore, ArgStore.sizeof);
    }
    argStore.args = a;
    void function(ref BindPayloadBase!F) dtor = (ref Base b)
    {
      auto derived = cast(typeof(this)*)&b;
      .destroy(*derived);
    };
    base.dtor = dtor;
  }
  this(this) @disable;
  ~this()
  {
    if (registered)
    {
      unregisterBindPayload(&base.func);
    }
    static if (hasIndirections!(ArgStore))
    {
      pureGcRemoveRange(&argStore);
    }
  }
  void register()
  {
    assert(!registered);
    ParamSlice[ParametersCount] desc;
    static foreach(i, ind; Index)
    {
      static if (InvalidIndex != ind)
      {
        {
          const ii = ParametersCount - i - 1; // reverse params
          desc[ii].data = &(argStore.args[ind]);
          desc[ii].size = (argStore.args[ind]).sizeof;
          alias T = FuncParams[i];
          desc[ii].type = (isAggregateType!T || isDelegate!T || isStaticArray!T ? ParamType.Aggregate : ParamType.Simple);
        }
      }
    }

    alias Ret = ReturnType!F;
    alias Params = Parameters!F;
    @dynamicCompileEmit static Ret exampleFunc(Params) { assert(false); }
    registerBindPayload(&base.func, cast(void*)originalFunc, cast(void*)&exampleFunc, desc.ptr, desc.length);
    registered = true;
  }

  alias toDelegate = base.toDelegate;
}

extern(C)
{
enum ParamType : uint {
  Simple = 0,
  Aggregate = 1
}
struct ParamSlice
{
  const(void)* data = null;
  size_t size = 0;
  ParamType type = ParamType.Simple;
}

void progressHandlerWrapper(void* context, const char* desc, const char* obj)
{
  import std.string;
  alias DelType = typeof(CompilerSettings.progressHandler);
  auto del = cast(DelType*)context;
  (*del)(fromStringz(desc), fromStringz(obj));
}

void dumpHandlerWrapper(void* context, DumpStage stage, const char* buff, size_t len)
{
  alias DelType = typeof(CompilerSettings.dumpHandler);
  auto del = cast(DelType*)context;
  assert(buff !is null);
  (*del)(stage, buff[0..len]);
}


// must be synchronized with cpp
struct Context
{
  uint optLevel = 0;
  uint sizeLevel = 0;
  void function(void*, const char*, const char*) interruptPointHandler = null;
  void* interruptPointHandlerData = null;
  void function(void*, const char*) fatalHandler = null;
  void* fatalHandlerData = null;
  void function(void*, DumpStage, const char*, size_t) dumpHandler = null;
  void* dumpHandlerData = null;
}
extern void rtCompileProcessImpl(const ref Context context, size_t contextSize);

void registerBindPayload(void* handle, void* originalFunc, void* exampleFunc, const ParamSlice* params, size_t paramsSize);
void unregisterBindPayload(void* handle);
}

