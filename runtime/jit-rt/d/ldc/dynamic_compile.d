/**
 * Contains dynamic compilation API.
 *
 * Copyright: the LDC team
 * License:   $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 */

module ldc.dynamic_compile;

version(LDC_DynamicCompilation):

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

auto bind(F, Args...)(F func, Args args)
{
  import std.format;
  static assert(isFunctionPointer!F);
  alias FuncParams = Parameters!(F);
  enum ParametersCount = FuncParams.length;
  static assert(ParametersCount == Args.length, format("Invalid bind parameter count: %s, expected %s", Args.length, ParametersCount));
  assert(func !is null);
  enum Index = bindParamsInd!(0, 0, Args)();
  alias BindTypes = typeof(mapBindParams(args).expand);
  alias PartialF = ReturnType!F function(UnbindTypes!(Index, FuncParams));
  alias BindPtrType = BindPtr!PartialF;
  return BindPtrType.make!Index(func, mapBindParams(args).expand);
}

immutable placeholder = _placeholder();
private struct _placeholder
{
}

private:
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

auto mapBindParams(Args...)(Args args)
{
  static if (Args.length == 0)
  {
    return tuple();
  }
  else static if (is(Unqual!(Args[0]) == _placeholder))
  {
    return mapBindParams(args[1..$]);
  }
  else
  {
    return tuple(args[0], mapBindParams(args[1..$]).expand);
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
  void function(ref BindPayloadBase!F) dtor;
  F func = null;
  int counter = 1;
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
    Args args;
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
    Slice[ParametersCount] desc;
    static foreach(i, ind; Index)
    {
      static if (InvalidIndex != ind)
      {
        desc[i].data = &(argStore.args[ind]);
        desc[i].size = (argStore.args[ind]).sizeof;
      }
    }

    registerBindPayload(&base.func, cast(void*)originalFunc, desc.ptr, desc.length);
    registered = true;
  }
}

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

  bool isCallable() const pure nothrow @safe @nogc
  {
    return _payload.func !is null;
  }

  Ret opCall(FuncParams args)
  {
    assert(_payload !is null);
    assert(isCallable());
    return _payload.func(args);
  }

  auto toDelegate()
  {
    return &opCall;
  }
}

extern(C)
{
struct Slice
{
  const(void)* data = null;
  size_t size = 0;
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

void registerBindPayload(void* handle, void* originalFunc, const Slice* params, size_t paramsSize);
void unregisterBindPayload(void* handle);
}

