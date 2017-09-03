module ldc.runtimecompile;

version(LDC_RuntimeCompilation)
{

struct CompilerSettings
{
  uint optLevel = 0;
  uint sizeLevel = 0;
  void delegate(in char[]) dumpHandler = null;
}

void compileDynamicCode(in CompilerSettings settings = CompilerSettings.init)
{
  Context context;
  context.optLevel = settings.optLevel;
  context.sizeLevel = settings.sizeLevel;

  context.fatalHandler = &defaultFatalHandler;
  if (settings.dumpHandler !is null)
  {
    context.dumpHandler = &delegateWrapper!(const char*, size_t);
    context.dumpHandlerData = cast(void*)&settings.dumpHandler;
  }
  rtCompileProcessImpl(context, context.sizeof);
}

private:

extern(C)
{

void delegateWrapper(T...)(void* context, T params)
{
  alias del_type = void delegate(T);
  auto del = cast(del_type*)context;
  (*del)(params);
}

void defaultFatalHandler(void*, const char* reason)
{
  import std.conv;
  throw new Error(reason.text.idup);
}

// must be synchronized with cpp
align(1) struct Context
{
align(1):
  uint optLevel = 0;
  uint sizeLevel = 0;
  void function(void*, const char*, const char*) interruptPointHandler = null;
  void* interruptPointHandlerData = null;
  void function(void*, const char*) fatalHandler = null;
  void* fatalHandlerData = null;
  void function(void*, const char*, size_t) dumpHandler = null;
  void* dumpHandlerData = null;
}
extern void rtCompileProcessImpl(const ref Context context, size_t contextSize);
}

}
