module ldc.runtimecompile;

immutable runtimeCompile = _runtimeCompile();
private struct _runtimeCompile {}

struct CompilerSettings
{
  uint optLevel = 0;
  uint sizeLeve = 0;
  void delegate(const char*,const char*) interruptHandler = null;
  void delegate(const char*) fatalHandler = null;
  void delegate(in char[]) dumpHandler = null;
}

void rtCompileProcess(in CompilerSettings settings = CompilerSettings.init)
{
  Context context;
  context.optLevel = settings.optLevel;
  context.sizeLeve = settings.sizeLeve;
  if (settings.interruptHandler !is null)
  {
    context.interruptPointHandler = &delegateWrapper!(const char*,const char*);
    context.interruptPointHandlerData = cast(void*)&settings.interruptHandler;
  }
  if (settings.fatalHandler !is null)
  {
    context.fatalHandler = &delegateWrapper!(const char*);
    context.fatalHandlerData = cast(void*)&settings.fatalHandler;
  }
  else
  {
    context.fatalHandler = &defaultFatalHandler;
  }
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
  throw new Exception(reason.text.idup);
}

// must be synchronized with cpp
align(1) struct Context
{
align(1):
  uint optLevel = 0;
  uint sizeLeve = 0;
  void function(void*, const char*, const char*) interruptPointHandler = null;
  void* interruptPointHandlerData = null;
  void function(void*, const char*) fatalHandler = null;
  void* fatalHandlerData = null;
  void function(void*, const char*, size_t) dumpHandler = null;
  void* dumpHandlerData = null;
}
extern void rtCompileProcessImpl(const ref Context context, size_t contextSize);
}