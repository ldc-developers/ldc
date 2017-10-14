module ldc.runtimecompile;

version(LDC_RuntimeCompilation)
{

struct CompilerSettings
{
  uint optLevel = 0;
  uint sizeLevel = 0;
  void delegate(in char[], in char[]) progressHandler = null;
  void delegate(in char[]) dumpHandler = null;
}

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

private:

extern(C)
{

void progressHandlerWrapper(void* context, const char* desc, const char* obj)
{
  import std.string;
  alias DelType = typeof(CompilerSettings.progressHandler);
  auto del = cast(DelType*)context;
  (*del)(fromStringz(desc), fromStringz(obj));
}

void dumpHandlerWrapper(void* context, const char* buff, size_t len)
{
  alias DelType = typeof(CompilerSettings.dumpHandler);
  auto del = cast(DelType*)context;
  assert(buff !is null);
  (*del)(buff[0..len]);
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
