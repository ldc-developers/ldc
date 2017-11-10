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
}

