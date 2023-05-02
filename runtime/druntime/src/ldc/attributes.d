/**
 * Contains compiler-recognized user-defined attribute types.
 *
 * Copyright: Authors 2015-2018
 * License:   $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Authors:   LDC team
 */
module ldc.attributes;

/// Helper template
private template AliasSeq(TList...)
{
    alias AliasSeq = TList;
}

/**
 * Specifies that the function returns `null` or a pointer to at least a
 * certain number of allocated bytes. `sizeArgIdx` and `numArgIdx` specify
 * the 0-based index of the function arguments that should be used to calculate
 * the number of bytes returned:
 *
 *   bytes = arg[sizeArgIdx] * (numArgIdx < 0) ? arg[numArgIdx] : 1
 *
 * The optimizer may assume that an @allocSize function has no other side
 * effects and that it is valid to eliminate calls to the function if the
 * allocated memory is not used. The optimizer will eliminate all code from
 * `foo` in this example:
 *     @allocSize(0) void* myAlloc(size_t size);
 *     void foo() {
 *         auto p = myAlloc(100);
 *         p[0] = 1;
 *     }
 *
 * See LLVM LangRef for more details:
 *    http://llvm.org/docs/LangRef.html#function-attributes
 *
 * This attribute has no effect for LLVM < 3.9.
 *
 * Example:
 * ---
 * import ldc.attributes;
 *
 * @allocSize(0) extern(C) void* malloc(size_t size);
 * @allocSize(2,1) extern(C) void* reallocarray(void *ptr, size_t nmemb,
 *                                              size_t size);
 * @allocSize(0,1) void* my_calloc(size_t element_size, size_t count,
 *                                 bool irrelevant);
 * ---
 */
struct allocSize
{
    int sizeArgIdx;

    /// If numArgIdx < 0, there is no argument specifying the element count
    int numArgIdx = int.min;
}

/++
 + When applied to a global symbol, the compiler, assembler, and linker are
 + required to treat the symbol as if there is a reference to the symbol that
 + it cannot see (which is why they have to be named). For example, it
 + prevents the deletion by the linker of an unreferenced symbol.
 +
 + This attribute corresponds to “attribute((used))” in GNU C.
 +
 + Examples:
 + ---
 + import ldc.attributes;
 +
 + @assumeUsed int dont_remove;
 + ---
 +/
immutable assumeUsed = _assumeUsed();
private struct _assumeUsed
{
}

/++
 + Meant for expert use. Overrides the default calling convention. The supported
 + names for calling conventions depends on the target.
 + If specified multiple times, the last applied UDA is used.
 + The calling convention is not part of the type of the function, which means that
 + this attribute cannot be used in combination with function pointers (the function
 + referenced by a function pointer will be called using the default calling convention).
 + Semantic analysis does NOT (yet?) check for correctness. For example when
 + this is applied to a class function, it is up to the user to ensure that the base
 + function and all overrides (in child classes) have the same calling convention
 + applied.
 +
 + Example (for X86):
 + ---
 + import ldc.attributes;
 +
 + @callingConvention("vectorcall"): // all functions in scope get this UDA
 +
 + int vector_call_convention() { return 42; }
 +
 + @callingConvention("default") // overrides the first UDA
 + int func_with_default_calling_convention() { return 1; }
 + ---
 +/
struct callingConvention
{
    string convention;
}

/++
 + When applied to a function, marks this function for dynamic compilation.
 + Calls to the function will be to the dynamically compiled function,
 + instead of to the statically compiled function (the statically compiled
 + function is still emitted into the object file).
 + All functions marked with this attribute must be explicitly compiled in
 + runtime via ldc.dynamic_compile api before usage.
 +
 + This attribute has no effect if dynamic compilation wasn't enabled with
 + -enable-dynamic-compile
 +
 + Examples:
 + ---
 + import ldc.attributes;
 +
 + @dynamicCompile int foo() { return 42; }
 + ---
 +/
immutable dynamicCompile = _dynamicCompile();
private struct _dynamicCompile
{
}

/++
 + When applied to global variable, this variable will be treated as constant
 + by any dynamically compiled functions and is subject to optimizations.
 + All dynamically compiled functions must be recompiled after any update to
 + such variable.
 +
 + This attribute has no effect if dynamic compilation wasn't enabled with
 + -enable-dynamic-compile
 +
 + Examples:
 + ---
 + import ldc.attributes;
 +
 + @dynamicCompileConst __gshared int value = 1;
 +
 + @dynamicCompile int foo() { return value * 42; }
 + ---
 +/
immutable dynamicCompileConst = _dynamicCompileConst();
private struct _dynamicCompileConst
{
}

/++
 + When applied to a function, makes this function available for dynamic
 + compilation.
 + In contrast to `@dynamicCompile`, calls to the function will be to the
 + statically compiled function (like normal functions). The function body
 + is made available for dynamic compilation with the jit facilities (e.g.
 + jit bind).
 + If both @dynamicCompile and @dynamicCompileEmit attributes are
 + applied to function, @dynamicCompile will get precedence.
 +
 + This attribute has no effect if dynamic compilation wasn't enabled with
 + -enable-dynamic-compile
 +
 + Examples:
 + ---
 + import ldc.attributes;
 +
 + @dynamicCompileEmit int foo() { return 42; }
 + ---
 +/
immutable dynamicCompileEmit = _dynamicCompileEmit();
private struct _dynamicCompileEmit
{
}

/**
 * Explicitly sets "fast math" for a function, enabling aggressive math
 * optimizations. These optimizations may dramatically change the outcome of
 * floating point calculations (e.g. because of reassociation).
 *
 * Example:
 * ---
 * import ldc.attributes;
 *
 * @fastmath
 * double dot(double[] a, double[] b) {
 *     double s = 0;
 *     foreach(size_t i; 0..a.length)
 *     {
 *         // will result in vectorized fused-multiply-add instructions
 *         s += a * b;
 *     }
 *     return s;
 * }
 * ---
 */
alias fastmath = AliasSeq!(llvmAttr("unsafe-fp-math", "true"), llvmFastMathFlag("fast"));

/**
 * Sets the visibility of a function or global variable to "hidden".
 * Such symbols aren't directly accessible from outside the DSO
 * (executable or DLL/.so/.dylib) and are resolved inside the DSO
 * during linking. If unreferenced within the DSO, the linker can
 * strip a hidden symbol.
 * An `export` visibility overrides this attribute.
 */
immutable hidden = _hidden();
private struct _hidden {}

/**
 * Adds an LLVM attribute to a function, without checking the validity or
 * applicability of the attribute.
 * The attribute is specified as key-value pair:
 * @llvmAttr("key", "value")
 * If the value string is empty, just the key is added as attribute.
 *
 * Example:
 * ---
 * import ldc.attributes;
 *
 * @llvmAttr("unsafe-fp-math", "true")
 * double dot(double[] a, double[] b) {
 *     double s = 0;
 *     foreach(size_t i; 0..a.length)
 *     {
 *         import ldc.llvmasm: __ir;
 *         s = __ir!(`%p = fmul fast double %0, %1
 *                    %r = fadd fast double %p, %2
 *                    ret double %r`, double)(a[i], b[i], s);
 *     }
 *     return s;
 * }
 * ---
 */
struct llvmAttr
{
    string key;
    string value;
}

/**
 * Sets LLVM's fast-math flags for floating point operations in the function
 * this attribute is applied to.
 * See LLVM LangRef for possible values:
 *    http://llvm.org/docs/LangRef.html#fast-math-flags
 * @llvmFastMathFlag("clear") clears all flags.
 */
struct llvmFastMathFlag
{
    string flag;
}

/**
 * Adds LLVM's "naked" attribute to a function, disabling function prologue /
 * epilogue emission, incl. LDC's.
 * Intended to be used in combination with a function body defined via
 * ldc.llvmasm.__asm() and/or ldc.simd.inlineIR().
 */
enum naked = llvmAttr("naked");

/**
 * Adds LLVM's "noalias" attribute to a function parameter, with semantics
 * very similar to C99 "restrict".
 * The parameter needs to boil down to a pointer, e.g., be a D pointer, class
 * reference or a `ref` parameter.
 */
enum restrict = llvmAttr("noalias");

/**
 * Adds LLVM's "cold" attribute to a function, indicating that this function is
 * rarely called. Control-flow paths calling cold functions are thus considered
 * to be cold too.
 */
enum cold = llvmAttr("cold");

/**
 * Add LLVM's "nonlazybind" attribute to a function, suppresses lazy symbol binding.
 * This may make calls to function faster, at the cost of extra program startup time
 * if the function is not called during program startup.
 */
enum noplt = llvmAttr("nonlazybind");

/**
 * Disables a particular sanitizer for this function.
 * Valid sanitizer names are all names accepted by `-fsanitize=` commandline option.
 * Multiple sanitizers can be disabled by applying this UDA multiple times, e.g.
 * `@noSanitize("address") `@noSanitize("thread")` to disable both ASan and TSan.
 */
struct noSanitize {
    string sanitizerName;
}

/++
 + Disables split-stack instrumentation for this function, overriding the
 + `-fsplit-stack` commandline function.
 +
 + Examples:
 + ---
 + import ldc.attributes;
 +
 + @noSplitStack int user_function() { return 1; }
 + ---
 +/
immutable noSplitStack = _noSplitStack();
private struct _noSplitStack
{
}

/**
 * Sets the optimization strategy for a function.
 * Valid strategies are "none", "optsize", "minsize". The strategies are mutually exclusive.
 *
 * @optStrategy("none") in particular is useful to selectively debug functions when a
 * fully unoptimized program cannot be used (e.g. due to too low performance).
 *
 * Strategy "none":
 *     Disables most optimizations for a function.
 *     It implies `pragma(inline, false)`: the function is never inlined
 *     in a calling function, and the attribute cannot be combined with
 *     `pragma(inline, true)`.
 *     Functions with `pragma(inline, true)` are still candidates for inlining into
 *     the function.
 *
 * Strategy "optsize":
 *     Tries to keep the code size of the function low and does optimizations to
 *     reduce code size as long as they do not significantly impact runtime performance.
 *
 * Strategy "minsize":
 *     Tries to keep the code size of the function low and does optimizations to
 *     reduce code size that may reduce runtime performance.
 */
struct optStrategy {
    string strategy;
}

/++
 + When applied to a function, specifies that the function should be optimzed by
 + Polly, LLVM's polyhedral optimizer. Useful for optimizing loops for data locatily,
 + vectorization and parallelism.
 +
 + Experimental!
 +
 + Only effective when LDC was built with Polly included.
 +/

 immutable polly = _polly();
 private struct _polly
 {
 }

/**
 * When applied to a global variable or function, causes it to be emitted to a
 * non-standard object file/executable section.
 *
 * The target platform might impose certain restrictions on the format for
 * section names.
 *
 * Examples:
 * ---
 * import ldc.attributes;
 *
 * @section(".mySection") int myGlobal;
 * ---
 */
struct section
{
    string name;
}

/**
 * When applied to a function, specifies that the function should be compiled
 * with different target options than on the command line.
 *
 * The passed string should be a comma-separated list of options. The options
 * are passed to LLVM by adding them to the "target-features" function
 * attribute, after minor processing: negative options (e.g. "no-sse") have the
 * "no" stripped (--> "-sse"), whereas positive options (e.g. sse") gain a
 * leading "+" (--> "+sse"). Negative options override positive options
 * regardless of their order.
 * The "arch=" option is a special case and is passed to LLVM via the
 * "target-cpu" function attribute.
 *
 * Examples:
 * ---
 * import ldc.attributes;
 *
 * @target("no-sse")
 * void foo_nosse(float *A, float* B, float K, uint n) {
 *     for (int i = 0; i < n; ++i)
 *         A[i] *= B[i] + K;
 * }
 * @target("arch=haswell")
 * void foo_haswell(float *A, float* B, float K, uint n) {
 *     for (int i = 0; i < n; ++i)
 *         A[i] *= B[i] + K;
 * }
 * ---
 */
struct target
{
    string specifier;
}

/++
 + When applied to a global symbol, specifies that the symbol should be emitted
 + with weak linkage. An example use case is a library function that should be
 + overridable by user code.
 +
 + Quote from the LLVM manual: "Note that weak linkage does not actually allow
 + the optimizer to inline the body of this function into callers because it
 + doesn’t know if this definition of the function is the definitive definition
 + within the program or whether it will be overridden by a stronger
 + definition."
 +
 + Examples:
 + ---
 + import ldc.attributes;
 +
 + @weak int user_hook() { return 1; }
 + ---
 +/
immutable weak = _weak();
private struct _weak
{
}
