/**
 * Contains compiler-recognized user-defined attribute types.
 *
 * Copyright: Authors 2015-2016
 * License:   $(LINK2 http://www.boost.org/LICENSE_1_0.txt, Boost License 1.0)
 * Authors:   David Nadlinger, Johan Engelen
 */
module ldc.attributes;

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
struct section {
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
struct target {
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
private struct _weak {}
