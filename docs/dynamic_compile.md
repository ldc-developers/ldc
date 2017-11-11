# Dynamic compilation facilities design overview

`@dynamicCompile` attribute can be applied to any function and non thread-local variable (including lambdas and class virtual methods)

## Compiler part:

* In DtoDeclareFunction read function uda's, if there is a `@dynamicCompile` attribute set `IrFunction::dynamicCompile = true`
* If `IrFunction::dynamicCompile` is `true`, call `declareDynamicCompiledFunction` for this function which will create thunk function and save it to `IrFunction::rtCompileFunc`
* `IrFunction::rtCompileFunc` has same signature and attributes as the original function and any attemt to call or take address of the this function will be redirected to rtCompileFunc in DtoCallee
* Call `defineDynamicCompiledFunction` for any function which have `dynamicCompile == true`
* In `defineDynamicCompiledFunction` create global variable which will hold pointer to jitted function and create thunk function body which just calls function pointer from this global var
* (TODO: It is possible to get rid of this global var, can hotpatch code in runtime with compiled code address, does it worth it? How to do this in llvm?)
* For each `@dynamicCompileConst` variable calls `addDynamicCompiledVar`
* In `writeAndFreeLLModule` call function `generateBitcodeForDynamicCompile` for each module
* In `generateBitcodeForDynamicCompile` first recursively search functions calls inside functions body, starting with `@dynamicCompile` functions.
* After that we have two lists of functions:
  * Directly marked by user `@dynamicCompile`
  * Not marked by user but used by `@dynamicCompile` functions directly or indirectly and with body available
* Both lists are subject to dynamic compilation. For the first list dynamic version will be always used. For the second static version will be used when called from static code and dynamic version when called from dynamic code.
* Search for thread local valiables access in jitted code (required for TLS workaround)
* Also we now have a list of symbols required by dynamic functions (variables and external functions)
* (TODO: Option for limit recursion depth?)
* Clone module via `llvm::CloneModule` copying functions marked for dynamic compilations (directly or indirectly) and `@dynamicCompileConst` variables
* Remove `target-cpu` and `target-features` attributes from all jitted functions except those user set explicitly (these attributes will be set back before jit to host)
* Apply thread local storage workaround to jitted functions if enabled (`replaceDynamicThreadLocals` function, controlled via `-runtime-compile-tls-workaround` command line switch, default on)
  * For each thread local variable accessed in jit code generate static accessor function which returns variable address.
  * Replace direct access to thread local variables with call to accessor function
* Replace calls to jit thunks in jitted functions with direct calls (`fixRtModule`)
* Create module data structures and bitcode array required for dynamic compilation and add module constructor which registers `RtComileModuleList` into global linked list (`setupModuleBitcodeData`)

```
/// Symbol required by dynamic code
struct RtCompileSymList {
  const char *name; // Symbol name
  void *sym;        // Symbol address
};

/// Dynamic function
struct RtCompileFuncList {
  const char *name; // Function name
  void *func;       // Thunk global var address to store compiled function pointer
};

/// @dynamicCompileConst variable
struct RtCompileVarList {
  const char *name;
  const void *init;
};

/// Runtime compiled functions info per module
/// organized in linked list 
/// starting with `DynamicCompileModulesHeadName` (defined in runtime)
struct RtComileModuleList {
  RtComileModuleList *next; // Next module
  const void *irData;       // Ir data
  int32 irDataSize;         // Ir data size in bytes
  RtCompileFuncList *funcList; // Dynamic functions
  int32 funcListSize;            // Dynamic functions count
  RtCompileSymList *symList;   // Symbols
  int32 symListSize;             // Symbols count
};
```

## Runtime part:

* Defines runtime compiled modules list head
* Defines `rtCompileProcessImplSo` function which do actual compilation
* `rtCompileProcessImplSo` for each module in list:
  * Parses ir data into llvm module
  * Set `target-cpu` and `target-features` attributes to host for all function which don't have it
  * For each `@dynamicCompileConst` variable
    * Parse variable value from host process and create initializer
    * Set variable const so llvm can optimize it
* Merge all modules into one via `llvm::Linker::linkModules`
* Optimizes resulting module (TODO: optimization setup taken from LDC but not all options available to compiler available in jit)
* Compile module, resolve functions using RtComileModuleList data and update thunk vars
