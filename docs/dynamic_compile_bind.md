# Dynamic compile bind

## D part

* bind function returns `BindPtr` object which is reference counted internally
* `BindPtr` have `opCall` and `toDelegate` methods to call directly or create D delegate
* On creation `BindPtr` call `void registerBindPayload(void* handle, void* originalFunc, void* exampleFunc, const ParamSlice* params, size_t paramsSize)` function
    * `handle` - pointer to pointer to function, which uniquely identifies bind object, actual function pointer to generated code will be written here during `compileDynamicCode` call.
    * `originalFunc` - pointer to `original` function, this is special function, generated inside `bind`, which just forwards parameter to user function or delegate, this function always have `@dynamicCompileEmit` attribute, so jit runtime will fins it even if user function wasn't marked `@dynamicCompile`
    * `exampleFunc` - special function with parameters matched to original user function, runtime will extracts parameters types from it, never called
    * `params` - list of slices to bind parameters, will be null for placeholders
    * `paramsSize` - items count in `params`
* On destruction `BindPtr` call `void unregisterBindPayload(void* handle);`
    * `handle` - same handle as passed in `registerBindPayload` previously

## Runtime part

* `registerBindPayload` add handle to internal list
* During `compileDynamicCode`
    * `generateBind` - Generate new function for each bind handle
        * Parse each bind parameter into llvm constant using existing `parseInitializer` (previously used for `@dynamicCompileConst`)
        * If parameter is function pointer from another bind handle, replace with direct reference to that function
        * Generate call to original function (this call will be inlined)
    * Generate and optimize module as usual
    * `applyBind` - update handles to generated code
