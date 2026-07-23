// REQUIRES: target_WebAssembly, link_WebAssembly

// emit textual IR *and* compile & link

// RUN: %ldc -mtriple=wasm32-unknown-unknown -output-ll -output-o -of=%t.wasm %s
// RUN: %ldc -mtriple=wasm64-unknown-unknown -L-mwasm64 -output-ll -output-o -of=%t.wasm %s
// RUN: FileCheck %s < %t.ll


// test predefined versions:

version (WebAssembly) {} else static assert(0);


// make sure TLS globals are emitted as regular __gshared globals:

// CHECK: @_D4wasm13definedGlobali = hidden global i32 123
int definedGlobal = 123;


// make sure the ModuleInfo ref is emitted into the __minfo section:

// CHECK: @_D4wasm11__moduleRefZ = linkonce_odr hidden global ptr {{.*}}, section "__minfo"
// CHECK: @llvm.used = appending global [1 x ptr] [ptr {{.*}}@_D4wasm11__moduleRefZ


// test the magic linker symbols via linkability of the following:

extern(C) extern __gshared
{
    void* __start___minfo;
    void* __stop___minfo;
}

extern(C) void _start()
{
    auto size = __stop___minfo - __start___minfo;
}
