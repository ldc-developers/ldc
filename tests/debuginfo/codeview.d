// REQUIRES: atleast_llvm308
// REQUIRES: Windows
// REQUIRES: cdb
// RUN: %ldc -g -of=%t.exe %s
// RUN: sed -e "/^\\/\\/ CDB:/!d" -e "s,// CDB:,," %s \
// RUN:    | %cdb -snul -lines -y . %t.exe >%t.out
// RUN: FileCheck %s -check-prefix=CHECK -check-prefix=%arch < %t.out

int main(string[] args)
{
    string[] nargs = args;
    string ns = "a";
    wstring ws = "b";
    dstring ds = "c";

// CDB: ld /f codeview*
// enable case sensitive symbol lookup
// CDB: .symopt-1
// CDB: bp `codeview.d:21`
// CDB: g
    return 0;
// CHECK: !D main

// CDB: dt string
// CHECK: !string
// capture size_t and pointer representation
// x64: +0x000 length {{ *}}: [[SIZE_T:Uint8B]]
// x64: +[[OFF:0x008]] ptr {{ *}}: [[PTR:Ptr64]] UChar
// x86: +0x000 length {{ *}}: [[SIZE_T:Uint4B]]
// x86: +[[OFF:0x004]] ptr {{ *}}: [[PTR:Ptr32]] UChar

// wchar unsupported by cdb
// CDB: dt wstring
// CHECK: !wstring
// CHECK: +0x000 length {{ *}}: [[SIZE_T]]
// CHECK: +[[OFF]] ptr {{ *}}: [[PTR]]

// dchar unsupported by cdb
// CDB: dt dstring
// CHECK: !dstring
// CHECK: +0x000 length {{ *}}: [[SIZE_T]]
// CHECK: +[[OFF]] ptr {{ *}}: [[PTR]]

// CDB: dv /t
// struct arguments passed by reference on x64
// x64: string[] * args
// x86: string[] args
// CHECK: string[] nargs
// CHECK: string ns
// CHECK: string ws
// CHECK: string ds

// CDB: ?? ns
// CHECK: +0x000 length {{ *}}: 1
// CHECK: +[[OFF]] ptr {{ *}}: 0x{{[0-9a-f`]* *}} "a"
}

// CDB: q
// CHECK: quit:
