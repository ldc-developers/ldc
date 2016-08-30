// REQUIRES: atleast_llvm308
// REQUIRES: Windows_x64
// REQUIRES: cdb
// RUN: %ldc -g -of=%t.exe %s \
// RUN:   && sed -e "/^\\/\\/ CDB:/!d" -e "s,// CDB:,," %s \
// RUN:    | %cdb -lines %t.exe >%t.out \
// RUN:   && FileCheck %s < %t.out

int main(string[] args)
{
    string[] nargs = args;
    string ns = "a";
    wstring ws = "b";
    dstring ds = "c";

// CDB: ld codeview
// CDB: bp `codeview.d:19`
// CDB: g
    return 0;
// CHECK: !D main

// CDB: dt string
// CHECK: !string
// CHECK: +0x000 length {{ *}}: Uint8B
// CHECK: +0x008 ptr {{ *}}: Ptr64 UChar

// wchar unsupported by cdb
// CDB: dt wstring
// CHECK: !wstring
// CHECK: +0x000 length {{ *}}: Uint8B
// CHECK: +0x008 ptr {{ *}}: Ptr64

// dchar unsupported by cdb
// CDB: dt dstring
// CHECK: !dstring
// CHECK: +0x000 length {{ *}}: Uint8B
// CHECK: +0x008 ptr {{ *}}: Ptr64

// CDB: dv /t
// struct arguments passed by reference
// CHECK: string[] * args
// CHECK: string[] nargs
// CHECK: string ns
// CHECK: string ws
// CHECK: string ds
}

// CDB: q
// CHECK: quit:
