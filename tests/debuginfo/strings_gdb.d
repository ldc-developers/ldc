// REQUIRES: atleast_llvm308
// REQUIRES: Linux
// RUN: %ldc -g -of=%t %s
// RUN: sed -e "/^\\/\\/ GDB:/!d" -e "s,// GDB:,," %s \
// RUN:    | gdb %t >%t.out
// RUN: FileCheck %s -check-prefix=CHECK < %t.out

int main(string[] args)
{
    string[] nargs = args;
    string ns = "a";
    wstring ws = "b";
    dstring ds = "c";

// GDB: b 'strings_gdb.d:17'
// GDB: r
    return 0;
// CHECK: D main

// GDB: ptype string
// CHECK: struct string
// capture size_t representation
// CHECK: [[SIZE_T:(ulong|uint)]] length;
// CHECK: immutable(char) *ptr;

// GDB: ptype wstring
// CHECK: struct wstring
// CHECK: [[SIZE_T]] length;
// CHECK: immutable(wchar) *ptr;

// GDB: ptype dstring
// CHECK: struct dstring
// CHECK: [[SIZE_T]] length;
// CHECK: immutable(dchar) *ptr;

// GDB: info args
// CHECK: args = {"

// GDB: info locals
// CHECK: nargs = {"
// CHECK: ns = "a"
// CHECK: ws = {98}
// CHECK: ds = {99}

// GDB: p ns
// CHECK: = "a"
}

// GDB: q
