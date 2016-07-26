// Tests debug info generation for nested functions

// REQUIRES: atleast_llvm308

// RUN: %ldc -g -c -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// Also test compilation with optimization on, because it uncovers debuginfo errors (see e.g. PR 1598)
// RUN: %ldc -g -c -O3 -of=%t %s

module mod;

// CHECK: define {{.*}} @{{.*}}encloser
// CHECK-SAME: !dbg
void encloser(int arg0, int arg1)
{
    int enc_n;
    // Check allocation of the frame with alignment that is checked later in the debuginfo
    // CHECK: %.frame = alloca %{{.*}} align 4
    // CHECK: call void @llvm.dbg.declare(metadata %{{.*}} %.frame, metadata ![[FRAME:[0-9]+]], metadata ![[EXPR:[0-9]+]]), !dbg ![[ENCL_LOC:[0-9]+]]

    // CHECK-LABEL: define {{.*}}encloser{{.*}}nested
    void nested(int nes_i)
    {
        arg0 = arg1 = enc_n = nes_i; // accessing arg0, arg1 and enc_n from a nested function turns them into closure variables

        // TODO: Debuginfo generation in the nested function is incorrect at the moment. The first argument (nested frame) should have struct debuginfo attached to it. Perhaps the exact same as in the enclosing function.
    }
}

void pr1598(string fmt)
{
    size_t fmtIdx;
    void nested()
    {
        auto a = fmt[fmtIdx .. $];
    }
}


// CHECK: ![[ENCL_SCOPE:[0-9]+]] ={{.*}} !DISubprogram(name: "mod.encloser"
// CHECK: ![[INTTYPE:[0-9]+]] = !DIBasicType(name: "int"
// CHECK: ![[FRAME]] = !DILocalVariable(name: ".frame", scope: ![[ENCL_SCOPE]],{{.*}} type: ![[FRAME2:[0-9]+]],{{.*}} flags: DIFlagArtificial
// CHECK: ![[FRAME2]] = !DICompositeType(tag: DW_TAG_structure_type,{{.*}} line: 14, size: 96, align: 32, flags: DIFlagArtificial, elements: ![[FRAME3:[0-9]+]]
// CHECK: ![[FRAME3]] = !{![[FRAME_ARG0:[0-9]+]], ![[FRAME_ARG1:[0-9]+]], ![[FRAME_ENCN:[0-9]+]]}
// CHECK: ![[FRAME_ARG0]] = !DIDerivedType(tag: DW_TAG_member, name: "arg0",{{.*}} line: 14, baseType: ![[INTTYPE]], size: 32, align: 32
// CHECK: ![[FRAME_ARG1]] = !DIDerivedType(tag: DW_TAG_member, name: "arg1",{{.*}} line: 14, baseType: ![[INTTYPE]], size: 32, align: 32, offset: 32
// CHECK: ![[FRAME_ENCN]] = !DIDerivedType(tag: DW_TAG_member, name: "enc_n",{{.*}} line: 16, baseType: ![[INTTYPE]], size: 32, align: 32, offset: 64

// CHECK: ![[ENCL_LOC]] = !DILocation(line: 14, column: 6, scope: ![[ENCL_SCOPE]])
// CHECK: ![[EXPR]] = !DIExpression()

// CHECK: ![[NEST_SCOPE:[0-9]+]] ={{.*}} !DISubprogram(name: "{{.*}}encloser.nested"
// CHECK: !DILocalVariable{{.*}}nes_i
// CHECK-SAME: arg: 2

