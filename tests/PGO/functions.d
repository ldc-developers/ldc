// Test PGO for different kinds of functions
//
// The tests should not test function name mangling, therefore the functionname
// matching strings contain regexp wildcards.

// REQUIRES: PGO_RT

// RUN: %ldc -c -output-ll -fprofile-instr-generate -of=%t.ll %s  \
// RUN:   &&  FileCheck -allow-deprecated-dag-overlap %s --check-prefix=PROFGEN < %t.ll

// RUN: %ldc -fprofile-instr-generate=%t.profraw -run %s  \
// RUN:   &&  %profdata merge %t.profraw -o %t.profdata \
// RUN:   &&  %ldc -c -output-ll -of=%t2.ll -fprofile-instr-use=%t.profdata %s \
// RUN:   &&  FileCheck -allow-deprecated-dag-overlap %s -check-prefix=PROFUSE < %t2.ll

// PROFGEN-DAG: @[[SMPL:__(llvm_profile_counters|profc).*simplefunction[A-Za-z0-9]*]] ={{.*}} [2 x i64] zeroinitializer
// PROFGEN-DAG: @[[TMPL:__(llvm_profile_counters|profc).*templatefunc[A-Za-z0-9]*]] ={{.*}} [2 x i64] zeroinitializer
// PROFGEN-DAG: @[[OUTR:__(llvm_profile_counters|profc).*outerfunc[A-Za-z0-9]*]] ={{.*}} [2 x i64] zeroinitializer
// PROFGEN-DAG: @[[NEST:__(llvm_profile_counters|profc).*nestedfunc[A-Za-z0-9]*]] ={{.*}} [2 x i64] zeroinitializer
// PROFGEN-DAG: @[[LMBD:__(llvm_profile_counters|profc).*testanonymous.*lambda.*]] ={{.*}} [2 x i64] zeroinitializer
// PROFGEN-DAG: @[[KCTR:__(llvm_profile_counters|profc).*Klass.*__ctor.*]] ={{.*}} [2 x i64] zeroinitializer
// PROFGEN-DAG: @[[KMTH:__(llvm_profile_counters|profc).*Klass.*stdmethod.*]] ={{.*}} [2 x i64] zeroinitializer
// PROFGEN-DAG: @[[KDTR:__(llvm_profile_counters|profc).*Klass.*__dtor.*]] ={{.*}} [2 x i64] zeroinitializer
// PROFGEN-DAG: @[[KSTC:__(llvm_profile_counters|profc).*Klass.*staticmethod.*]] ={{.*}} [2 x i64] zeroinitializer
// PROFGEN-DAG: @[[SCTR:__(llvm_profile_counters|profc).*Strukt.*__ctor.*]] ={{.*}} [2 x i64] zeroinitializer
// PROFGEN-DAG: @[[SMTH:__(llvm_profile_counters|profc).*Strukt.*stdmethod.*]] ={{.*}} [2 x i64] zeroinitializer
// PROFGEN-DAG: @[[SDTR:__(llvm_profile_counters|profc).*Strukt.*__dtor.*]] ={{.*}} [2 x i64] zeroinitializer
// PROFGEN-DAG: @[[CNTR:__(llvm_profile_counters|profc).*contractprog.*]] ={{.*}} [5 x i64] zeroinitializer
// PROFGEN-NOT: @{{__llvm_profile_counters_.*fwddecl.*}}
// PROFGEN-NOT: @{{__profc.*fwddecl.*}}


// PROFGEN-LABEL: define {{.*}} @{{.*}}simplefunction{{.*}}(
// PROFUSE-LABEL: define {{.*}} @{{.*}}simplefunction{{.*}}(
// PROFGEN: store {{.*}} @[[SMPL]]
// PROFUSE-SAME: !prof ![[SMPL0:[0-9]+]]
void simplefunction(int i) {
  // PROFGEN: store {{.*}} @[[SMPL]], i{{32|64}} 0, i{{32|64}} 1
  // PROFUSE: br {{.*}} !prof ![[SMPL1:[0-9]+]]
  if (i % 3) {}
}


// PROFGEN-LABEL: define {{.*}} @{{.*}}templatefunc{{.*}}(
// PROFUSE-LABEL: define {{.*}} @{{.*}}templatefunc{{.*}}(
// PROFGEN: store {{.*}} @[[TMPL]]
// PROFUSE-SAME: !prof ![[TMPL0:[0-9]+]]
void templatefunc(T)(T i) {
  // PROFGEN: store {{.*}} @[[TMPL]], i{{32|64}} 0, i{{32|64}} 1
  // PROFUSE: br {{.*}} !prof ![[TMPL1:[0-9]+]]
  if (i % 3) {}
}
// The purpose of this function is to pin the location of the instantiation of
// templatefunc at a defined location in IR.
void call_templatefunc(int i) {
    templatefunc!uint(i);
}


// PROFGEN-LABEL: define {{.*}} @{{.*}}outerfunc{{.*}}(
// PROFUSE-LABEL: define {{.*}} @{{.*}}outerfunc{{.*}}(
// PROFGEN: store {{.*}} @[[OUTR]]
// PROFUSE-SAME: !prof ![[OUTR0:[0-9]+]]
// PROFGEN: store {{.*}} @[[OUTR]], i{{32|64}} 0, i{{32|64}} 1
// PROFUSE: br {{.*}} !prof ![[OUTR1:[0-9]+]]
// PROFGEN-LABEL: define {{.*}} @{{.*}}nestedfunc{{.*}}(
// PROFUSE-LABEL: define {{.*}} @{{.*}}nestedfunc{{.*}}(
// PROFGEN: store {{.*}} @[[NEST]]
// PROFUSE-SAME: !prof ![[NEST0:[0-9]+]]
// PROFGEN: store {{.*}} @[[NEST]], i{{32|64}} 0, i{{32|64}} 1
// PROFUSE: br {{.*}} !prof ![[NEST1:[0-9]+]]
void outerfunc(int i) {
  void nestedfunc(int i) {
    if (!(i % 3)) {}
  }
  nestedfunc(i);
  if (i % 3) {}
}


void takedelegate(int i, int delegate(int) fd) {
  if (fd(i) % 3) {}
}
// PROFGEN-LABEL: define {{.*}} @{{.*}}testanonymous{{.*}}lambda{{.*}}(
// PROFUSE-LABEL: define {{.*}} @{{.*}}testanonymous{{.*}}lambda{{.*}}(
// PROFGEN: store {{.*}} @[[LMBD]]
// PROFUSE-SAME: !prof ![[LMBD0:[0-9]+]]
// PROFGEN: store {{.*}} @[[LMBD]], i{{32|64}} 0, i{{32|64}} 1
// PROFUSE: br {{.*}} !prof ![[LMBD1:[0-9]+]]
void testanonymous(int i) {
  takedelegate(i, (i) { if (i % 5) {} return i+1;} );
}


class Klass {
  int a;

  // PROFGEN-LABEL: define {{.*}} @{{.*}}Klass{{.*}}__ctor{{.*}}(
  // PROFUSE-LABEL: define {{.*}} @{{.*}}Klass{{.*}}__ctor{{.*}}(
  // PROFGEN: store {{.*}} @[[KCTR]]
  // PROFUSE-SAME: !prof ![[KCTR0:[0-9]+]]
  // PROFGEN: store {{.*}} @[[KCTR]], i{{32|64}} 0, i{{32|64}} 1
  // PROFUSE: br {{.*}} !prof ![[KCTR1:[0-9]+]]
  this(int i) {
    a = i;
    if (a % 3) {}
  }

  // PROFGEN-LABEL: define {{.*}} @{{.*}}Klass{{.*}}__dtor{{.*}}(
  // PROFUSE-LABEL: define {{.*}} @{{.*}}Klass{{.*}}__dtor{{.*}}(
  // PROFGEN: store {{.*}} @[[KDTR]]
  // PROFUSE-SAME: !prof ![[KDTR0:[0-9]+]]
  // PROFGEN: store {{.*}} @[[KDTR]], i{{32|64}} 0, i{{32|64}} 1
  // PROFUSE: br {{.*}} !prof ![[KDTR1:[0-9]+]]
  ~this() {
    if (!(a % 3)) {}
  }

  // PROFGEN-LABEL: define {{.*}} @{{.*}}Klass{{.*}}stdmethod{{.*}}(
  // PROFUSE-LABEL: define {{.*}} @{{.*}}Klass{{.*}}stdmethod{{.*}}(
  // PROFGEN: store {{.*}} @[[KMTH]]
  // PROFUSE-SAME: !prof ![[KMTH0:[0-9]+]]
  // PROFGEN: store {{.*}} @[[KMTH]], i{{32|64}} 0, i{{32|64}} 1
  // PROFUSE: br {{.*}} !prof ![[KMTH1:[0-9]+]]
  void stdmethod() {
    if (a % 4) {}
  }

  // PROFGEN-LABEL: define {{.*}} @{{.*}}Klass{{.*}}staticmethod{{.*}}(
  // PROFUSE-LABEL: define {{.*}} @{{.*}}Klass{{.*}}staticmethod{{.*}}(
  // PROFGEN: store {{.*}} @[[KSTC]]
  // PROFUSE-SAME: !prof ![[KSTC0:[0-9]+]]
  // PROFGEN: store {{.*}} @[[KSTC]], i{{32|64}} 0, i{{32|64}} 1
  // PROFUSE: br {{.*}} !prof ![[KSTC1:[0-9]+]]
  static void staticmethod(int i) {
    if (i % 2) {}
  }
}


struct Strukt {
  int a;

  // PROFGEN-LABEL: define {{.*}} @{{.*}}Strukt{{.*}}__ctor{{.*}}(
  // PROFUSE-LABEL: define {{.*}} @{{.*}}Strukt{{.*}}__ctor{{.*}}(
  // PROFGEN: store {{.*}} @[[SCTR]]
  // PROFUSE-SAME: !prof ![[SCTR0:[0-9]+]]
  // PROFGEN: store {{.*}} @[[SCTR]], i{{32|64}} 0, i{{32|64}} 1
  // PROFUSE: br {{.*}} !prof ![[SCTR1:[0-9]+]]
  this(int i) {
    a = i;
    if (a % 3) {}
  }

  // PROFGEN-LABEL: define {{.*}} @{{.*}}Strukt{{.*}}stdmethod{{.*}}(
  // PROFUSE-LABEL: define {{.*}} @{{.*}}Strukt{{.*}}stdmethod{{.*}}(
  // PROFGEN: store {{.*}} @[[SMTH]]
  // PROFUSE-SAME: !prof ![[SMTH0:[0-9]+]]
  // PROFGEN: store {{.*}} @[[SMTH]], i{{32|64}} 0, i{{32|64}} 1
  // PROFUSE: br {{.*}} !prof ![[SMTH1:[0-9]+]]
  void stdmethod() {
    if (a % 4) {}
  }

  // PROFGEN-LABEL: define {{.*}} @{{.*}}Strukt{{.*}}__dtor{{.*}}(
  // PROFUSE-LABEL: define {{.*}} @{{.*}}Strukt{{.*}}__dtor{{.*}}(
  // PROFGEN: store {{.*}} @[[SDTR]]
  // PROFUSE-SAME: !prof ![[SDTR0:[0-9]+]]
  // PROFGEN: store {{.*}} @[[SDTR]], i{{32|64}} 0, i{{32|64}} 1
  // PROFUSE: br {{.*}} !prof ![[SDTR1:[0-9]+]]
  ~this() {
    if (!(a % 3)) {}
  }
}

// PROFGEN-LABEL: define {{.*}} @{{.*}}contractprog{{.*}}(
// PROFUSE-LABEL: define {{.*}} @{{.*}}contractprog{{.*}}(
// PROFGEN: store {{.*}} @[[CNTR]]
// PROFUSE-SAME: !prof ![[CNTR0:[0-9]+]]
void contractprog(int i)
in {
  // PROFGEN: store {{.*}} @[[CNTR]], i{{32|64}} 0, i{{32|64}} 1
  // PROFUSE: br {{.*}} !prof ![[CNTR1:[0-9]+]]
  if (i < 3) {}
}
out {
  if (i < 6) {}
}
body {
  // PROFGEN: store {{.*}} @[[CNTR]], i{{32|64}} 0, i{{32|64}} 2
  // PROFUSE: br {{.*}} !prof ![[CNTR2:[0-9]+]]
  if (i % 2) {}
}
// Out label+body:
// PROFGEN: store {{.*}} @[[CNTR]], i{{32|64}} 0, i{{32|64}} 3
// PROFGEN: store {{.*}} @[[CNTR]], i{{32|64}} 0, i{{32|64}} 4
// PROFUSE: br {{.*}} !prof ![[CNTR4:[0-9]+]]


// Check that no code is generated for function declarations without definition.
// PROFGEN-NOT: define {{.*}} @fwddecl(
// PROFUSE-NOT: define {{.*}} @fwddecl(
extern (C) void fwddecl(int);


void main() {
  foreach (int i; 0..10) {
    simplefunction(i);
    call_templatefunc(i);
    outerfunc(i);
    testanonymous(i);
    scope k = new Klass(i); // `scope` for deterministic finalization
    k.stdmethod();
    Klass.staticmethod(i);
    auto s = Strukt(i);
    s.stdmethod();
    contractprog(i);
  }
}

// PROFUSE-DAG: ![[SMPL0]] = !{!"function_entry_count", i64 10}
// PROFUSE-DAG: ![[SMPL1]] = !{!"branch_weights", i32 7, i32 5}

// PROFUSE-DAG: ![[TMPL0]] = !{!"function_entry_count", i64 10}
// PROFUSE-DAG: ![[TMPL1]] = !{!"branch_weights", i32 7, i32 5}

// PROFUSE-DAG: ![[OUTR0]] = !{!"function_entry_count", i64 10}
// PROFUSE-DAG: ![[OUTR1]] = !{!"branch_weights", i32 7, i32 5}
// PROFUSE-DAG: ![[NEST0]] = !{!"function_entry_count", i64 10}
// PROFUSE-DAG: ![[NEST1]] = !{!"branch_weights", i32 5, i32 7}

// PROFUSE-DAG: ![[LMBD0]] = !{!"function_entry_count", i64 10}
// PROFUSE-DAG: ![[LMBD1]] = !{!"branch_weights", i32 9, i32 3}

// PROFUSE-DAG: ![[KCTR0]] = !{!"function_entry_count", i64 10}
// PROFUSE-DAG: ![[KCTR1]] = !{!"branch_weights", i32 7, i32 5}
// PROFUSE-DAG: ![[KMTH0]] = !{!"function_entry_count", i64 10}
// PROFUSE-DAG: ![[KMTH1]] = !{!"branch_weights", i32 8, i32 4}
// PROFUSE-DAG: ![[KDTR0]] = !{!"function_entry_count", i64 10}
// PROFUSE-DAG: ![[KDTR1]] = !{!"branch_weights", i32 5, i32 7}
// PROFUSE-DAG: ![[KSTC0]] = !{!"function_entry_count", i64 10}
// PROFUSE-DAG: ![[KSTC1]] = !{!"branch_weights", i32 6, i32 6}

// PROFUSE-DAG: ![[SCTR0]] = !{!"function_entry_count", i64 10}
// PROFUSE-DAG: ![[SCTR1]] = !{!"branch_weights", i32 7, i32 5}
// PROFUSE-DAG: ![[SMTH0]] = !{!"function_entry_count", i64 10}
// PROFUSE-DAG: ![[SMTH1]] = !{!"branch_weights", i32 8, i32 4}
// PROFUSE-DAG: ![[SDTR0]] = !{!"function_entry_count", i64 10}
// PROFUSE-DAG: ![[SDTR1]] = !{!"branch_weights", i32 5, i32 7}

// PROFUSE-DAG: ![[CNTR0]] = !{!"function_entry_count", i64 10}
// PROFUSE-DAG: ![[CNTR1]] = !{!"branch_weights", i32 4, i32 8}
// PROFUSE-DAG: ![[CNTR2]] = !{!"branch_weights", i32 6, i32 6}
// PROFUSE-DAG: ![[CNTR4]] = !{!"branch_weights", i32 7, i32 5}

