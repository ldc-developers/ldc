// REQUIRES: target_WebAssembly, link_WebAssembly

// optimize to create SSA IR
// RUN: %ldc -mtriple=wasm32-unknown-unknown -O3 -c -output-ll -of=%t.ll %s
// RUN: FileCheck %s < %t.ll

void* ptrGlobal;

// CHECK: %wasm_spill_pointers_pass.U = type { float }
union U { float f; void* p; }

void* getPtr();
size_t getSizeT();
float getF();
void*[] getSlice();
bool getBool();

void usePtr(void*);
void useSizeT(size_t);
void useF(float);
void useSlice(void*[]);

void blackbox();
void blackbox2();
void blackbox3();
void blackboxNoThrow() nothrow;

// CHECK-LABEL: define {{.*}}_D24wasm_spill_pointers_pass5test1FZPv
// Pointer, but not live across call. No spilling
void* test1()
{
    // CHECK-NEXT: [[ptr:%[0-9]+]] = tail call ptr @_D24wasm_spill_pointers_pass6getPtrFZPv()
    auto ptr = getPtr();

    // CHECK-NEXT: ret ptr [[ptr]]
    return ptr;
}

// CHECK-LABEL: define {{.*}}_D24wasm_spill_pointers_pass5test2FZPv
// Pointer live across a call. Spill it
void* test2()
{
    // CHECK-NEXT: [[spill:%stackSpill\..+]] = alloca ptr

    // CHECK-NEXT: [[ptr:%[0-9]+]] = tail call ptr @_D24wasm_spill_pointers_pass6getPtrFZPv()
    // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[spill]])
    // CHECK-NEXT: store volatile ptr [[ptr]], ptr [[spill]]
    auto ptr = getPtr();

    // CHECK-NEXT: tail call void @_D24wasm_spill_pointers_pass8blackboxFZv()
    blackbox();
    // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[spill]])

    // CHECK-NEXT: ret ptr [[ptr]]
    return ptr;
}

// CHECK-LABEL: define {{.*}}_D24wasm_spill_pointers_pass5test3FZPv
// Pointer live across multiple calls.
// Spill once, lifetime ends right after last call.
void* test3()
{
    // CHECK-NEXT: [[spill:%stackSpill\..+]] = alloca ptr

    // CHECK-NEXT: [[ptr:%[0-9]+]] = tail call ptr @_D24wasm_spill_pointers_pass6getPtrFZPv()
    // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[spill]])
    // CHECK-NEXT: store volatile ptr [[ptr]], ptr [[spill]]
    auto ptr = getPtr();

    // CHECK-NEXT: tail call void @_D24wasm_spill_pointers_pass8blackboxFZv()
    blackbox();

    // CHECK-NEXT: tail call void @_D24wasm_spill_pointers_pass8blackboxFZv()
    blackbox();
    // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[spill]])

    // CHECK-NEXT: ret ptr [[ptr]]
    return ptr;
}

// CHECK-LABEL: define {{.*}}_D24wasm_spill_pointers_pass5test4FZk
// Non-pointers don't need to be spilled.
size_t test4()
{
    // CHECK-NOT: stackSpill
    auto val = getSizeT();
    blackbox();
    return val;
}

// CHECK-LABEL: define {{.*}}_D24wasm_spill_pointers_pass5test5FZv
// Pointers don't need to spilled if they don't cross a call
void test5()
{
    // CHECK-NOT: stackSpill
    auto ptr = getPtr();
    ptrGlobal = ptr;

    blackbox();
}

// CHECK-LABEL: define {{.*}}_D24wasm_spill_pointers_pass5test6FZk
// If they aren't used as a pointer, we allow them die early or skip spilling
size_t test6()
{
    // CHECK-NOT: stackSpill
    auto val = cast(size_t)getPtr();
    blackbox();
    return val;
}

// CHECK-LABEL: define {{.*}}_D24wasm_spill_pointers_pass5test7FZv
// Calls that use the pointer also trigger spills.
// Also, lifetimes are disjoint whenever possible
void test7()
{
    // CHECK-NEXT: [[spill1:%stackSpill\..+]] = alloca ptr
    // CHECK-NEXT: [[spill2:%stackSpill\..+]] = alloca ptr

    // CHECK-NEXT: [[ptr1:%[0-9]+]] = tail call ptr @_D24wasm_spill_pointers_pass6getPtrFZPv()
    // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[spill1]])
    // CHECK-NEXT: store volatile ptr [[ptr1]], ptr [[spill1]]
    auto ptr = getPtr();

    // CHECK-NEXT: tail call void @_D24wasm_spill_pointers_pass6usePtrFPvZv(ptr [[ptr1]])
    usePtr(ptr);
    // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[spill1]])


    // CHECK-NEXT: [[ptr2:%[0-9]+]] = tail call ptr @_D24wasm_spill_pointers_pass6getPtrFZPv()
    // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[spill2]])
    // CHECK-NEXT: store volatile ptr [[ptr2]], ptr [[spill2]]
    ptr = getPtr();

    // CHECK-NEXT: tail call void @_D24wasm_spill_pointers_pass6usePtrFPvZv(ptr [[ptr2]])
    usePtr(ptr);
    // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[spill2]])

    return;
}

// CHECK-LABEL: define {{.*}}_D24wasm_spill_pointers_pass5test8FZPv
// Unions don't deter us. Trace potential pointers back
void* test8()
{
    // CHECK-NEXT: [[spill:%stackSpill\..+]] = alloca float

    // CHECK-NEXT: [[u:%[0-9]+]] = tail call float @_D24wasm_spill_pointers_pass4getFFZf()
    // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[spill]])
    // CHECK-NEXT: store volatile float [[u]], ptr [[spill]]
    U u;
    u.f = getF();

    // CHECK-NEXT: tail call void @_D24wasm_spill_pointers_pass4useFFfZv(float [[u]])
    useF(u.f);
    // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[spill]])

    // CHECK-NEXT: [[bitcast:%[0-9]+]] = bitcast float [[u]] to i32
    // CHECK-NEXT: [[ptr:%[0-9]+]] = inttoptr i32 [[bitcast]] to ptr
    // CHECK-NEXT: ret ptr [[ptr]]
    return u.p;
}

// CHECK-LABEL: define {{.*}}_D24wasm_spill_pointers_pass5test9FZAPv
// Spill aggregate containing pointer
void*[] test9()
{
    // CHECK-NEXT: [[spill:%stackSpill\..+]] = alloca { i32, ptr }

    // CHECK-NEXT: [[slice:%[0-9]+]] = tail call { i32, ptr } @_D24wasm_spill_pointers_pass8getSliceFZAPv()
    // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[spill]])
    // CHECK-NEXT: store volatile { i32, ptr } [[slice]], ptr [[spill]]
    auto slice = getSlice();

    // CHECK-NEXT: tail call void @_D24wasm_spill_pointers_pass8blackboxFZv()
    blackbox();
    // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[spill]])

    // CHECK-NEXT: ret { i32, ptr } [[slice]]
    return slice;
}

// CHECK-LABEL: define {{.*}}_D24wasm_spill_pointers_pass6test10FZPv
// Spill lifetime ends properly in the presence of branching
void* test10()
{
    // CHECK-NEXT: [[spill:%stackSpill\..+]] = alloca ptr
    // CHECK-NEXT: [[ptr:%[0-9]+]] = tail call ptr @_D24wasm_spill_pointers_pass6getPtrFZPv()
    // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[spill]])
    // CHECK-NEXT: store volatile ptr [[ptr]], ptr [[spill]]
    auto ptr = getPtr();

    // CHECK-NEXT: [[cond:%[0-9]+]] = tail call zeroext i1 @_D24wasm_spill_pointers_pass7getBoolFZb()
    // CHECK-NEXT: br i1 [[cond]], label %[[ifBB:.+]], label %[[elseBB:.+]]
    if (getBool()) {
        // CHECK: [[ifBB]]:

        // CHECK-NEXT: tail call void @_D24wasm_spill_pointers_pass8blackboxFZv()
        blackbox();
        // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[spill]])

        // CHECK-NEXT: br label %[[afterBB:.+]]
    } else {
        // CHECK: [[elseBB]]:

        // CHECK-NEXT: tail call void @_D24wasm_spill_pointers_pass9blackbox2FZv()
        blackbox2();

        // CHECK-NEXT: tail call void @_D24wasm_spill_pointers_pass9blackbox3FZv()
        blackbox3();
        // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[spill]])

        // CHECK-NEXT: br label %[[afterBB]]
    }

    // CHECK: [[afterBB]]:
    // CHECK-NEXT: ret ptr [[ptr]]
    return ptr;
}

// CHECK-LABEL: define {{.*}}_D24wasm_spill_pointers_pass6test11FZv
// Spill lifetime ends ASAP even if def is in different block
void test11()
{
    // CHECK-NEXT: [[spill:%stackSpill\..+]] = alloca ptr
    // CHECK-NEXT: [[ptr:%[0-9]+]] = tail call ptr @_D24wasm_spill_pointers_pass6getPtrFZPv()
    // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[spill]])
    // CHECK-NEXT: store volatile ptr [[ptr]], ptr [[spill]]
    auto ptr = getPtr();

    // CHECK-NEXT: [[cond:%[0-9]+]] = tail call zeroext i1 @_D24wasm_spill_pointers_pass7getBoolFZb()
    // CHECK-NEXT: br i1 [[cond]], label %[[ifBB:.+]], label %[[elseBB:.+]]
    if (getBool()) {
        // CHECK: [[ifBB]]:
        //
        // CHECK-NEXT: tail call void @_D24wasm_spill_pointers_pass6usePtrFPvZv(ptr %1)
        usePtr(ptr);
        // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[spill]])

        // CHECK-NEXT: tail call void @_D24wasm_spill_pointers_pass8blackboxFZv()
        blackbox();

        // CHECK-NEXT: br label %[[afterBB:.+]]
    } else {
        // CHECK: [[elseBB]]:

        // CHECK-NEXT: tail call void @_D24wasm_spill_pointers_pass9blackbox2FZv()
        blackbox2();

        // CHECK-NEXT: tail call void @_D24wasm_spill_pointers_pass6usePtrFPvZv(ptr %1)
        usePtr(ptr);
        // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[spill]])

        // CHECK-NEXT: br label %[[afterBB]]
    }

    // CHECK: [[afterBB]]:
    // CHECK-NEXT: ret void
}

// CHECK-LABEL: define {{.*}}_D24wasm_spill_pointers_pass6test12FZPv
// Potential pointer detection is conservative
void* test12()
{
    // CHECK-NEXT: [[spill1:%stackSpill\..+]] = alloca i32
    // CHECK-NEXT: [[spill2:%stackSpill\..+]] = alloca i32
    // CHECK-NEXT: [[spill3:%stackSpill\..+]] = alloca i32

    // CHECK-NEXT: [[a:%[0-9]+]] = tail call i32 @_D24wasm_spill_pointers_pass8getSizeTFZk()
    // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[spill2]])
    // CHECK-NEXT: store volatile i32 [[a]], ptr [[spill2]]
    auto a = getSizeT();

    // CHECK-NEXT: [[b:%[0-9]+]] = tail call i32 @_D24wasm_spill_pointers_pass8getSizeTFZk()
    // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[spill3]])
    // CHECK-NEXT: store volatile i32 [[b]], ptr [[spill3]]
    auto b = getSizeT();

    // CHECK-NEXT: [[c:%[0-9]+]] = tail call i32 @_D24wasm_spill_pointers_pass8getSizeTFZk()
    // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[spill1]])
    // CHECK-NEXT: store volatile i32 [[c]], ptr [[spill1]]
    auto c = getSizeT();

    // CHECK-NEXT: [[d:%[0-9]+]] = tail call i32 @_D24wasm_spill_pointers_pass8getSizeTFZk()
    auto d = getSizeT();
    // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[spill1]])
    // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[spill2]])
    // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[spill3]])

    return cast(void*)(
        ((a + b) * c) >> d
    );
}

// CHECK-LABEL: define {{.*}}_D24wasm_spill_pointers_pass6test13FkZPv(i32 %n_arg)
// Loops work as intended
void* test13(size_t n)
{
    // CHECK-NEXT: [[spill1:%stackSpill\..+]] = alloca ptr, align 4
    // CHECK-NEXT: [[spill2:%stackSpill\..+]] = alloca ptr, align 4
    // CHECK-NEXT: [[spill3:%stackSpill\..+]] = alloca i32, align 4

    // CHECK-NEXT: [[ptr:%[0-9]+]] = tail call ptr @_D24wasm_spill_pointers_pass6getPtrFZPv()
    auto ptr = getPtr();

    // CHECK-NEXT: [[shouldSkip:%.+]] = icmp eq i32 %n_arg, 0
    // CHECK-NEXT: br i1 [[shouldSkip]], label %[[endFor:.+]], label %[[forBody:.+]]
    foreach (i; 0..n) {
        // CHECK: [[forBody]]:
        // CHECK-NEXT: [[ptr_phi:%.+]] = phi ptr [ [[ptr_deref:%[0-9]+]], %[[forBody]] ], [ [[ptr]], %{{[0-9]+}} ]
        // CHECK-NEXT: [[i_phi:%.+]] = phi i32 [ [[i_inc:%[0-9]+]], %[[forBody]] ], [ 0, %{{[0-9]+}} ]
        // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[spill3]])
        // CHECK-NEXT: store volatile i32 [[i_phi]], ptr [[spill3]], align 4

        // CHECK-NEXT: [[ptr_deref:%[0-9]+]] = load ptr, ptr [[ptr_phi]], align 4
        // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[spill1]])
        // CHECK-NEXT: store volatile ptr [[ptr_deref]], ptr [[spill1]], align 4
        ptr = *cast(void**)ptr;

        // CHECK-NEXT: [[i_as_ptr:%[0-9]+]] = inttoptr i32 [[i_phi]] to ptr
        // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[spill2]])
        // CHECK-NEXT: store volatile ptr [[i_as_ptr]], ptr [[spill2]], align 4
        //
        // CHECK-NEXT: tail call void @_D24wasm_spill_pointers_pass6usePtrFPvZv(ptr [[i_as_ptr]])
        usePtr(cast(void*)i);
        // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[spill2]])
        // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[spill1]])
        // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[spill3]])


        // CHECK-NEXT: [[i_inc]] = add nuw i32 [[i_phi]], 1
        // CHECK-NEXT: [[shouldExit:%.+]] = icmp eq i32 [[i_inc]], %n_arg
        // CHECK-NEXT: br i1 [[shouldExit]], label %[[endFor]], label %[[forBody]]
    }

    // CHECK: [[endFor]]:
    // CHECK-NEXT: [[ptr_final:%.+]] = phi ptr [ [[ptr]], %{{[0-9]+}} ], [ [[ptr_deref]], %[[forBody]] ]
    // CHECK-NEXT: ret ptr [[ptr_final]]
    return ptr;
}


// Spill are positioned correctly after invokes.
// TODO: add CHECK
void test14()
{
    scope(failure) blackbox();

    usePtr(getPtr());
}

// Spills are placed correctly when PHIs occur in a catchswitch
// TODO: add CHECK
void* test15(int cond, int value) {
    void* ptr;

    try {
        ptr = getPtr();
        blackbox();
    } catch (Exception e) {}

    return ptr;
}


// TODO: more tests
