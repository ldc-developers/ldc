// Make sure the LDC_no_typeinfo pragma prevents emission of TypeInfos for structs.

// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

// CHECK: _D42TypeInfo_S18pragma_no_typeinfo10NoTypeInfo6__initZ = external global %object.TypeInfo_Struct
pragma(LDC_no_typeinfo)
struct NoTypeInfo {}

// CHECK: _D44TypeInfo_S18pragma_no_typeinfo12WithTypeInfo6__initZ = linkonce_odr global %object.TypeInfo_Struct
struct WithTypeInfo {}

pragma(LDC_no_typeinfo):
// CHECK: _D43TypeInfo_S18pragma_no_typeinfo11NoTypeInfoA6__initZ = external global %object.TypeInfo_Struct
struct NoTypeInfoA {}
// CHECK: _D43TypeInfo_S18pragma_no_typeinfo11NoTypeInfoB6__initZ = external global %object.TypeInfo_Struct
struct NoTypeInfoB {}
