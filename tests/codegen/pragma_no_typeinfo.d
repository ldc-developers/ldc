// Make sure the LDC_no_typeinfo pragma prevents TypeInfo emission.

// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

pragma(LDC_no_moduleinfo); // prevent ModuleInfo from referencing class TypeInfos


// CHECK: _D50TypeInfo_S18pragma_no_typeinfo18StructWithTypeInfo6__initZ = linkonce_odr global %object.TypeInfo_Struct
struct StructWithTypeInfo {}
// force emission
auto ti = typeid(StructWithTypeInfo);

// CHECK: _D18pragma_no_typeinfo17ClassWithTypeInfo7__ClassZ = global %object.TypeInfo_Class
class ClassWithTypeInfo {}

// CHECK: _D18pragma_no_typeinfo21InterfaceWithTypeInfo11__InterfaceZ = global %object.TypeInfo_Class
interface InterfaceWithTypeInfo {}


pragma(LDC_no_typeinfo):

// CHECK-NOT: _D48TypeInfo_S18pragma_no_typeinfo16StructNoTypeInfo6__initZ
struct StructNoTypeInfo {}

// CHECK-NOT: _D18pragma_no_typeinfo15ClassNoTypeInfo7__ClassZ
class ClassNoTypeInfo {}

// CHECK-NOT: _D18pragma_no_typeinfo19InterfaceNoTypeInfo11__InterfaceZ
interface InterfaceNoTypeInfo {}
