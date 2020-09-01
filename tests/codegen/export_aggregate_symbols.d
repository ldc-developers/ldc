// Tests -fvisibility={default,hidden} for special symbols generated for
// aggregates on non-Windows targets.

// UNSUPPORTED: Windows

// RUN: %ldc %s -shared -fvisibility=default -of=lib%t_default%so
// RUN: nm -g lib%t_default%so | FileCheck -check-prefix=DEFAULT -check-prefix=BOTH %s

// RUN: %ldc %s -shared -fvisibility=hidden -of=lib%t_hidden%so
// RUN: nm -g lib%t_hidden%so | FileCheck -check-prefix=HIDDEN -check-prefix=BOTH %s


// DEFAULT:    _D24export_aggregate_symbols8DefaultC11__interface24export_aggregate_symbols8DefaultI{{.*}}__vtblZ
// HIDDEN-NOT: _D24export_aggregate_symbols8DefaultC11__interface24export_aggregate_symbols8DefaultI{{.*}}__vtblZ
// DEFAULT:    _D24export_aggregate_symbols8DefaultC16__interfaceInfosZ
// HIDDEN-NOT: _D24export_aggregate_symbols8DefaultC16__interfaceInfosZ
// DEFAULT:    _D24export_aggregate_symbols8DefaultC6__initZ
// HIDDEN-NOT: _D24export_aggregate_symbols8DefaultC6__initZ
// DEFAULT:    _D24export_aggregate_symbols8DefaultC6__vtblZ
// HIDDEN-NOT: _D24export_aggregate_symbols8DefaultC6__vtblZ
// DEFAULT:    _D24export_aggregate_symbols8DefaultC7__ClassZ
// HIDDEN-NOT: _D24export_aggregate_symbols8DefaultC7__ClassZ
class DefaultC : DefaultI { void foo() {} }

// DEFAULT:    _D24export_aggregate_symbols8DefaultI11__InterfaceZ
// HIDDEN-NOT: _D24export_aggregate_symbols8DefaultI11__InterfaceZ
interface DefaultI { void foo(); }

// DEFAULT:    _D24export_aggregate_symbols8DefaultS6__initZ
// HIDDEN-NOT: _D24export_aggregate_symbols8DefaultS6__initZ
struct DefaultS { int nonZero = 1; }



// BOTH: _D24export_aggregate_symbols9ExportedC11__interface24export_aggregate_symbols9ExportedI{{.*}}__vtblZ
// BOTH: _D24export_aggregate_symbols9ExportedC16__interfaceInfosZ
// BOTH: _D24export_aggregate_symbols9ExportedC6__initZ
// BOTH: _D24export_aggregate_symbols9ExportedC6__vtblZ
// BOTH: _D24export_aggregate_symbols9ExportedC7__ClassZ
export class ExportedC : ExportedI { void foo() {} }

// BOTH: _D24export_aggregate_symbols9ExportedI11__InterfaceZ
export interface ExportedI { void foo(); }

// BOTH: _D24export_aggregate_symbols9ExportedS6__initZ
export struct ExportedS { int nonZero = 1; }



// struct TypeInfos:

// DEFAULT:    _D45TypeInfo_S24export_aggregate_symbols8DefaultS6__initZ
// HIDDEN-NOT: _D45TypeInfo_S24export_aggregate_symbols8DefaultS6__initZ
auto ti_defaultS = typeid(DefaultS);

// BOTH: _D46TypeInfo_S24export_aggregate_symbols9ExportedS6__initZ
auto ti_exportedS = typeid(ExportedS);
