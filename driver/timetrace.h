//===-- driver/timetrace.h --------------------------------------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Compilation time tracing, --ftime-trace.
// Supported from LLVM 10. (LLVM 9 supports time tracing, but without
// granularity check which makes profiles way too large)
//
//===----------------------------------------------------------------------===//

#pragma once

#if LDC_LLVM_VER >= 1000
#define LDC_WITH_TIMETRACER 1
#endif

#if LDC_WITH_TIMETRACER

#include "llvm/Support/TimeProfiler.h"

void initializeTimeTracer();
void deinitializeTimeTracer();
void writeTimeTraceProfile();

/// RAII helper class to call the begin and end functions of the time trace
/// profiler.  When the object is constructed, it begins the section; and when
/// it is destroyed, it stops it.
/// The StringRefs passed are not stored.
struct TimeTraceScope {
  TimeTraceScope() = delete;
  TimeTraceScope(const TimeTraceScope &) = delete;
  TimeTraceScope &operator=(const TimeTraceScope &) = delete;
  TimeTraceScope(TimeTraceScope &&) = delete;
  TimeTraceScope &operator=(TimeTraceScope &&) = delete;

  TimeTraceScope(llvm::StringRef Name) {
    if (llvm::timeTraceProfilerEnabled())
      llvm::timeTraceProfilerBegin(Name, llvm::StringRef(""));
  }
  TimeTraceScope(llvm::StringRef Name, llvm::StringRef Detail) {
    if (llvm::timeTraceProfilerEnabled())
      llvm::timeTraceProfilerBegin(Name, Detail);
  }
  TimeTraceScope(llvm::StringRef Name,
                 llvm::function_ref<std::string()> Detail) {
    if (llvm::timeTraceProfilerEnabled())
      llvm::timeTraceProfilerBegin(Name, Detail);
  }

  ~TimeTraceScope() {
    if (llvm::timeTraceProfilerEnabled())
      llvm::timeTraceProfilerEnd();
  }
};

// Helper function to interface with LLVM's `void
// timeTraceProfilerBegin(StringRef Name, StringRef Detail)`
void timeTraceProfilerBegin(size_t name_length, const char *name_ptr,
                            size_t detail_length, const char *detail_ptr);

#else // LDC_WITH_TIMETRACER

// Provide dummy implementations when not supporting time tracing.

#include "llvm/ADT/StringRef.h"

inline void initializeTimeTracer() {}
inline void deinitializeTimeTracer() {}
inline void writeTimeTraceProfile() {}
struct TimeTraceScope {
  TimeTraceScope() = delete;
  TimeTraceScope(const TimeTraceScope &) = delete;
  TimeTraceScope &operator=(const TimeTraceScope &) = delete;
  TimeTraceScope(TimeTraceScope &&) = delete;
  TimeTraceScope &operator=(TimeTraceScope &&) = delete;

  TimeTraceScope(llvm::StringRef Name) {}
  TimeTraceScope(llvm::StringRef Name, llvm::StringRef Detail) {}
  TimeTraceScope(llvm::StringRef Name,
                 llvm::function_ref<std::string()> Detail) {}
};
inline void timeTraceProfilerBegin(size_t name_length, const char *name_ptr,
                                   size_t detail_length,
                                   const char *detail_ptr) {}

#endif // LDC_WITH_TIMETRACER
