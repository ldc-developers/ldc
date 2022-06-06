//===-- gen/scope_exit.h - scope exit helper --------------------*- C++ -*-===//
//
//                         LDC - the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// SCOPE_EXIT helper construct.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <utility>
#include <type_traits>

namespace details {

struct Ownership {
  bool flag = false;

  Ownership() = default;
  Ownership(bool f): flag(f) {}

  Ownership(const Ownership&) = delete;
  Ownership& operator=(const Ownership&) = delete;

  Ownership(Ownership&& rhs):
    flag(rhs.flag) {
    rhs.flag = false;
  }

  Ownership& operator=(Ownership&& rhs) {
    flag = rhs.flag;
    rhs.flag = false;
    return *this;
  }

  operator bool() const {
    return flag;
  }
};

template<typename Func>
struct ScopeExit {
  Func func;
  Ownership active = false;

  ScopeExit(Func&& f):
    func(std::move(f)),
    active(true) {}

  ~ScopeExit() {
    if (active) {
      func();
    }
  }

  ScopeExit(const ScopeExit<Func>&) = delete;
  ScopeExit<Func>& operator=(const ScopeExit<Func>&) = delete;

  ScopeExit(ScopeExit<Func>&&) = default;
  ScopeExit<Func>& operator=(ScopeExit<Func>&&) = default;
};

struct ScopeExitTag {};

template<typename Func>
inline ScopeExit<typename std::decay<Func>::type> operator<<(const ScopeExitTag&, Func&& func) {
  return ScopeExit<typename std::decay<Func>::type>(std::forward<Func>(func));
}

}

#define LDC_STRINGIZE2(a,b) a##b
#define LDC_STRINGIZE(a,b) LDC_STRINGIZE2(a,b)
#define LDC_UNNAME_VAR(basename) LDC_STRINGIZE(basename, __LINE__)

#define SCOPE_EXIT auto LDC_UNNAME_VAR(scope_exit) = details::ScopeExitTag{} << [&]()
