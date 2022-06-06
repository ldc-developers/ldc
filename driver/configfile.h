//===-- driver/configfile.h - LDC config file handling ----------*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Handles reading and parsing of an LDC config file (ldc.conf/ldc2.conf).
//
//===----------------------------------------------------------------------===//

#pragma once

#include "dmd/root/array.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

struct ConfigFile {
public:
  static ConfigFile instance;

  bool read(const char *explicitConfFile, const char *triple);

  llvm::StringRef path() {
    return pathcstr ? llvm::StringRef(pathcstr) : llvm::StringRef();
  }

  void extendCommandLine(llvm::SmallVectorImpl<const char *> &args);

  const Array<const char *> &libDirs() const { return _libDirs; }

  llvm::StringRef rpath() const {
    return rpathcstr ? llvm::StringRef(rpathcstr) : llvm::StringRef();
  }

private:
  bool locate(std::string &pathstr);
  static bool sectionMatches(const char *section, const char *triple);

  // implemented in D
  bool readConfig(const char *cfPath, const char *triple, const char *binDir);

  const char *pathcstr = nullptr;
  Array<const char *> switches;
  Array<const char *> postSwitches;
  Array<const char *> _libDirs;
  const char *rpathcstr = nullptr;
};
