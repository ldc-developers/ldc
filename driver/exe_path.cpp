//===-- exe_path.cpp ------------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "exe_path.h"

#include <llvm/Support/Path.h>
#if LDC_LLVM_VER >= 304
#include <llvm/Support/FileSystem.h>
#endif

using std::string;
namespace path = llvm::sys::path;

namespace { string exePath; }

void exe_path::initialize(const char* arg0, void* mainAddress)
{
    assert(exePath.empty());
#if LDC_LLVM_VER >= 304
    exePath = llvm::sys::fs::getMainExecutable(arg0, mainAddress);
#else
    exePath = llvm::sys::Path::GetMainExecutable(arg0, mainAddress).str();
#endif
}

const string& exe_path::getExePath()
{
    assert(!exePath.empty());
    return exePath;
}

string exe_path::getBinDir()
{
    assert(!exePath.empty());
    return path::parent_path(exePath);
}

string exe_path::getBaseDir()
{
    string binDir = getBinDir();
    assert(!binDir.empty());
    return path::parent_path(binDir);
}

string exe_path::prependBinDir(const char* suffix)
{
    llvm::SmallString<128> r(getBinDir());
    path::append(r, suffix);
    return r.str();
}
