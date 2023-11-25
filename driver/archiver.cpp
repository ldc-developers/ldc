//===-- archiver.cpp ------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LLVM LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "dmd/errors.h"
#include "dmd/globals.h"
#include "dmd/target.h"
#include "driver/cl_options.h"
#include "driver/timetrace.h"
#include "driver/tool.h"
#include "gen/logger.h"
#if LDC_LLVM_VER < 1700
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Host.h"
#else
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#endif
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ToolDrivers/llvm-lib/LibDriver.h"
#include <cstring>

using namespace llvm;

/* Unlike the llvm-lib driver, llvm-ar is not available as library; it's
 * unfortunately a separate tool.
 * The following is a stripped-down version of LLVM's
 * `tools/llvm-ar/llvm-ar.cpp` (based on LLVM 6.0), as LDC only needs
 * support for `llvm-ar rcs <archive name> <member> ...`.
 * It also makes sure the process isn't simply exited whenever a problem arises.
 */
namespace llvm_ar {

StringRef ArchiveName;
std::vector<const char *> Members;

bool Symtab = true;
bool Deterministic = true;
bool Thin = false;

void fail(Twine Error) { errs() << "llvm-ar: " << Error << ".\n"; }

void fail(std::error_code EC, StringRef Context = {}) {
  fail(Context.empty() ? EC.message() : Context + ": " + EC.message());
}

void fail(Error E, StringRef Context = {}) {
  handleAllErrors(std::move(E), [&](const ErrorInfoBase &EIB) {
    fail(Context.empty() ? EIB.message() : Context + ": " + EIB.message());
  });
}

#define failIfError(Error, Context) \
  if (auto _E = (Error)) { \
    fail(std::move(_E), (Context)); \
    return 1; \
  }

int addMember(std::vector<NewArchiveMember> &Members, StringRef FileName,
              int Pos = -1) {
  Expected<NewArchiveMember> NMOrErr =
      NewArchiveMember::getFile(FileName, Deterministic);
  failIfError(NMOrErr.takeError(), FileName);

  // Use the basename of the object path for the member name.
  NMOrErr->MemberName = sys::path::filename(NMOrErr->MemberName);

  if (Pos == -1)
    Members.push_back(std::move(*NMOrErr));
  else
    Members[Pos] = std::move(*NMOrErr);

  return 0;
}

int addMember(std::vector<NewArchiveMember> &Members,
              const object::Archive::Child &M, int Pos = -1) {
  if (Thin && !M.getParent()->isThin()) {
    fail("Cannot convert a regular archive to a thin one");
    return 1;
  }
  Expected<NewArchiveMember> NMOrErr =
      NewArchiveMember::getOldMember(M, Deterministic);
  failIfError(NMOrErr.takeError(), "");
  if (Pos == -1)
    Members.push_back(std::move(*NMOrErr));
  else
    Members[Pos] = std::move(*NMOrErr);

  return 0;
}

int computeNewArchiveMembers(object::Archive *OldArchive,
                             std::vector<NewArchiveMember> &Ret) {
  if (OldArchive) {
    Error Err = Error::success();
    for (auto &Child : OldArchive->children(Err)) {
      auto NameOrErr = Child.getName();
      failIfError(NameOrErr.takeError(), "");
      StringRef Name = NameOrErr.get();

      auto MemberI = find_if(Members, [Name](StringRef Path) {
        return Name == sys::path::filename(Path);
      });

      if (MemberI == Members.end()) {
        // add old member
        if (int Status = addMember(Ret, Child))
          return Status;
      } else {
        // new member replaces old one with same name at old position
        if (int Status = addMember(Ret, *MemberI))
          return Status;
        Members.erase(MemberI);
      }
    }
    failIfError(std::move(Err), "");
  }

  const int InsertPos = Ret.size();
  for (unsigned I = 0; I != Members.size(); ++I)
    Ret.insert(Ret.begin() + InsertPos, NewArchiveMember());
  int Pos = InsertPos;
  for (auto &Member : Members) {
    if (int Status = addMember(Ret, Member, Pos))
      return Status;
    ++Pos;
  }

  return 0;
}

object::Archive::Kind getDefaultForHost() {
  return Triple(sys::getProcessTriple()).isOSDarwin()
             ? object::Archive::K_DARWIN
             : object::Archive::K_GNU;
}

object::Archive::Kind getKindFromMember(const NewArchiveMember &Member) {
  auto OptionalObject =
      object::ObjectFile::createObjectFile(Member.Buf->getMemBufferRef());

  if (OptionalObject) {
    return isa<object::MachOObjectFile>(**OptionalObject)
               ? object::Archive::K_DARWIN
               : object::Archive::K_GNU;
  }

  // squelch the error in case we had a non-object file
  consumeError(OptionalObject.takeError());
  return getDefaultForHost();
}

int performWriteOperation(object::Archive *OldArchive,
                          std::unique_ptr<MemoryBuffer> OldArchiveBuf) {
  std::vector<NewArchiveMember> NewMembers;
  if (int Status = computeNewArchiveMembers(OldArchive, NewMembers))
    return Status;

  object::Archive::Kind Kind;
  if (Thin)
    Kind = object::Archive::K_GNU;
  else if (OldArchive)
    Kind = OldArchive->kind();
  else
    Kind = getKindFromMember(NewMembers.front());

  auto Result =
      writeArchive(ArchiveName, NewMembers, Symtab, Kind, Deterministic, Thin,
                   std::move(OldArchiveBuf));

  failIfError(std::move(Result), ("error writing '" + ArchiveName + "'").str());

  return 0;
}

int performOperation() {
  if (!sys::fs::exists(ArchiveName)) {
    return performWriteOperation(nullptr, nullptr);
  }

  // Open the archive object.
  auto Buf = MemoryBuffer::getFile(ArchiveName, -1, false);
  std::error_code EC = Buf.getError();
  failIfError(EC, ("error opening '" + ArchiveName + "'").str());

  Error Err = Error::success();
  object::Archive Archive(Buf.get()->getMemBufferRef(), Err);
  EC = errorToErrorCode(std::move(Err));
  failIfError(EC, ("error loading '" + ArchiveName + "'").str());
  return performWriteOperation(&Archive, std::move(Buf.get()));
}

} // namespace llvm_ar

////////////////////////////////////////////////////////////////////////////////

namespace {

int internalAr(ArrayRef<const char *> args) {
  if (args.size() < 4 || strcmp(args[0], "llvm-ar") != 0 ||
      strcmp(args[1], "rcs") != 0) {
    llvm_unreachable(
        "Expected archiver command line: llvm-ar rcs <archive file> "
        "<object file> ...");
    return -1;
  }

  llvm_ar::ArchiveName = args[2];

  auto membersSlice = args.slice(3);
  llvm_ar::Members.clear();
  llvm_ar::Members.insert(llvm_ar::Members.end(), membersSlice.begin(),
                          membersSlice.end());

  return llvm_ar::performOperation();
}

int internalLib(ArrayRef<const char *> args) {
  if (args.size() < 1 || strcmp(args[0], "llvm-lib") != 0) {
    llvm_unreachable("Expected archiver command line: llvm-lib ...");
    return -1;
  }

  return libDriverMain(args);
}

std::string getOutputPath() {
  std::string libName;

  if (global.params.libname.length) { // explicit
    // DMD adds the default extension if there is none
    libName = opts::invokedByLDMD
                  ? FileName::defaultExt(global.params.libname.ptr,
                                         target.lib_ext.ptr)
                  : global.params.libname.ptr;
  } else { // infer from first object file
    libName =
        global.params.objfiles.length
            ? FileName::removeExt(FileName::name(global.params.objfiles[0]))
            : "a.out";
    libName += '.';
    libName += target.lib_ext.ptr;
  }

  // DMD creates static libraries in the objects directory (unless using an
  // absolute output path via `-of`).
  if (opts::invokedByLDMD && global.params.objdir.length &&
      !FileName::absolute(libName.c_str())) {
    libName = FileName::combine(global.params.objdir.ptr, libName.c_str());
  }

  return libName;
}

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

static llvm::cl::opt<std::string> ar("ar", llvm::cl::desc("Archiver"),
                                     llvm::cl::Hidden, llvm::cl::ZeroOrMore);

// path to the produced static library
static std::string gStaticLibPath;

int createStaticLibrary() {
  Logger::println("*** Creating static library ***");
  ::TimeTraceScope timeScope("Create static library");

  const bool isTargetMSVC =
      global.params.targetTriple->isWindowsMSVCEnvironment();

  const bool useInternalArchiver = ar.empty();

#ifdef _WIN32
  windows::MsvcEnvironmentScope msvcEnv;
#endif

  // find archiver
  std::string tool;
  if (useInternalArchiver) {
    tool = isTargetMSVC ? "llvm-lib" : "llvm-ar";
  } else {
#ifdef _WIN32
    if (isTargetMSVC)
      msvcEnv.setup();
#endif

    tool = getProgram(isTargetMSVC ? "lib.exe" : "ar", &ar);
  }

  // remember output path for later
  gStaticLibPath = getOutputPath();

  createDirectoryForFileOrFail(gStaticLibPath);

  // build arguments
  std::vector<std::string> args;

  // ask ar to create a new library
  if (!isTargetMSVC) {
    args.push_back("rcs");
  }

  // ask lib.exe to be quiet
  if (isTargetMSVC) {
    args.push_back("/NOLOGO");
  }

  if (isTargetMSVC) {
    args.push_back("/OUT:" + gStaticLibPath);
  } else {
    args.push_back(gStaticLibPath);
  }

  // object files
  for (auto objfile : global.params.objfiles) {
    args.push_back(objfile);
  }

  // user libs
  for (auto libfile : global.params.libfiles) {
    args.push_back(libfile);
  }

  // .res/.def files for lib.exe
  if (isTargetMSVC) {
    if (global.params.resfile.length)
      args.push_back(global.params.resfile.ptr);
    if (global.params.deffile.length)
      args.push_back(std::string("/DEF:") + global.params.deffile.ptr);
  }

  if (useInternalArchiver) {
    const auto fullArgs =
        getFullArgs(tool.c_str(), args, global.params.v.verbose);

    const int exitCode =
        isTargetMSVC ? internalLib(fullArgs) : internalAr(fullArgs);
    if (exitCode)
      error(Loc(), "%s failed with status: %d", tool.c_str(), exitCode);

    return exitCode;
  }

  // invoke external archiver
  return executeToolAndWait(Loc(), tool, args, global.params.v.verbose);
}

const char *getPathToProducedStaticLibrary() {
  assert(!gStaticLibPath.empty());
  return gStaticLibPath.c_str();
}
