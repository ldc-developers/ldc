//===-- archiver.cpp ------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LLVM LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Builds up (relatively) standard unix archive files (.a) containing LLVM
// bitcode or other files.
//
//===----------------------------------------------------------------------===//

#if LDC_LLVM_VER >= 309

#include "driver/archiver.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#if LDC_LLVM_VER >= 500
#include "llvm/ToolDrivers/llvm-lib/LibDriver.h"
#else
#include "llvm/LibDriver/LibDriver.h"
#endif

#include <cstring>

using namespace llvm;

/* Unlike the llvm-lib driver, llvm-ar is not available as library; it's
 * unfortunately a separate tool.
 * The following is a stripped-down version of LLVM's
 * `tools/llvm-ar/llvm-ar.cpp` (based on early LLVM 5.0), as LDC only needs
 * support for `llvm-ar rcs <archive name> <member> ...`.
 * It also makes sure the process isn't simply exited whenever a problem arises.
 */
namespace {

StringRef ArchiveName;
std::vector<const char *> Members;

bool Symtab = true;
bool Deterministic = true;
bool Thin = false;

void fail(Twine Error) { outs() << "llvm-ar: " << Error << ".\n"; }

void fail(std::error_code EC, std::string Context = {}) {
  if (Context.empty())
    fail(EC.message());
  else
    fail(Context + ": " + EC.message());
}

void fail(Error E, std::string Context = {}) {
  if (!Context.empty())
    Context += ": ";

  handleAllErrors(std::move(E), [&](const ErrorInfoBase &EIB) {
    if (Context.empty())
      fail(EIB.message());
    else
      fail(Context + EIB.message());
  });
}

int addMember(std::vector<NewArchiveMember> &Members, StringRef FileName,
              int Pos = -1) {
  Expected<NewArchiveMember> NMOrErr =
      NewArchiveMember::getFile(FileName, Deterministic);
  if (auto Error = NMOrErr.takeError()) {
    fail(std::move(Error), FileName);
    return 1;
  }
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
  if (auto Error = NMOrErr.takeError()) {
    fail(std::move(Error));
    return 1;
  }
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
#if LDC_LLVM_VER < 400
      auto NameOrErr = Child.getName();
      if (auto Error = NameOrErr.getError()) {
#else
      Expected<StringRef> NameOrErr = Child.getName();
      if (auto Error = NameOrErr.takeError()) {
#endif
        fail(std::move(Error));
        return 1;
      }
      StringRef Name = NameOrErr.get();

      auto MemberI = find_if(Members, [Name](StringRef Path) {
        return Name == sys::path::filename(Path);
      });

      if (MemberI == Members.end()) {
        if (int Status = addMember(Ret, Child))
          return Status;
      } else {
        if (int Status = addMember(Ret, *MemberI))
          return Status;
        Members.erase(MemberI);
      }
    }
    if (Err) {
      fail(std::move(Err));
      return 1;
    }
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
#if LDC_LLVM_VER >= 500
             ? object::Archive::K_DARWIN
#else
             ? object::Archive::K_BSD
#endif
             : object::Archive::K_GNU;
}

object::Archive::Kind getKindFromMember(const NewArchiveMember &Member) {
  auto OptionalObject =
      object::ObjectFile::createObjectFile(Member.Buf->getMemBufferRef());

  if (OptionalObject) {
    return isa<object::MachOObjectFile>(**OptionalObject)
#if LDC_LLVM_VER >= 500
               ? object::Archive::K_DARWIN
#else
               ? object::Archive::K_BSD
#endif
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

  const auto Result =
      writeArchive(ArchiveName, NewMembers, Symtab, Kind, Deterministic, Thin,
                   std::move(OldArchiveBuf));

  if (Result.second) {
    fail(Result.second, Result.first);
    return 1;
  }

  return 0;
}

int performWriteOperation() {
  // Create or open the archive object.
  auto Buf = MemoryBuffer::getFile(ArchiveName, -1, false);
  std::error_code EC = Buf.getError();
  if (EC && EC != errc::no_such_file_or_directory) {
    fail("error opening '" + ArchiveName + "': " + EC.message() + "!");
    return 1;
  }

  if (!EC) {
    Error Err = Error::success();
    object::Archive Archive(Buf.get()->getMemBufferRef(), Err);
    EC = errorToErrorCode(std::move(Err));
    if (EC) {
      fail(
          EC,
          ("error loading '" + ArchiveName + "': " + EC.message() + "!").str());
      return 1;
    }
    return performWriteOperation(&Archive, std::move(Buf.get()));
  }

  assert(EC == errc::no_such_file_or_directory);

  return performWriteOperation(nullptr, nullptr);
}

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

namespace ldc {

int ar(ArrayRef<const char *> args) {
  if (args.size() < 4 || strcmp(args[0], "llvm-ar") != 0 ||
      strcmp(args[1], "rcs") != 0) {
    llvm_unreachable(
        "Expected archiver command line: llvm-ar rcs <archive file> "
        "<object file> ...");
    return -1;
  }

  ArchiveName = args[2];

  auto membersSlice = args.slice(3);
  Members.clear();
  Members.insert(Members.end(), membersSlice.begin(), membersSlice.end());

  return performWriteOperation();
}

int lib(ArrayRef<const char *> args) {
  if (args.size() < 1 || strcmp(args[0], "llvm-lib.exe") != 0) {
    llvm_unreachable("Expected archiver command line: llvm-lib.exe ...");
    return -1;
  }

  return libDriverMain(args);
}

} // namespace ldc

#endif // LDC_LLVM_VER >= 309
