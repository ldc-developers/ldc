//===-- disassembler.cpp --------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the Boost Software License. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

#include "disassembler.h"

#include <algorithm>
#include <unordered_map>

#if LDC_LLVM_VER < 1700
#include "llvm/ADT/Triple.h"
#else
#include "llvm/TargetParser/Triple.h"
#endif
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCDisassembler/MCSymbolizer.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#if LDC_LLVM_VER >= 1400
#include "llvm/MC/TargetRegistry.h"
#else
#include "llvm/Support/TargetRegistry.h"
#endif
#include "llvm/Target/TargetMachine.h"

namespace {
template <typename T> std::unique_ptr<T> unique(T *ptr) {
  return std::unique_ptr<T>(ptr);
}

enum class Stage {
  Scan,
  Emit,
};

class SymTable final {
  llvm::MCContext &context;
  Stage stage;
  std::unordered_map<uint64_t, llvm::MCSymbol *> labelsPos;
  std::unordered_map<uint64_t, llvm::MCSymbol *> labelsTargets;
  std::unordered_map<uint64_t, llvm::MCSymbol *> externalSymbols;

public:
  SymTable(llvm::MCContext &ctx) : context(ctx) {}

  llvm::MCContext &getContext() { return context; }

  void setStage(Stage s) { stage = s; }

  Stage getStage() const { return stage; }

  void reset() {
    labelsPos.clear();
    labelsTargets.clear();
    externalSymbols.clear();
  }

  void addLabel(uint64_t pos, uint64_t target, llvm::StringRef name = {}) {
    if (auto label = getTargetLabel(target)) {
      labelsPos.insert({pos, label});
      return;
    }
    auto sym = name.empty() ? context.createTempSymbol("", false)
                            : context.getOrCreateSymbol(name);
    assert(nullptr != sym);
    labelsPos.insert({pos, sym});
    labelsTargets.insert({target, sym});
  }

  llvm::MCSymbol *getPosLabel(uint64_t pos) const {
    auto it = labelsPos.find(pos);
    if (labelsPos.end() != it) {
      return it->second;
    }
    return nullptr;
  }

  llvm::MCSymbol *getTargetLabel(uint64_t target) const {
    auto it = labelsTargets.find(target);
    if (labelsTargets.end() != it) {
      return it->second;
    }
    return nullptr;
  }

  void addExternalSymbolRel(uint64_t pos, llvm::StringRef name) {
    auto sym = context.getOrCreateSymbol(name);
    assert(nullptr != sym);
    externalSymbols.insert({pos, sym});
  }

  llvm::MCSymbol *getExternalSymbolRel(uint64_t pos) const {
    auto it = externalSymbols.find(pos);
    if (externalSymbols.end() != it) {
      return it->second;
    }
    return nullptr;
  }
};

void printFunction(const llvm::MCDisassembler &disasm,
                   const llvm::MCInstrAnalysis &mcia,
                   llvm::ArrayRef<uint8_t> data, SymTable &symTable,
                   const llvm::MCSubtargetInfo &sti,
                   llvm::MCStreamer &streamer) {
  const Stage stages[] = {Stage::Scan, Stage::Emit};
  for (auto stage : stages) {
    symTable.setStage(stage);
    uint64_t size = 0;
    for (uint64_t pos = 0; pos < static_cast<uint64_t>(data.size());
         pos += size) {
      llvm::MCInst inst;

      std::string comment;
      llvm::raw_string_ostream commentStream(comment);
      auto status = disasm.getInstruction(inst, size, data.slice(pos), pos,
                                          commentStream);

      switch (status) {
      case llvm::MCDisassembler::Fail:
        streamer.emitRawText("failed to disassemble");
        return;

      case llvm::MCDisassembler::SoftFail:
        streamer.emitRawText("potentially undefined instruction encoding:");
        LLVM_FALLTHROUGH;

      case llvm::MCDisassembler::Success:
        if (Stage::Scan == stage) {
          if (mcia.isBranch(inst) || mcia.isCall(inst)) {
            uint64_t target = 0;
            if (mcia.evaluateBranch(inst, pos, size, target)) {
              symTable.addLabel(pos, target);
            }
          }
        } else if (Stage::Emit == stage) {
          if (auto label = symTable.getTargetLabel(pos)) {
            streamer.emitLabel(label);
          }
          commentStream.flush();
          if (!comment.empty()) {
            streamer.AddComment(comment);
            comment.clear();
          }
          streamer.emitInstruction(inst, sti);
        }
        break;
      }
      assert(0 != size);
    }
  }
}

class Symbolizer final : public llvm::MCSymbolizer {
  SymTable &symTable;

  const llvm::MCExpr *createExpr(llvm::MCSymbol *sym, int64_t offset = 0) {
    assert(nullptr != sym);
    auto &ctx = symTable.getContext();
    auto expr = llvm::MCSymbolRefExpr::create(sym, ctx);
    if (0 == offset) {
      return expr;
    }
    auto off = llvm::MCConstantExpr::create(offset, ctx);
    return llvm::MCBinaryExpr::createAdd(expr, off, ctx);
  }

public:
  Symbolizer(llvm::MCContext &Ctx,
             std::unique_ptr<llvm::MCRelocationInfo> RelInfo,
             SymTable &symtable)
      : MCSymbolizer(Ctx, std::move(RelInfo)), symTable(symtable) {}

  virtual bool tryAddingSymbolicOperand(llvm::MCInst &Inst,
                                        llvm::raw_ostream & /*cStream*/,
                                        int64_t Value, uint64_t Address,
                                        bool IsBranch, uint64_t Offset,
                                        uint64_t /*InstSize*/) override {
    if (Stage::Emit == symTable.getStage()) {
      if (IsBranch) {
        if (auto label = symTable.getPosLabel(Address)) {
          Inst.addOperand(llvm::MCOperand::createExpr(createExpr(label)));
          return true;
        }
      }

      if (auto sym = symTable.getExternalSymbolRel(Address + Offset)) {
        Inst.addOperand(llvm::MCOperand::createExpr(createExpr(sym, Value)));
        return true;
      }
    }
    return false;
  }

  virtual void tryAddingPcLoadReferenceComment(llvm::raw_ostream &cStream,
                                               int64_t Value,
                                               uint64_t /*Address*/) override {
    if (Value >= 0) {
      if (auto sym =
              symTable.getExternalSymbolRel(static_cast<uint64_t>(Value))) {
        cStream << sym->getName();
      }
    }
  }
};

void processRelocations(SymTable &symTable, uint64_t offset,
                        const llvm::object::ObjectFile &object,
                        const llvm::object::SectionRef &sec) {
  for (const auto &reloc : sec.relocations()) {
    const auto symIt = reloc.getSymbol();
    if (object.symbol_end() != symIt) {
      const auto sym = *symIt;
      auto relOffet = reloc.getOffset();
      if (relOffet >= offset) {
        symTable.addExternalSymbolRel(relOffet - offset,
                                      llvm::cantFail(sym.getName()));
      }
    }
  }
}
}

void disassemble(const llvm::TargetMachine &tm,
                 const llvm::object::ObjectFile &object,
                 llvm::raw_ostream &os) {
  auto &target = tm.getTarget();

  auto mri = tm.getMCRegisterInfo();
  auto mai = tm.getMCAsmInfo();
  auto sti = tm.getMCSubtargetInfo();
  auto mii = tm.getMCInstrInfo();

  if (nullptr == mri || nullptr == mai || nullptr == sti || nullptr == mii) {
    // TODO: proper error handling
    return;
  }

  llvm::MCObjectFileInfo mofi;
  llvm::MCContext ctx(mai, mri, &mofi);
  mofi.InitMCObjectFileInfo(tm.getTargetTriple(), tm.isPositionIndependent(),
                            ctx, tm.getCodeModel() == llvm::CodeModel::Large);

  auto disasm = unique(target.createMCDisassembler(*sti, ctx));
  if (nullptr == disasm) {
    return;
  }

  SymTable symTable(ctx);
  disasm->setSymbolizer(std::make_unique<Symbolizer>(
      ctx, std::make_unique<llvm::MCRelocationInfo>(ctx), symTable));

  auto mcia = unique(target.createMCInstrAnalysis(mii));
  if (nullptr == mcia) {
    return;
  }

  auto mip = unique(
      target.createMCInstPrinter(tm.getTargetTriple(), 0, *mai, *mii, *mri));
  if (nullptr == mip) {
    return;
  }

  llvm::MCTargetOptions opts;
  auto mab = unique(target.createMCAsmBackend(*sti, *mri, opts));
  if (nullptr == mab) {
    return;
  }

  // Streamer takes ownership of mip mab
  auto asmStreamer = unique(target.createAsmStreamer(
      ctx, std::make_unique<llvm::formatted_raw_ostream>(os), true, true,
      mip.release(), nullptr, std::move(mab), false));
  if (nullptr == asmStreamer) {
    return;
  }

  asmStreamer->InitSections(false);

  std::unordered_map<uint64_t, std::vector<uint64_t>> sectionsToProcess;
  for (const auto &symbol : object.symbols()) {
    const auto secIt = llvm::cantFail(symbol.getSection());
    if (object.section_end() != secIt) {
      auto offset = llvm::cantFail(symbol.getValue());
      sectionsToProcess[secIt->getIndex()].push_back(offset);
    }
  }
  for (auto &sec : sectionsToProcess) {
    auto &vec = sec.second;
    std::sort(vec.begin(), vec.end());
    auto end = std::unique(vec.begin(), vec.end());
    vec.erase(end, vec.end());
  }

  for (const auto &symbol : object.symbols()) {
    const auto name = llvm::cantFail(symbol.getName());
    const auto secIt = llvm::cantFail(symbol.getSection());
    if (object.section_end() != secIt) {
      const auto sec = *secIt;
      llvm::StringRef data = llvm::cantFail(sec.getContents());

      if (llvm::object::SymbolRef::ST_Function ==
          llvm::cantFail(symbol.getType())) {
        symTable.reset();
        symTable.addLabel(0, 0, name); // Function start
        auto offset = llvm::cantFail(symbol.getValue());
        processRelocations(symTable, offset, object, sec);

        // TODO: something more optimal
        for (const auto &globalSec : object.sections()) {
          auto rs = globalSec.getRelocatedSection();
          if (rs && *rs == secIt) {
            processRelocations(symTable, offset, object, globalSec);
          }
        }
        auto size = data.size() - offset;
        auto &ranges = sectionsToProcess[sec.getIndex()];
        if (!ranges.empty()) {
          for (std::size_t i = 0; i < ranges.size() - 1; ++i) {
            if (ranges[i] == offset) {
              size = std::min(size, ranges[i + 1] - offset);
            }
          }
        }
        llvm::ArrayRef<uint8_t> buff(
            reinterpret_cast<const uint8_t *>(data.data() + offset), size);

        printFunction(*disasm, *mcia, buff, symTable, *sti, *asmStreamer);
        asmStreamer->emitRawText("");
      }
    }
  }
}
