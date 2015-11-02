//===-- gen/asm-x86.h - x86/x86_64 inline assembler handling ----*- C++ -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file originates from work by David Friedman for GDC released under
// the GPL 2 and Artistic licenses. See the LICENSE file for details.
//
//===----------------------------------------------------------------------===//
//
// Parses "DMD-style" x86/x86_64 inline assembly blocks and converts them to
// GDC/LLVM inline assembler syntax.
//
// This file is designed to be included twice, once with the ASM_X86_64 define
// set to get the 64 bit asm parser, and once without to get the 32 bit one.
// This is a direct result of merging two disparate but largely identical
// implementations, and should be refactored to just use a runtime parameter
// for choosing the architecture.
//
//===----------------------------------------------------------------------===//

#include "id.h"
#include "gen/llvmhelpers.h" // printLabelName
#include <cctype>

#ifndef ASM_X86_64
namespace AsmParserx8632 {
#else
namespace AsmParserx8664 {
#endif

typedef enum {
  Reg_Invalid = -1,
  Reg_EAX = 0,
  Reg_EBX,
  Reg_ECX,
  Reg_EDX,
  Reg_ESI,
  Reg_EDI,
  Reg_EBP,
  Reg_ESP,
  Reg_ST,
  Reg_ST1,
  Reg_ST2,
  Reg_ST3,
  Reg_ST4,
  Reg_ST5,
  Reg_ST6,
  Reg_ST7,
  Reg_MM0,
  Reg_MM1,
  Reg_MM2,
  Reg_MM3,
  Reg_MM4,
  Reg_MM5,
  Reg_MM6,
  Reg_MM7,
  Reg_XMM0,
  Reg_XMM1,
  Reg_XMM2,
  Reg_XMM3,
  Reg_XMM4,
  Reg_XMM5,
  Reg_XMM6,
  Reg_XMM7,

#ifdef ASM_X86_64
  Reg_RAX,
  Reg_RBX,
  Reg_RCX,
  Reg_RDX,
  Reg_RSI,
  Reg_RDI,
  Reg_RBP,
  Reg_RSP,
  Reg_R8,
  Reg_R9,
  Reg_R10,
  Reg_R11,
  Reg_R12,
  Reg_R13,
  Reg_R14,
  Reg_R15,
  Reg_R8B,
  Reg_R9B,
  Reg_R10B,
  Reg_R11B,
  Reg_R12B,
  Reg_R13B,
  Reg_R14B,
  Reg_R15B,
  Reg_R8W,
  Reg_R9W,
  Reg_R10W,
  Reg_R11W,
  Reg_R12W,
  Reg_R13W,
  Reg_R14W,
  Reg_R15W,
  Reg_R8D,
  Reg_R9D,
  Reg_R10D,
  Reg_R11D,
  Reg_R12D,
  Reg_R13D,
  Reg_R14D,
  Reg_R15D,
  Reg_XMM8,
  Reg_XMM9,
  Reg_XMM10,
  Reg_XMM11,
  Reg_XMM12,
  Reg_XMM13,
  Reg_XMM14,
  Reg_XMM15,
  Reg_RIP,
  Reg_SIL,
  Reg_DIL,
  Reg_BPL,
  Reg_SPL,
#endif

  Reg_EFLAGS,
  Reg_CS,
  Reg_DS,
  Reg_SS,
  Reg_ES,
  Reg_FS,
  Reg_GS,
  Reg_AX,
  Reg_BX,
  Reg_CX,
  Reg_DX,
  Reg_SI,
  Reg_DI,
  Reg_BP,
  Reg_SP,
  Reg_AL,
  Reg_AH,
  Reg_BL,
  Reg_BH,
  Reg_CL,
  Reg_CH,
  Reg_DL,
  Reg_DH,
  Reg_CR0,
  Reg_CR2,
  Reg_CR3,
  Reg_CR4,
  Reg_DR0,
  Reg_DR1,
  Reg_DR2,
  Reg_DR3,
  Reg_DR6,
  Reg_DR7,
  Reg_TR3,
  Reg_TR4,
  Reg_TR5,
  Reg_TR6,
  Reg_TR7
} Reg;

static const int N_Regs = /*gp*/ 8 + /*fp*/ 8 + /*mmx*/ 8 + /*sse*/ 8 +
                          /*seg*/ 6 + /*16bit*/ 8 + /*8bit*/ 8 + /*sys*/ 4 + 6 +
                          5 + /*flags*/ +1
#ifdef ASM_X86_64
                          + 8 /*RAX, etc*/
                          + 8 /*R8-15*/
                          + 4 /*SIL, etc. 8-bit*/
                          + 8 /*R8-15B*/
                          + 8 /*R8-15W*/
                          + 8 /*R8-15D*/
                          + 8 /*XMM8-15*/
                          + 1 /*RIP*/
#endif
    ;

#define NULL_TREE ""

static struct {
  const char *name;
  std::string gccName; // GAS will take upper case, but GCC won't (needed for
                       // the clobber list)
  Identifier *ident;
  char size;
  signed char baseReg; // %% todo: Reg, Reg_XX
} regInfo[N_Regs] = {
    {"EAX", NULL_TREE, nullptr, 4, Reg_EAX},
    {"EBX", NULL_TREE, nullptr, 4, Reg_EBX},
    {"ECX", NULL_TREE, nullptr, 4, Reg_ECX},
    {"EDX", NULL_TREE, nullptr, 4, Reg_EDX},
    {"ESI", NULL_TREE, nullptr, 4, Reg_ESI},
    {"EDI", NULL_TREE, nullptr, 4, Reg_EDI},
    {"EBP", NULL_TREE, nullptr, 4, Reg_EBP},
    {"ESP", NULL_TREE, nullptr, 4, Reg_ESP},
    {"ST", NULL_TREE, nullptr, 10, Reg_ST},
    {"ST(1)", NULL_TREE, nullptr, 10, Reg_ST1},
    {"ST(2)", NULL_TREE, nullptr, 10, Reg_ST2},
    {"ST(3)", NULL_TREE, nullptr, 10, Reg_ST3},
    {"ST(4)", NULL_TREE, nullptr, 10, Reg_ST4},
    {"ST(5)", NULL_TREE, nullptr, 10, Reg_ST5},
    {"ST(6)", NULL_TREE, nullptr, 10, Reg_ST6},
    {"ST(7)", NULL_TREE, nullptr, 10, Reg_ST7},
    {"MM0", NULL_TREE, nullptr, 8, Reg_MM0},
    {"MM1", NULL_TREE, nullptr, 8, Reg_MM1},
    {"MM2", NULL_TREE, nullptr, 8, Reg_MM2},
    {"MM3", NULL_TREE, nullptr, 8, Reg_MM3},
    {"MM4", NULL_TREE, nullptr, 8, Reg_MM4},
    {"MM5", NULL_TREE, nullptr, 8, Reg_MM5},
    {"MM6", NULL_TREE, nullptr, 8, Reg_MM6},
    {"MM7", NULL_TREE, nullptr, 8, Reg_MM7},
    {"XMM0", NULL_TREE, nullptr, 16, Reg_XMM0},
    {"XMM1", NULL_TREE, nullptr, 16, Reg_XMM1},
    {"XMM2", NULL_TREE, nullptr, 16, Reg_XMM2},
    {"XMM3", NULL_TREE, nullptr, 16, Reg_XMM3},
    {"XMM4", NULL_TREE, nullptr, 16, Reg_XMM4},
    {"XMM5", NULL_TREE, nullptr, 16, Reg_XMM5},
    {"XMM6", NULL_TREE, nullptr, 16, Reg_XMM6},
    {"XMM7", NULL_TREE, nullptr, 16, Reg_XMM7},

#ifdef ASM_X86_64
    {"RAX", NULL_TREE, nullptr, 8, Reg_RAX},
    {"RBX", NULL_TREE, nullptr, 8, Reg_RBX},
    {"RCX", NULL_TREE, nullptr, 8, Reg_RCX},
    {"RDX", NULL_TREE, nullptr, 8, Reg_RDX},
    {"RSI", NULL_TREE, nullptr, 8, Reg_RSI},
    {"RDI", NULL_TREE, nullptr, 8, Reg_RDI},
    {"RBP", NULL_TREE, nullptr, 8, Reg_RBP},
    {"RSP", NULL_TREE, nullptr, 8, Reg_RSP},
    {"R8", NULL_TREE, nullptr, 8, Reg_R8},
    {"R9", NULL_TREE, nullptr, 8, Reg_R9},
    {"R10", NULL_TREE, nullptr, 8, Reg_R10},
    {"R11", NULL_TREE, nullptr, 8, Reg_R11},
    {"R12", NULL_TREE, nullptr, 8, Reg_R12},
    {"R13", NULL_TREE, nullptr, 8, Reg_R13},
    {"R14", NULL_TREE, nullptr, 8, Reg_R14},
    {"R15", NULL_TREE, nullptr, 8, Reg_R15},
    {"R8B", NULL_TREE, nullptr, 1, Reg_R8},
    {"R9B", NULL_TREE, nullptr, 1, Reg_R9},
    {"R10B", NULL_TREE, nullptr, 1, Reg_R10},
    {"R11B", NULL_TREE, nullptr, 1, Reg_R11},
    {"R12B", NULL_TREE, nullptr, 1, Reg_R12},
    {"R13B", NULL_TREE, nullptr, 1, Reg_R13},
    {"R14B", NULL_TREE, nullptr, 1, Reg_R14},
    {"R15B", NULL_TREE, nullptr, 1, Reg_R15},
    {"R8W", NULL_TREE, nullptr, 2, Reg_R8},
    {"R9W", NULL_TREE, nullptr, 2, Reg_R9},
    {"R10W", NULL_TREE, nullptr, 2, Reg_R10},
    {"R11W", NULL_TREE, nullptr, 2, Reg_R11},
    {"R12W", NULL_TREE, nullptr, 2, Reg_R12},
    {"R13W", NULL_TREE, nullptr, 2, Reg_R13},
    {"R14W", NULL_TREE, nullptr, 2, Reg_R14},
    {"R15W", NULL_TREE, nullptr, 2, Reg_R15},
    {"R8D", NULL_TREE, nullptr, 4, Reg_R8},
    {"R9D", NULL_TREE, nullptr, 4, Reg_R9},
    {"R10D", NULL_TREE, nullptr, 4, Reg_R10},
    {"R11D", NULL_TREE, nullptr, 4, Reg_R11},
    {"R12D", NULL_TREE, nullptr, 4, Reg_R12},
    {"R13D", NULL_TREE, nullptr, 4, Reg_R13},
    {"R14D", NULL_TREE, nullptr, 4, Reg_R14},
    {"R15D", NULL_TREE, nullptr, 4, Reg_R15},
    {"XMM8", NULL_TREE, nullptr, 16, Reg_XMM8},
    {"XMM9", NULL_TREE, nullptr, 16, Reg_XMM9},
    {"XMM10", NULL_TREE, nullptr, 16, Reg_XMM10},
    {"XMM11", NULL_TREE, nullptr, 16, Reg_XMM11},
    {"XMM12", NULL_TREE, nullptr, 16, Reg_XMM12},
    {"XMM13", NULL_TREE, nullptr, 16, Reg_XMM13},
    {"XMM14", NULL_TREE, nullptr, 16, Reg_XMM14},
    {"XMM15", NULL_TREE, nullptr, 16, Reg_XMM15},
    {"RIP", NULL_TREE, nullptr, 8, Reg_RIP},
    {"SIL", NULL_TREE, nullptr, 1, Reg_SIL},
    {"DIL", NULL_TREE, nullptr, 1, Reg_DIL},
    {"BPL", NULL_TREE, nullptr, 1, Reg_BPL},
    {"SPL", NULL_TREE, nullptr, 1, Reg_SPL},
#endif

    {"FLAGS", NULL_TREE, nullptr, 0,
     Reg_EFLAGS}, // the gcc name is "flags"; not used in assembler input
    {"CS", NULL_TREE, nullptr, 2, -1},
    {"DS", NULL_TREE, nullptr, 2, -1},
    {"SS", NULL_TREE, nullptr, 2, -1},
    {"ES", NULL_TREE, nullptr, 2, -1},
    {"FS", NULL_TREE, nullptr, 2, -1},
    {"GS", NULL_TREE, nullptr, 2, -1},
    {"AX", NULL_TREE, nullptr, 2, Reg_EAX},
    {"BX", NULL_TREE, nullptr, 2, Reg_EBX},
    {"CX", NULL_TREE, nullptr, 2, Reg_ECX},
    {"DX", NULL_TREE, nullptr, 2, Reg_EDX},
    {"SI", NULL_TREE, nullptr, 2, Reg_ESI},
    {"DI", NULL_TREE, nullptr, 2, Reg_EDI},
    {"BP", NULL_TREE, nullptr, 2, Reg_EBP},
    {"SP", NULL_TREE, nullptr, 2, Reg_ESP},
    {"AL", NULL_TREE, nullptr, 1, Reg_EAX},
    {"AH", NULL_TREE, nullptr, 1, Reg_EAX},
    {"BL", NULL_TREE, nullptr, 1, Reg_EBX},
    {"BH", NULL_TREE, nullptr, 1, Reg_EBX},
    {"CL", NULL_TREE, nullptr, 1, Reg_ECX},
    {"CH", NULL_TREE, nullptr, 1, Reg_ECX},
    {"DL", NULL_TREE, nullptr, 1, Reg_EDX},
    {"DH", NULL_TREE, nullptr, 1, Reg_EDX},
    {"CR0", NULL_TREE, nullptr, 0, -1},
    {"CR2", NULL_TREE, nullptr, 0, -1},
    {"CR3", NULL_TREE, nullptr, 0, -1},
    {"CR4", NULL_TREE, nullptr, 0, -1},
    {"DR0", NULL_TREE, nullptr, 0, -1},
    {"DR1", NULL_TREE, nullptr, 0, -1},
    {"DR2", NULL_TREE, nullptr, 0, -1},
    {"DR3", NULL_TREE, nullptr, 0, -1},
    {"DR6", NULL_TREE, nullptr, 0, -1},
    {"DR7", NULL_TREE, nullptr, 0, -1},
    {"TR3", NULL_TREE, nullptr, 0, -1},
    {"TR4", NULL_TREE, nullptr, 0, -1},
    {"TR5", NULL_TREE, nullptr, 0, -1},
    {"TR6", NULL_TREE, nullptr, 0, -1},
    {"TR7", NULL_TREE, nullptr, 0, -1}};

typedef enum {
  No_Type_Needed,
  Int_Types,
  Word_Types, // same as Int_Types, but byte is not allowed
  FP_Types,
  FPInt_Types,
  Byte_NoType, // byte only, but no type suffix
} TypeNeeded;

typedef enum { No_Link, Out_Mnemonic, Next_Form } OpLink;

typedef enum {
  Clb_SizeAX = 0x01,
  Clb_SizeDXAX = 0x02,
  Clb_EAX = 0x03,
#ifndef ASM_X86_64
  Clb_DXAX_Mask = 0x03,
#else
  Clb_DXAX_Mask = 0x103,
  Clb_SizeRDXRAX = 0x100,
#endif

  Clb_Flags = 0x04,
  Clb_DI = 0x08,
  Clb_SI = 0x10,
  Clb_CX = 0x20,
  Clb_ST = 0x40,
  Clb_SP = 0x80 // Doesn't actually let GCC know the frame pointer is modified
} ImplicitClober;

// "^ +/..\([A-Za-z_0-9]+\).*" -> "    \1,"
typedef enum {
  Op_Invalid,
  Op_Adjust,
  Op_Dst,
  Op_Upd,
  Op_DstW,
  Op_DstF,
  Op_UpdF,
  Op_DstSrc,
  Op_DstSrcF,
  Op_UpdSrcF,
  Op_DstSrcFW,
  Op_UpdSrcFW,
  Op_DstSrcSSE,
  Op_DstSrcMMX,
  Op_DstSrcImmS,
  Op_DstSrcImmM,
  Op_ExtSrcImmS,
  Op_UpdSrcShft,
  Op_DstSrcNT,
  Op_UpdSrcNT,
  Op_DstMemNT,
  Op_DstRMBNT,
  Op_DstRMWNT,
  Op_UpdUpd,
  Op_UpdUpdF,
  Op_Src,
  Op_SrcRMWNT,
  Op_SrcW,
  Op_SrcImm,
  Op_Src_DXAXF,
  Op_SrcMemNT,
  Op_SrcMemNTF,
  Op_SrcSrc,
  Op_SrcSrcF,
  Op_SrcSrcFW,
  Op_SrcSrcSSEF,
  Op_SrcSrcMMX,
  Op_Shift,
  Op_Branch,
  Op_CBranch,
  Op_0,
  Op_0_AX,
  Op_0_DXAX,
  Op_Loop,
  Op_Flags,
  Op_F0_ST,
  Op_F0_P,
  Op_Fs_P,
  Op_Fis,
  Op_Fis_ST,
  Op_Fis_P,
  Op_Fid,
  Op_Fid_P,
  Op_FidR_P,
  Op_Ffd,
  Op_FfdR,
  Op_Ffd_P,
  Op_FfdR_P,
  Op_FfdRR_P,
  Op_Fd_P,
  Op_FdST,
  Op_FMath,
  Op_FMath0,
  Op_FMath2,
  Op_FdSTiSTi,
  Op_FdST0ST1,
  Op_FPMath,
  Op_FCmp,
  Op_FCmp1,
  Op_FCmpP,
  Op_FCmpP1,
  Op_FCmpFlg,
  Op_FCmpFlgP,
  Op_fld,
  Op_fldR,
  Op_fxch,
  Op_fxch1,
  Op_fxch0,
  Op_SizedStack,
  Op_bound,
  Op_bswap,
  Op_cmps,
  Op_cmpsd,
  Op_cmpsX,
  Op_cmpxchg,
#ifdef ASM_X86_64
  Op_cmpxchg16b,
#endif
  Op_cmpxchg8b,
  Op_cpuid,
  Op_enter,
  Op_fdisi,
  Op_feni,
  Op_fsetpm,
  Op_fXstsw,
  Op_imul,
  Op_imul2,
  Op_imul1,
  Op_in,
  Op_ins,
  Op_insX,
  Op_iret,
  Op_iretd,
#ifdef ASM_X86_64
  Op_iretq,
#endif
  Op_lods,
  Op_lodsX,
  Op_movs,
  Op_movsd,
  Op_movsX,
  Op_movsx,
  Op_movzx,
  Op_mul,
  Op_out,
  Op_outs,
  Op_outsX,
  Op_push,
  Op_ret,
  Op_retf,
  Op_scas,
  Op_scasX,
  Op_stos,
  Op_stosX,
  Op_xgetbv,
  Op_xlat,
  N_AsmOpInfo,
  Op_Align,
  Op_Even,
  Op_Naked,
  Op_db,
  Op_ds,
  Op_di,
  Op_dl,
  Op_df,
  Op_dd,
  Op_de
} AsmOp;

typedef enum {
  Opr_None = 0,
  OprC_MRI = 1,
  OprC_MR = 2,
  OprC_Mem = 3,
  OprC_Reg = 4,
  OprC_Imm = 5,
  OprC_SSE = 6,
  OprC_SSE_Mem = 7,
  OprC_R32 = 8,
  OprC_RWord = 9,
  OprC_RFP = 10,
  OprC_AbsRel = 11,
  OprC_Relative = 12,
  OprC_Port = 13, // DX or imm
  OprC_AX = 14,   // AL,AX,EAX
  OprC_DX = 15,   // only DX
  OprC_MMX = 16,
  OprC_MMX_Mem = 17,
  OprC_Shift = 18, // imm or CL

  Opr_ClassMask = 0x1f,

  Opr_Dest = 0x20,
  Opr_Update = 0x60,

  Opr_NoType = 0x80,
} OprVals;

typedef struct {
} AsmOprInfo;

typedef unsigned char Opr;

typedef struct {
  Opr operands[3];
#ifndef ASM_X86_64
  unsigned char needsType : 3, implicitClobbers : 8, linkType : 2;
#else
  unsigned short needsType : 3, implicitClobbers : 9, linkType : 2;
#endif
  unsigned link;

  unsigned nOperands() {
    if (!operands[0]) {
      return 0;
    }
    if (!operands[1]) {
      return 1;
    }
    if (!operands[2]) {
      return 2;
    }
    return 3;
  }
} AsmOpInfo;

typedef enum {
  Mn_fdisi,
  Mn_feni,
  Mn_fsetpm,
  Mn_iretw,
  Mn_iret,
#ifdef ASM_X86_64
  Mn_iretq,
#endif
  Mn_lret,
  Mn_cmpxchg8b,
#ifdef ASM_X86_64
  Mn_cmpxchg16b,
#endif
  N_AltMn
} Alternate_Mnemonics;

static const char *alternateMnemonics[N_AltMn] = {
    ".byte 0xdb, 0xe1",
    ".byte 0xdb, 0xe0",
    ".byte 0xdb, 0xe4",
    "iretw",
    "iret",
#ifdef ASM_X86_64
    "iretq",
#endif
    "lret",
    "cmpxchg8b",
#ifdef ASM_X86_64
    "cmpxchg16b",
#endif
};

#define mri OprC_MRI
#define mr OprC_MR
#define mem OprC_Mem
// for now mfp=mem
#define mfp OprC_Mem
#define reg OprC_Reg
#define imm OprC_Imm
#define sse OprC_SSE
#define ssem OprC_SSE_Mem
#define mmx OprC_MMX
#define mmxm OprC_MMX_Mem
#define r32 OprC_R32
#define rw OprC_RWord
#define rfp OprC_RFP
#define port OprC_Port
#define ax OprC_AX
#define dx OprC_DX
#define shft OprC_Shift
#define D Opr_Dest
#define U Opr_Update
#define N Opr_NoType

// D=dest, N=notype
static AsmOpInfo asmOpInfo[N_AsmOpInfo] = {
    /* Op_Invalid   */ {},
    /* Op_Adjust    */ {{0, 0, 0}, 0, Clb_EAX /*just AX*/},
    /* Op_Dst       */ {{D | mr, 0, 0}, 1},
    /* Op_Upd       */ {{U | mr, 0, 0}, 1},
    /* Op_DstW      */ {{D | mr, 0, 0}, Word_Types},
    /* Op_DstF      */ {{D | mr, 0, 0}, 1, Clb_Flags},
    /* Op_UpdF      */ {{U | mr, 0, 0}, 1, Clb_Flags},
    /* Op_DstSrc    */ {{D | mr, mri, 0}, /**/ 1},
    /* Op_DstSrcF   */ {{D | mr, mri, 0}, /**/ 1, Clb_Flags},
    /* Op_UpdSrcF   */ {{U | mr, mri, 0}, /**/ 1, Clb_Flags},
    /* Op_DstSrcFW  */ {{D | mr, mri, 0}, /**/ Word_Types, Clb_Flags},
    /* Op_UpdSrcFW  */ {{U | mr, mri, 0}, /**/ Word_Types, Clb_Flags},
    /* Op_DstSrcSSE */ {{U | sse, ssem, 0}},       // some may not be update %%
    /* Op_DstSrcMMX */ {{U | mmx, mmxm, 0}},       // some may not be update %%
    /* Op_DstSrcImmS*/ {{U | sse, ssem, N | imm}}, // some may not be update %%
    /* Op_DstSrcImmM*/ {{U | mmx, mmxm, N | imm}}, // some may not be update %%
    /* Op_ExtSrcImmS*/ {{D | mr, sse, N | imm}},   // used for extractps
    /* Op_UpdSrcShft*/ {{U | mr, reg, N | shft}, 1, Clb_Flags}, // 16/32 only
    /* Op_DstSrcNT  */ {{D | mr, mr, 0},
                        0}, // used for movd .. operands can be rm32,sse,mmx
    /* Op_UpdSrcNT  */ {{U | mr, mr, 0},
                        0}, // used for movd .. operands can be rm32,sse,mmx
    /* Op_DstMemNT  */ {{D | mem, 0, 0}},
    /* Op_DstRMBNT  */ {{D | mr, 0, 0}, Byte_NoType},
    /* Op_DstRMWNT  */ {{D | mr, 0, 0}},
    /* Op_UpdUpd    */ {{U | mr, U | mr, 0}, /**/ 1},
    /* Op_UpdUpdF   */ {{U | mr, U | mr, 0}, /**/ 1, Clb_Flags},
    /* Op_Src       */ {{mri, 0, 0}, 1},
    /* Op_SrcRMWNT  */ {{mr, 0, 0}, 0},
    /* Op_SrcW      */ {{mri, 0, 0}, Word_Types},
    /* Op_SrcImm    */ {{imm, 0, 0}},
    /* Op_Src_DXAXF */ {{mr, 0, 0}, 1, Clb_SizeDXAX | Clb_Flags},
    /* Op_SrcMemNT  */ {{mem, 0, 0}},
    /* Op_SrcMemNTF */ {{mem, 0, 0}, 0, Clb_Flags},
    /* Op_SrcSrc    */ {{mr, mri, 0}, 1},
    /* Op_SrcSrcF   */ {{mr, mri, 0}, 1, Clb_Flags},
    /* Op_SrcSrcFW  */ {{mr, mri, 0}, Word_Types, Clb_Flags},
    /* Op_SrcSrcSSEF*/ {{sse, ssem, 0}, 0, Clb_Flags},
    /* Op_SrcSrcMMX */ {{mmx, mmx, 0}},
    /* Op_Shift     */ {{D | mr, N | shft, 0}, /**/ 1, Clb_Flags},
    /* Op_Branch    */ {{mri, 0, 0}},
    /* Op_CBranch   */ {{imm, 0, 0}},
    /* Op_0         */ {{0, 0, 0}},
    /* Op_0_AX      */ {{0, 0, 0}, 0, Clb_SizeAX},
    /* Op_0_DXAX    */ {{0, 0, 0}, 0, Clb_SizeDXAX}, // but for cwd/cdq -- how
                                                     // do know the size..
    /* Op_Loop      */ {{imm, 0, 0}, 0, Clb_CX},
    /* Op_Flags     */ {{0, 0, 0}, 0, Clb_Flags},
    /* Op_F0_ST     */ {{0, 0, 0}, 0, Clb_ST},
    /* Op_F0_P      */ {{0, 0, 0}, 0, Clb_ST}, // push, pops, etc. not sure how
                                               // to inform gcc..
    /* Op_Fs_P      */ {{mem, 0, 0}, 0, Clb_ST},   // "
    /* Op_Fis       */ {{mem, 0, 0}, FPInt_Types}, // only 16bit and 32bit, DMD
                                                   // defaults to 16bit
    /* Op_Fis_ST    */ {{mem, 0, 0}, FPInt_Types, Clb_ST}, // "
    /* Op_Fis_P     */ {{mem, 0, 0},
                        FPInt_Types,
                        Clb_ST}, // push and pop, fild so also 64 bit
    /* Op_Fid       */ {{D | mem, 0, 0}, FPInt_Types}, // only 16bit and 32bit,
                                                       // DMD defaults to 16bit
    /* Op_Fid_P     */ {{D | mem, 0, 0},
                        FPInt_Types,
                        Clb_ST,
                        Next_Form,
                        Op_FidR_P}, // push and pop, fild so also 64 bit
    /* Op_FidR_P    */ {{D | mem, rfp, 0},
                        FPInt_Types,
                        Clb_ST}, // push and pop, fild so also 64 bit
    /* Op_Ffd       */ {{D | mfp, 0, 0},
                        FP_Types,
                        0,
                        Next_Form,
                        Op_FfdR}, // only 16bit and 32bit, DMD defaults to
                                  // 16bit, reg form doesn't need type
    /* Op_FfdR      */ {{D | rfp, 0, 0}},
    /* Op_Ffd_P     */ {{D | mfp, 0, 0},
                        FP_Types,
                        Clb_ST,
                        Next_Form,
                        Op_FfdR_P}, // pop, fld so also 80 bit, "
    /* Op_FfdR_P    */ {{D | rfp, 0, 0}, 0, Clb_ST, Next_Form, Op_FfdRR_P},
    /* Op_FfdRR_P   */ {{D | rfp, rfp, 0}, 0, Clb_ST},
    /* Op_Fd_P      */ {{D | mem, 0, 0}, 0, Clb_ST}, // "
    /* Op_FdST      */ {{D | rfp, 0, 0}},
    /* Op_FMath     */ {{mfp, 0, 0},
                        FP_Types,
                        Clb_ST,
                        Next_Form,
                        Op_FMath0}, // and only single or double prec
    /* Op_FMath0    */ {{0, 0, 0}, 0, Clb_ST, Next_Form, Op_FMath2}, // pops
    /* Op_FMath2    */ {{D | rfp, rfp, 0},
                        0,
                        Clb_ST,
                        Next_Form,
                        Op_FdST0ST1}, // and only single or double prec
    /* Op_FdSTiSTi  */ {{D | rfp, rfp, 0}},
    /* Op_FdST0ST1  */ {{0, 0, 0}},
    /* Op_FPMath    */ {{D | rfp, rfp, 0},
                        0,
                        Clb_ST,
                        Next_Form,
                        Op_F0_P}, // pops
    /* Op_FCmp      */ {{mfp, 0, 0},
                        FP_Types,
                        0,
                        Next_Form,
                        Op_FCmp1}, // DMD defaults to float ptr
    /* Op_FCmp1     */ {{rfp, 0, 0}, 0, 0, Next_Form, Op_0},
    /* Op_FCmpP     */ {{mfp, 0, 0}, FP_Types, 0, Next_Form, Op_FCmpP1}, // pops
    /* Op_FCmpP1    */ {{rfp, 0, 0}, 0, 0, Next_Form, Op_F0_P},          // pops
    /* Op_FCmpFlg   */ {{rfp, 0, 0}, 0, Clb_Flags},
    /* Op_FCmpFlgP  */ {{rfp, 0, 0}, 0, Clb_Flags}, // pops
    /* Op_fld       */ {{mfp, 0, 0}, FP_Types, Clb_ST, Next_Form, Op_fldR},
    /* Op_fldR      */ {{rfp, 0, 0}, 0, Clb_ST},
    /* Op_fxch      */ {{D | rfp, D | rfp, 0},
                        0,
                        Clb_ST,
                        Next_Form,
                        Op_fxch1}, // not in intel manual?, but DMD allows it
                                   // (gas won't), second arg must be ST
    /* Op_fxch1     */ {{D | rfp, 0, 0}, 0, Clb_ST, Next_Form, Op_fxch0},
    /* Op_fxch0     */ {{0, 0, 0}, 0, Clb_ST}, // Also clobbers ST(1)
    /* Op_SizedStack*/ {{0, 0, 0}, 0, Clb_SP}, // type suffix special case
    /* Op_bound     */ {{mr, mri, 0},
                        Word_Types}, // operands *not* reversed for gas
    /* Op_bswap     */ {{D | r32, 0, 0}},
    /* Op_cmps      */ {{mem, mem, 0}, 1, Clb_DI | Clb_SI | Clb_Flags},
    /* Op_cmpsd     */ {{0, 0, 0},
                        0,
                        Clb_DI | Clb_SI | Clb_Flags,
                        Next_Form,
                        Op_DstSrcImmS},
    /* Op_cmpsX     */ {{0, 0, 0}, 0, Clb_DI | Clb_SI | Clb_Flags},
    /* Op_cmpxchg   */ {{D | mr, reg, 0}, 1, Clb_SizeAX | Clb_Flags},
#ifdef ASM_X86_64
    /* Op_cmpxchg16b */ {{D | mem /*128*/, 0, 0},
                         0,
                         Clb_SizeRDXRAX /*64*/ | Clb_Flags,
                         Out_Mnemonic,
                         Mn_cmpxchg16b},
#endif
    /* Op_cmpxchg8b */ {{D | mem /*64*/, 0, 0},
                        0,
                        Clb_SizeDXAX /*32*/ | Clb_Flags,
                        Out_Mnemonic,
                        Mn_cmpxchg8b},
    /* Op_cpuid     */ {{0, 0, 0}}, // Clobbers eax, ebx, ecx, and edx. Handled
                                    // specially below.
    /* Op_enter     */ {{imm, imm, 0}}, // operands *not* reversed for gas, %%
                                        // inform gcc of EBP clobber?,
    /* Op_fdisi     */ {{0, 0, 0}, 0, 0, Out_Mnemonic, Mn_fdisi},
    /* Op_feni      */ {{0, 0, 0}, 0, 0, Out_Mnemonic, Mn_feni},
    /* Op_fsetpm    */ {{0, 0, 0}, 0, 0, Out_Mnemonic, Mn_fsetpm},
    /* Op_fXstsw    */ {{D | mr, 0, 0}}, // ax is the only allowed register
    /* Op_imul      */ {{D | reg, mr, imm},
                        1,
                        Clb_Flags,
                        Next_Form,
                        Op_imul2}, // 16/32 only
    /* Op_imul2     */ {{D | reg, mri, 0},
                        1,
                        Clb_Flags,
                        Next_Form,
                        Op_imul1}, // 16/32 only
    /* Op_imul1     */ {{mr, 0, 0}, 1, Clb_Flags | Clb_SizeDXAX},
    /* Op_in        */ {{D | ax, N | port, 0}, 1},
    /* Op_ins       */ {{mem, N | dx, 0},
                        1,
                        Clb_DI}, // can't override ES segment for this one
    /* Op_insX      */ {{0, 0, 0},
                        0,
                        Clb_DI}, // output segment overrides %% needs work
    /* Op_iret      */ {{0, 0, 0}, 0, 0, Out_Mnemonic, Mn_iretw},
    /* Op_iretd     */ {{0, 0, 0}, 0, 0, Out_Mnemonic, Mn_iret},
#ifdef ASM_X86_64
    /* Op_iretq     */ {{0, 0, 0}, 0, 0, Out_Mnemonic, Mn_iretq},
#endif
    /* Op_lods      */ {{mem, 0, 0}, 1, Clb_SI},
    /* Op_lodsX     */ {{0, 0, 0}, 0, Clb_SI},
    /* Op_movs      */ {{mem, mem, 0},
                        1,
                        Clb_DI | Clb_SI}, // only src/DS can be overridden
    /* Op_movsd     */ {{0, 0, 0},
                        0,
                        Clb_DI | Clb_SI,
                        Next_Form,
                        Op_DstSrcSSE}, // %% gas doesn't accept movsd .. has to
                                       // movsl
    /* Op_movsX     */ {{0, 0, 0}, 0, Clb_DI | Clb_SI},
    /* Op_movsx     */ {{D | reg, mr, 0}, 1}, // type suffix is special case
    /* Op_movzx     */ {{D | reg, mr, 0}, 1}, // type suffix is special case
    /* Op_mul       */ {{U | ax, mr, 0},
                        1,
                        Clb_SizeDXAX | Clb_Flags,
                        Next_Form,
                        Op_Src_DXAXF},
    /* Op_out       */ {{N | port, ax, 0}, 1},
    /* Op_outs      */ {{N | dx, mem, 0}, 1, Clb_SI},
    /* Op_outsX     */ {{0, 0, 0}, 0, Clb_SI},
#ifndef ASM_X86_64
    /* Op_push      */ {{mri, 0, 0}, Word_Types, Clb_SP}, // would be Op_SrcW,
                                                          // but DMD defaults to
                                                          // 32-bit for
                                                          // immediate form
#else
    /* Op_push      */ {{mri, 0, 0}, 0, Clb_SP}, // would be Op_SrcW, but DMD
                                                 // defaults to 32-bit for
                                                 // immediate form
#endif
    /* Op_ret       */ {{imm, 0, 0}, 0, 0, Next_Form, Op_0},
    /* Op_retf      */ {{0, 0, 0}, 0, 0, Out_Mnemonic, Mn_lret},
    /* Op_scas      */ {{mem, 0, 0}, 1, Clb_DI | Clb_Flags},
    /* Op_scasX     */ {{0, 0, 0}, 0, Clb_DI | Clb_Flags},
    /* Op_stos      */ {{mem, 0, 0}, 1, Clb_DI},
    /* Op_stosX     */ {{0, 0, 0}, 0, Clb_DI},
#ifndef ASM_X86_64
    /* Op_xgetbv    */ {{0, 0, 0}, 0, Clb_SizeDXAX},
#else
    /* Op_xgetbv    */ {{0, 0, 0}, 0, Clb_SizeRDXRAX},
#endif
    /* Op_xlat      */ {{mem, 0, 0}, 0, Clb_SizeAX}

    /// * Op_arpl      */  { D|mr,  reg }, // 16 only -> DstSrc
    /// * Op_bsX       */  {   rw,  mrw,  0,    1, Clb_Flags },//->srcsrcf
    /// * Op_bt        */  {   mrw, riw,  0,    1, Clb_Flags },//->srcsrcf
    /// * Op_btX       */  { D|mrw, riw,  0,    1, Clb_Flags },//->dstsrcf ..
    /// immediate does not contribute to size
    /// * Op_cmovCC    */  { D|rw,  mrw,  0,    1 } // ->dstsrc
};

#undef mri
#undef mr
#undef mem
#undef mfp
#undef reg
#undef imm
#undef sse
#undef ssem
#undef mmx
#undef mmxm
#undef r32
#undef rw
#undef rfp
#undef port
#undef ax
#undef dx
#undef shft
#undef D
#undef U
#undef N

typedef struct {
  const char *inMnemonic;
  AsmOp asmOp;
} AsmOpEnt;

/* Some opcodes have data size restrictions, which we don't check

   cmov, l<segreg> ?, lea, lsl, shld

   todo: push <immediate> is always the 32-bit form, even tho push <mem> is
   16-bit
*/

// WARNING: This table is expected to be SORTED ALPHABETICALLY.
static AsmOpEnt opData[] = {
#ifndef ASM_X86_64
    {"aaa", Op_Adjust},
    {"aad", Op_Adjust},
    {"aam", Op_Adjust},
    {"aas", Op_Adjust},
#endif
    {"adc", Op_UpdSrcF},
    {"add", Op_UpdSrcF},
    {"addpd", Op_DstSrcSSE},
    {"addps", Op_DstSrcSSE},
#ifdef ASM_X86_64
    {"addq", Op_DstSrcSSE}, // ?
#endif
    {"addsd", Op_DstSrcSSE},
    {"addss", Op_DstSrcSSE},
    {"addsubpd", Op_DstSrcSSE},
    {"addsubps", Op_DstSrcSSE},
    {"aesdec", Op_DstSrcSSE},
    {"aesdeclast", Op_DstSrcSSE},
    {"aesenc", Op_DstSrcSSE},
    {"aesenclast", Op_DstSrcSSE},
    {"aesimc", Op_DstSrcSSE},
    {"aeskeygenassist", Op_DstSrcImmS},
#ifndef ASM_X86_64
    {"align", Op_Align},
#endif
    {"and", Op_UpdSrcF},
    {"andnpd", Op_DstSrcSSE},
    {"andnps", Op_DstSrcSSE},
    {"andpd", Op_DstSrcSSE},
    {"andps", Op_DstSrcSSE},
#ifndef ASM_X86_64
    {"arpl", Op_UpdSrcNT},
#endif
    {"blendpd", Op_DstSrcImmS},
    {"blendps", Op_DstSrcImmS},
    {"blendvpd", Op_DstSrcSSE},
    {"blendvps", Op_DstSrcSSE},
    {"bound", Op_bound},
    {"bsf", Op_SrcSrcFW},
    {"bsr", Op_SrcSrcFW},
    {"bswap", Op_bswap},
    {"bt", Op_SrcSrcFW},
    {"btc", Op_UpdSrcFW},
    {"btr", Op_UpdSrcFW},
    {"bts", Op_UpdSrcFW},
    {"call", Op_Branch},
#ifdef ASM_X86_64
    {"callf", Op_Branch}, // ?
#endif
    {"cbw", Op_0_AX},
#ifndef ASM_X86_64
    {"cdq", Op_0_DXAX},
#else
    {"cdqe", Op_0_DXAX},
#endif
    {"clc", Op_Flags},
    {"cld", Op_Flags},
    {"clflush", Op_SrcMemNT},
    {"cli", Op_Flags},
    {"clts", Op_0},
    {"cmc", Op_Flags},
    {"cmova", Op_DstSrc},
    {"cmovae", Op_DstSrc},
    {"cmovb", Op_DstSrc},
    {"cmovbe", Op_DstSrc},
    {"cmovc", Op_DstSrc},
    {"cmove", Op_DstSrc},
    {"cmovg", Op_DstSrc},
    {"cmovge", Op_DstSrc},
    {"cmovl", Op_DstSrc},
    {"cmovle", Op_DstSrc},
    {"cmovna", Op_DstSrc},
    {"cmovnae", Op_DstSrc},
    {"cmovnb", Op_DstSrc},
    {"cmovnbe", Op_DstSrc},
    {"cmovnc", Op_DstSrc},
    {"cmovne", Op_DstSrc},
    {"cmovng", Op_DstSrc},
    {"cmovnge", Op_DstSrc},
    {"cmovnl", Op_DstSrc},
    {"cmovnle", Op_DstSrc},
    {"cmovno", Op_DstSrc},
    {"cmovnp", Op_DstSrc},
    {"cmovns", Op_DstSrc},
    {"cmovnz", Op_DstSrc},
    {"cmovo", Op_DstSrc},
    {"cmovp", Op_DstSrc},
    {"cmovpe", Op_DstSrc},
    {"cmovpo", Op_DstSrc},
    {"cmovs", Op_DstSrc},
    {"cmovz", Op_DstSrc},
    {"cmp", Op_SrcSrcF},
    {"cmppd", Op_DstSrcImmS},
    {"cmpps", Op_DstSrcImmS},
#ifdef ASM_X86_64
    {"cmpq", Op_DstSrcNT}, // ?
#endif
    {"cmps", Op_cmps},
    {"cmpsb", Op_cmpsX},
    {"cmpsd", Op_cmpsd}, // string cmp, and SSE cmp
#ifdef ASM_X86_64
    {"cmpsq", Op_cmpsX},
#endif
    {"cmpss", Op_DstSrcImmS},
    {"cmpsw", Op_cmpsX},
    {"cmpxchg", Op_cmpxchg},
#ifdef ASM_X86_64
    {"cmpxchg16b", Op_cmpxchg16b},
#endif
    {"cmpxchg8b", Op_cmpxchg8b},
    {"comisd", Op_SrcSrcSSEF},
    {"comiss", Op_SrcSrcSSEF},
    {"cpuid", Op_cpuid},
#ifdef ASM_X86_64
    {"cqo", Op_0_DXAX},
#endif
    {"crc32", Op_DstSrc},
    {"cvtdq2pd", Op_DstSrcSSE},
    {"cvtdq2ps", Op_DstSrcSSE},
    {"cvtpd2dq", Op_DstSrcSSE},
    {"cvtpd2pi", Op_DstSrcSSE},
    {"cvtpd2ps", Op_DstSrcSSE},
    {"cvtpi2pd", Op_DstSrcSSE},
    {"cvtpi2ps", Op_DstSrcSSE},
    {"cvtps2dq", Op_DstSrcSSE},
    {"cvtps2pd", Op_DstSrcSSE},
    {"cvtps2pi", Op_DstSrcSSE},
    {"cvtsd2si", Op_DstSrcSSE},
    {"cvtsd2ss", Op_DstSrcSSE},
    {"cvtsi2sd", Op_DstSrcSSE},
    {"cvtsi2ss", Op_DstSrcSSE},
    {"cvtss2sd", Op_DstSrcSSE},
    {"cvtss2si", Op_DstSrcSSE},
    {"cvttpd2dq", Op_DstSrcSSE},
    {"cvttpd2pi", Op_DstSrcSSE},
    {"cvttps2dq", Op_DstSrcSSE},
    {"cvttps2pi", Op_DstSrcSSE},
    {"cvttsd2si", Op_DstSrcSSE},
    {"cvttss2si", Op_DstSrcSSE},
    {"cwd", Op_0_DXAX},
#ifndef ASM_X86_64
    {"cwde", Op_0_AX},
#else
    {"cwde", Op_0_DXAX},
#endif
//{ "da", Op_ }, // dunno what this is -- takes labels?
#ifndef ASM_X86_64
    {"daa", Op_Adjust},
    {"das", Op_Adjust},
#endif
    {"db", Op_db},
    {"dd", Op_dd},
    {"de", Op_de},
    {"dec", Op_UpdF},
    {"df", Op_df},
    {"di", Op_di},
    {"div", Op_Src_DXAXF},
    {"divpd", Op_DstSrcSSE},
    {"divps", Op_DstSrcSSE},
    {"divsd", Op_DstSrcSSE},
    {"divss", Op_DstSrcSSE},
    {"dl", Op_dl},
    {"dppd", Op_DstSrcImmS},
    {"dpps", Op_DstSrcImmS},
    {"dq", Op_dl},
    {"ds", Op_ds},
    {"dt", Op_de},
    {"dw", Op_ds},
    {"emms", Op_0}, // clobber all mmx/fp?
    {"enter", Op_enter},
    {"even", Op_Even},
    {"extractps", Op_ExtSrcImmS},
    {"f2xm1", Op_F0_ST}, // %% most of these are update...
    {"fabs", Op_F0_ST},
    {"fadd", Op_FMath},
    {"faddp", Op_FPMath},
    {"fbld", Op_Fs_P},
    {"fbstp", Op_Fd_P},
    {"fchs", Op_F0_ST},
    {"fclex", Op_0},
    {"fcmovb", Op_FdSTiSTi}, // but only ST(0) can be the destination -- should
                             // be FdST0STi
    {"fcmovbe", Op_FdSTiSTi},
    {"fcmove", Op_FdSTiSTi},
    {"fcmovnb", Op_FdSTiSTi},
    {"fcmovnbe", Op_FdSTiSTi},
    {"fcmovne", Op_FdSTiSTi},
    {"fcmovnu", Op_FdSTiSTi},
    {"fcmovu", Op_FdSTiSTi},
    {"fcom", Op_FCmp},
    {"fcomi", Op_FCmpFlg},
    {"fcomip", Op_FCmpFlgP},
    {"fcomp", Op_FCmpP},
    {"fcompp", Op_F0_P}, // pops twice
    {"fcos", Op_F0_ST},
    {"fdecstp", Op_F0_P}, // changes stack
    {"fdisi", Op_fdisi},
    {"fdiv", Op_FMath},
    {"fdivp", Op_FPMath},
    {"fdivr", Op_FMath},
    {"fdivrp", Op_FPMath},
    {"feni", Op_feni},
    {"ffree", Op_FdST},
    {"fiadd", Op_Fis_ST},
    {"ficom", Op_Fis},
    {"ficomp", Op_Fis_P},
    {"fidiv", Op_Fis_ST},
    {"fidivr", Op_Fis_ST},
    {"fild", Op_Fis_P},
    {"fimul", Op_Fis_ST},
    {"fincstp", Op_F0_P},
    {"finit", Op_F0_P},
    {"fist", Op_Fid}, // only 16,32bit
    {"fistp", Op_Fid_P},
    {"fisttp", Op_Fid_P},
    {"fisub", Op_Fis_ST},
    {"fisubr", Op_Fis_ST},
    {"fld", Op_fld},
    {"fld1", Op_F0_P},
    {"fldcw", Op_SrcMemNT},
    {"fldenv", Op_SrcMemNT},
    {"fldl2e", Op_F0_P},
    {"fldl2t", Op_F0_P},
    {"fldlg2", Op_F0_P},
    {"fldln2", Op_F0_P},
    {"fldpi", Op_F0_P},
    {"fldz", Op_F0_P},
    {"fmul", Op_FMath},
    {"fmulp", Op_FPMath},
    {"fnclex", Op_0},
    {"fndisi", Op_fdisi}, // ??
    {"fneni", Op_feni},   // ??
    {"fninit", Op_0},
    {"fnop", Op_0},
    {"fnsave", Op_DstMemNT},
    {"fnstcw", Op_DstMemNT},
    {"fnstenv", Op_DstMemNT},
    {"fnstsw", Op_fXstsw},
    {"fpatan", Op_F0_P}, // pop and modify new ST
    {"fprem", Op_F0_ST},
    {"fprem1", Op_F0_ST},
    {"fptan", Op_F0_P}, // modify ST and push 1.0
    {"frndint", Op_F0_ST},
    {"frstor", Op_SrcMemNT}, // but clobbers everything
    {"fsave", Op_DstMemNT},
    {"fscale", Op_F0_ST},
    {"fsetpm", Op_fsetpm},
    {"fsin", Op_F0_ST},
    {"fsincos", Op_F0_P},
    {"fsqrt", Op_F0_ST},
    {"fst", Op_Ffd},
    {"fstcw", Op_DstMemNT},
    {"fstenv", Op_DstMemNT},
    {"fstp", Op_Ffd_P},
    {"fstsw", Op_fXstsw},
    {"fsub", Op_FMath},
    {"fsubp", Op_FPMath},
    {"fsubr", Op_FMath},
    {"fsubrp", Op_FPMath},
    {"ftst", Op_0},
    {"fucom", Op_FCmp},
    {"fucomi", Op_FCmpFlg},
    {"fucomip", Op_FCmpFlgP},
    {"fucomp", Op_FCmpP},
    {"fucompp", Op_F0_P}, // pops twice
    {"fwait", Op_0},
    {"fxam", Op_0},
    {"fxch", Op_fxch},
    {"fxrstor", Op_SrcMemNT}, // clobbers FP,MMX,SSE
    {"fxsave", Op_DstMemNT},
    {"fxtract", Op_F0_P}, // pushes
    {"fyl2x", Op_F0_P},   // pops
    {"fyl2xp1", Op_F0_P}, // pops
    {"haddpd", Op_DstSrcSSE},
    {"haddps", Op_DstSrcSSE},
    {"hlt", Op_0},
    {"hsubpd", Op_DstSrcSSE},
    {"hsubps", Op_DstSrcSSE},
    {"idiv", Op_Src_DXAXF},
    {"imul", Op_imul},
    {"in", Op_in},
    {"inc", Op_UpdF},
    {"ins", Op_ins},
    {"insb", Op_insX},
    {"insd", Op_insX},
    {"insertps", Op_DstSrcImmS},
    {"insw", Op_insX},
    {"int", Op_SrcImm},
    {"into", Op_0},
    {"invd", Op_0},
    {"invlpg", Op_SrcMemNT},
    {"iret", Op_iret},
    {"iretd", Op_iretd},
#ifdef ASM_X86_64
    {"iretq", Op_iretq}, // ?
#endif
    {"ja", Op_CBranch},
    {"jae", Op_CBranch},
    {"jb", Op_CBranch},
    {"jbe", Op_CBranch},
    {"jc", Op_CBranch},
    {"jcxz", Op_CBranch},
    {"je", Op_CBranch},
    {"jecxz", Op_CBranch},
    {"jg", Op_CBranch},
    {"jge", Op_CBranch},
    {"jl", Op_CBranch},
    {"jle", Op_CBranch},
    {"jmp", Op_Branch},
#ifdef ASM_X86_64
    {"jmpe", Op_Branch}, // ?
    {"jmpf", Op_Branch}, // ?
#endif
    {"jna", Op_CBranch},
    {"jnae", Op_CBranch},
    {"jnb", Op_CBranch},
    {"jnbe", Op_CBranch},
    {"jnc", Op_CBranch},
    {"jne", Op_CBranch},
    {"jng", Op_CBranch},
    {"jnge", Op_CBranch},
    {"jnl", Op_CBranch},
    {"jnle", Op_CBranch},
    {"jno", Op_CBranch},
    {"jnp", Op_CBranch},
    {"jns", Op_CBranch},
    {"jnz", Op_CBranch},
    {"jo", Op_CBranch},
    {"jp", Op_CBranch},
    {"jpe", Op_CBranch},
    {"jpo", Op_CBranch},
#ifdef ASM_X86_64
    {"jrcxz", Op_CBranch}, // Not supported by DMD
#endif
    {"js", Op_CBranch},
    {"jz", Op_CBranch},
    {"lahf", Op_0_AX},
    {"lar", Op_DstSrcFW}, // reg dest only
    {"lddqu", Op_DstSrcSSE},
    {"ldmxcsr", Op_SrcMemNT},
    {"lds", Op_DstSrc}, // reg dest only
    {"lea", Op_DstSrc}, // "
#ifdef ASM_X86_64
    {"leaq", Op_DstSrcSSE}, // ?
#endif
    {"leave", Op_0}, // EBP,ESP clobbers
#ifndef ASM_X86_64
    {"les", Op_DstSrc},
#endif
    {"lfence", Op_0},
    {"lfs", Op_DstSrc},
    {"lgdt", Op_SrcMemNT},
    {"lgs", Op_DstSrc},
    {"lidt", Op_SrcMemNT},
    {"lldt", Op_SrcRMWNT},
    {"lmsw", Op_SrcRMWNT},
    {"lock", Op_0},
    {"lods", Op_lods},
    {"lodsb", Op_lodsX},
    {"lodsd", Op_lodsX},
#ifdef ASM_X86_64
    {"lodsq", Op_lodsX},
#endif
    {"lodsw", Op_lodsX},
    {"loop", Op_Loop},
    {"loope", Op_Loop},
    {"loopne", Op_Loop},
    {"loopnz", Op_Loop},
    {"loopz", Op_Loop},
    {"lsl", Op_DstSrcFW}, // reg dest only
    {"lss", Op_DstSrc},
    {"ltr", Op_DstMemNT},
    {"maskmovdqu", Op_SrcSrcMMX}, // writes to [edi]
    {"maskmovq", Op_SrcSrcMMX},
    {"maxpd", Op_DstSrcSSE},
    {"maxps", Op_DstSrcSSE},
    {"maxsd", Op_DstSrcSSE},
    {"maxss", Op_DstSrcSSE},
    {"mfence", Op_0},
    {"minpd", Op_DstSrcSSE},
    {"minps", Op_DstSrcSSE},
    {"minsd", Op_DstSrcSSE},
    {"minss", Op_DstSrcSSE},
    {"monitor", Op_0},
    {"mov", Op_DstSrc},
    {"movapd", Op_DstSrcSSE},
    {"movaps", Op_DstSrcSSE},
#ifdef ASM_X86_64
    {"movb", Op_DstSrcNT}, // ?
#endif
    {"movd", Op_DstSrcNT}, // also mmx and sse
    {"movddup", Op_DstSrcSSE},
    {"movdq2q", Op_DstSrcNT}, // mmx/sse
    {"movdqa", Op_DstSrcSSE},
    {"movdqu", Op_DstSrcSSE},
    {"movhlps", Op_DstSrcSSE},
    {"movhpd", Op_DstSrcSSE},
    {"movhps", Op_DstSrcSSE},
#ifdef ASM_X86_64
    {"movl", Op_DstSrc}, // ?
#endif
    {"movlhps", Op_DstSrcSSE},
    {"movlpd", Op_DstSrcSSE},
    {"movlps", Op_DstSrcSSE},
    {"movmskpd", Op_DstSrcSSE},
    {"movmskps", Op_DstSrcSSE},
    {"movntdq", Op_DstSrcNT}, // limited to sse, but mem dest
    {"movntdqa", Op_DstSrcNT},
    {"movnti", Op_DstSrcNT},  // limited to gpr, but mem dest
    {"movntpd", Op_DstSrcNT}, // limited to sse, but mem dest
    {"movntps", Op_DstSrcNT}, // limited to sse, but mem dest
    {"movntq", Op_DstSrcNT},  // limited to mmx, but mem dest
    {"movq", Op_DstSrcNT},    // limited to sse and mmx
    {"movq2dq", Op_DstSrcNT}, // limited to sse <- mmx regs
    {"movs", Op_movs},
    {"movsb", Op_movsX},
    {"movsd", Op_movsd},
    {"movshdup", Op_DstSrcSSE},
    {"movsldup", Op_DstSrcSSE},
#ifdef ASM_X86_64
    {"movsq", Op_movsd},
#endif
    {"movss", Op_DstSrcSSE},
    {"movsw", Op_movsX},
    {"movsx", Op_movsx}, // word-only, reg dest
#ifdef ASM_X86_64
    {"movsxd", Op_movsx},
#endif
    {"movupd", Op_DstSrcSSE},
    {"movups", Op_DstSrcSSE},
#ifdef ASM_X86_64
    {"movzbl", Op_DstSrcNT},
#endif
    {"movzx", Op_movzx},
    {"mpsadbw", Op_DstSrcImmS},
    {"mul", Op_mul},
    {"mulpd", Op_DstSrcSSE},
    {"mulps", Op_DstSrcSSE},
    {"mulsd", Op_DstSrcSSE},
    {"mulss", Op_DstSrcSSE},
    {"mwait", Op_0},
    {"naked", Op_Naked},
    {"neg", Op_UpdF},
    {"nop", Op_0},
    {"not", Op_Upd},
    {"or", Op_UpdSrcF},
    {"orpd", Op_DstSrcSSE},
    {"orps", Op_DstSrcSSE},
    {"out", Op_out},
    {"outs", Op_outs},
    {"outsb", Op_outsX},
    {"outsd", Op_outsX},
    {"outsw", Op_outsX},
#ifdef ASM_X86_64
    {"pabsb", Op_DstSrcSSE},
    {"pabsq", Op_DstSrcSSE},
    {"pabsw", Op_DstSrcSSE},
#endif
    {"packssdw", Op_DstSrcMMX}, // %% also SSE
    {"packsswb", Op_DstSrcMMX},
    {"packusdw", Op_DstSrcSSE},
    {"packuswb", Op_DstSrcMMX},
    {"paddb", Op_DstSrcMMX},
    {"paddd", Op_DstSrcMMX},
    {"paddq", Op_DstSrcMMX},
    {"paddsb", Op_DstSrcMMX},
    {"paddsw", Op_DstSrcMMX},
    {"paddusb", Op_DstSrcMMX},
    {"paddusw", Op_DstSrcMMX},
    {"paddw", Op_DstSrcMMX},
    {"palignr", Op_DstSrcImmS},
    {"pand", Op_DstSrcMMX},
    {"pandn", Op_DstSrcMMX},
#ifdef ASM_X86_64
    {"pause", Op_DstSrcMMX}, // ?
#endif
    {"pavgb", Op_DstSrcMMX},
    {"pavgusb", Op_DstSrcMMX}, // AMD 3dNow!
    {"pavgw", Op_DstSrcMMX},
    {"pblendvb", Op_DstSrcSSE}, // implicit xmm0
    {"pblendw", Op_DstSrcImmS},
    {"pcmpeqb", Op_DstSrcMMX},
    {"pcmpeqd", Op_DstSrcMMX},
    {"pcmpeqq", Op_DstSrcSSE},
    {"pcmpeqw", Op_DstSrcMMX},
    {"pcmpestri", Op_DstSrcImmS},
    {"pcmpestrm", Op_DstSrcImmS},
    {"pcmpgtb", Op_DstSrcMMX},
    {"pcmpgtd", Op_DstSrcMMX},
    {"pcmpgtq", Op_DstSrcSSE},
    {"pcmpgtw", Op_DstSrcMMX},
    {"pcmpistri", Op_DstSrcImmS},
    {"pcmpistrm", Op_DstSrcImmS},
    {"pextrb", Op_DstSrcImmM}, // gpr32 dest
    {"pextrd", Op_DstSrcImmM}, // gpr32 dest
    {"pextrq", Op_DstSrcImmM}, // gpr32 dest
    {"pextrw", Op_DstSrcImmM}, // gpr32 dest
    {"pf2id", Op_DstSrcMMX},   // %% AMD 3dNow! opcodes
    {"pfacc", Op_DstSrcMMX},
    {"pfadd", Op_DstSrcMMX},
    {"pfcmpeq", Op_DstSrcMMX},
    {"pfcmpge", Op_DstSrcMMX},
    {"pfcmpgt", Op_DstSrcMMX},
    {"pfmax", Op_DstSrcMMX},
    {"pfmin", Op_DstSrcMMX},
    {"pfmul", Op_DstSrcMMX},
    {"pfnacc", Op_DstSrcMMX},  // 3dNow values are returned in MM0 register,
    {"pfpnacc", Op_DstSrcMMX}, // so should be correct to use Op_DstSrcMMX.
    {"pfrcp", Op_DstSrcMMX},
    {"pfrcpit1", Op_DstSrcMMX},
    {"pfrcpit2", Op_DstSrcMMX},
    {"pfrsqit1", Op_DstSrcMMX},
    {"pfrsqrt", Op_DstSrcMMX},
    {"pfsub", Op_DstSrcMMX},
    {"pfsubr", Op_DstSrcMMX},
#ifdef ASM_X86_64
    {"phaddd", Op_DstSrcSSE},
    {"phaddsw", Op_DstSrcSSE},
    {"phaddw", Op_DstSrcSSE},
    {"phminposuw", Op_DstSrcSSE},
    {"phsubd", Op_DstSrcSSE},
    {"phsubsw", Op_DstSrcSSE},
    {"phsubw", Op_DstSrcSSE},
#endif
    {"pi2fd", Op_DstSrcMMX},   // %%
    {"pinsrb", Op_DstSrcImmM}, // gpr32(16), mem16 src, sse too
    {"pinsrd", Op_DstSrcImmM}, // gpr32(16), mem16 src, sse too
    {"pinsrq", Op_DstSrcImmM}, // gpr32(16), mem16 src, sse too
    {"pinsrw", Op_DstSrcImmM}, // gpr32(16), mem16 src, sse too
#ifdef ASM_X86_64
    {"pmaddubsw", Op_DstSrcSSE},
#endif
    {"pmaddwd", Op_DstSrcMMX},
    {"pmaxsb", Op_DstSrcSSE},
    {"pmaxsd", Op_DstSrcSSE},
    {"pmaxsw", Op_DstSrcMMX},
    {"pmaxub", Op_DstSrcMMX},
    {"pmaxud", Op_DstSrcSSE},
    {"pmaxuw", Op_DstSrcSSE},
    {"pminsb", Op_DstSrcSSE},
    {"pminsd", Op_DstSrcSSE},
    {"pminsw", Op_DstSrcMMX},
    {"pminub", Op_DstSrcMMX},
    {"pminud", Op_DstSrcSSE},
    {"pminuw", Op_DstSrcSSE},
    {"pmovmskb", Op_DstSrcMMX},
    {"pmovsxbd", Op_DstSrcSSE},
    {"pmovsxbq", Op_DstSrcSSE},
    {"pmovsxbw", Op_DstSrcSSE},
    {"pmovsxdq", Op_DstSrcSSE},
    {"pmovsxwd", Op_DstSrcSSE},
    {"pmovsxwq", Op_DstSrcSSE},
    {"pmovzxbd", Op_DstSrcSSE},
    {"pmovzxbq", Op_DstSrcSSE},
    {"pmovzxbw", Op_DstSrcSSE},
    {"pmovzxdq", Op_DstSrcSSE},
    {"pmovzxwd", Op_DstSrcSSE},
    {"pmovzxwq", Op_DstSrcSSE},
    {"pmuldq", Op_DstSrcSSE},
#ifdef ASM_X86_64
    {"pmulhrsw", Op_DstSrcMMX},
#endif
    {"pmulhrw", Op_DstSrcMMX}, // AMD 3dNow!
    {"pmulhuw", Op_DstSrcMMX},
    {"pmulhw", Op_DstSrcMMX},
    {"pmulld", Op_DstSrcSSE},
    {"pmullw", Op_DstSrcMMX},
    {"pmuludq", Op_DstSrcMMX}, // also sse
    {"pop", Op_DstW},
#ifndef ASM_X86_64
    {"popa", Op_SizedStack},  // For intel this is always 16-bit
    {"popad", Op_SizedStack}, // GAS doesn't accept 'popad' -- these clobber
                              // everything, but supposedly it would be used to
                              // preserve clobbered regs
#endif
    {"popcnt", Op_DstSrc},
    {"popf", Op_0}, // rewrite the insn with a special case
#ifndef ASM_X86_64
    {"popfd", Op_0},
#else
    {"popfq", Op_0}, // Op_SizedStack?
    {"popq", Op_push},
#endif
    {"por", Op_DstSrcMMX},
    {"prefetchnta", Op_SrcMemNT},
    {"prefetcht0", Op_SrcMemNT},
    {"prefetcht1", Op_SrcMemNT},
    {"prefetcht2", Op_SrcMemNT},
    {"psadbw", Op_DstSrcMMX},
    {"pshufb", Op_DstSrcSSE},
#ifndef ASM_X86_64
    {"pshufd", Op_DstSrcImmM},
    {"pshufhw", Op_DstSrcImmM},
    {"pshuflw", Op_DstSrcImmM},
    {"pshufw", Op_DstSrcImmM},
#else
    {"pshufd", Op_DstSrcImmS},
    {"pshufhw", Op_DstSrcImmS},
    {"pshuflw", Op_DstSrcImmS},
    {"pshufw", Op_DstSrcImmS},
    {"psignb", Op_DstSrcSSE},
    {"psignd", Op_DstSrcSSE},
    {"psignw", Op_DstSrcSSE},
#endif
    {"pslld", Op_DstSrcMMX}, // immediate operands...
    {"pslldq", Op_DstSrcMMX},
    {"psllq", Op_DstSrcMMX},
    {"psllw", Op_DstSrcMMX},
    {"psrad", Op_DstSrcMMX},
    {"psraw", Op_DstSrcMMX},
    {"psrld", Op_DstSrcMMX},
    {"psrldq", Op_DstSrcMMX},
    {"psrlq", Op_DstSrcMMX},
    {"psrlw", Op_DstSrcMMX},
    {"psubb", Op_DstSrcMMX},
    {"psubd", Op_DstSrcMMX},
    {"psubq", Op_DstSrcMMX},
    {"psubsb", Op_DstSrcMMX},
    {"psubsw", Op_DstSrcMMX},
    {"psubusb", Op_DstSrcMMX},
    {"psubusw", Op_DstSrcMMX},
    {"psubw", Op_DstSrcMMX},
    {"pswapd", Op_DstSrcMMX}, // AMD 3dNow!
    {"ptest", Op_SrcSrcSSEF},
    {"punpckhbw", Op_DstSrcMMX},
    {"punpckhdq", Op_DstSrcMMX},
    {"punpckhqdq", Op_DstSrcMMX},
    {"punpckhwd", Op_DstSrcMMX},
    {"punpcklbw", Op_DstSrcMMX},
    {"punpckldq", Op_DstSrcMMX},
    {"punpcklqdq", Op_DstSrcMMX},
    {"punpcklwd", Op_DstSrcMMX},
    {"push", Op_push},
#ifndef ASM_X86_64
    {"pusha", Op_SizedStack},
    {"pushad", Op_SizedStack},
#endif
    {"pushf", Op_0},
#ifndef ASM_X86_64
    {"pushfd", Op_0},
#else
    {"pushfq", Op_0},   // Op_SizedStack?
    {"pushq", Op_push}, // ?
#endif
    {"pxor", Op_DstSrcMMX},
    {"rcl", Op_Shift}, // limited src operands -- change to shift
    {"rcpps", Op_DstSrcSSE},
    {"rcpss", Op_DstSrcSSE},
    {"rcr", Op_Shift},
    //        { rdfsbase, XXXX },
    //        { rdgsbase, XXXX },
    {"rdmsr", Op_0_DXAX},
    {"rdpmc", Op_0_DXAX},
    //        { rdrand, XXXX },
    {"rdtsc", Op_0_DXAX},
    {"rep", Op_0},
    {"repe", Op_0},
    {"repne", Op_0},
    {"repnz", Op_0},
    {"repz", Op_0},
    {"ret", Op_ret},
    {"retf", Op_retf},
#ifdef ASM_X86_64
    {"retn", Op_retf}, // ?
#endif
    {"rol", Op_Shift},
    {"ror", Op_Shift},
    {"roundpd", Op_DstSrcImmS},
    {"roundps", Op_DstSrcImmS},
    {"roundsd", Op_DstSrcImmS},
    {"roundss", Op_DstSrcImmS},
    {"rsm", Op_0},
    {"rsqrtps", Op_DstSrcSSE},
    {"rsqrtss", Op_DstSrcSSE},
    {"sahf", Op_Flags},
    {"sal", Op_Shift},
#ifdef ASM_X86_64
    {"salq", Op_DstSrcNT},
#endif
    {"sar", Op_Shift},
    {"sbb", Op_UpdSrcF},
    {"scas", Op_scas},
    {"scasb", Op_scasX},
    {"scasd", Op_scasX},
#ifdef ASM_X86_64
    {"scasq", Op_scasX},
#endif
    {"scasw", Op_scasX},
    {"seta", Op_DstRMBNT}, // also gpr8
    {"setae", Op_DstRMBNT},
    {"setb", Op_DstRMBNT},
    {"setbe", Op_DstRMBNT},
    {"setc", Op_DstRMBNT},
    {"sete", Op_DstRMBNT},
    {"setg", Op_DstRMBNT},
    {"setge", Op_DstRMBNT},
    {"setl", Op_DstRMBNT},
    {"setle", Op_DstRMBNT},
    {"setna", Op_DstRMBNT},
    {"setnae", Op_DstRMBNT},
    {"setnb", Op_DstRMBNT},
    {"setnbe", Op_DstRMBNT},
    {"setnc", Op_DstRMBNT},
    {"setne", Op_DstRMBNT},
    {"setng", Op_DstRMBNT},
    {"setnge", Op_DstRMBNT},
    {"setnl", Op_DstRMBNT},
    {"setnle", Op_DstRMBNT},
    {"setno", Op_DstRMBNT},
    {"setnp", Op_DstRMBNT},
    {"setns", Op_DstRMBNT},
    {"setnz", Op_DstRMBNT},
    {"seto", Op_DstRMBNT},
    {"setp", Op_DstRMBNT},
    {"setpe", Op_DstRMBNT},
    {"setpo", Op_DstRMBNT},
    {"sets", Op_DstRMBNT},
    {"setz", Op_DstRMBNT},
    {"sfence", Op_0},
    {"sgdt", Op_DstMemNT},
    {"sha1msg1", Op_DstSrcSSE},
    {"sha1msg2", Op_DstSrcSSE},
    {"sha1nexte", Op_DstSrcSSE},
    {"sha1rnds4", Op_DstSrcImmS},
    {"sha256msg1", Op_DstSrcSSE},
    {"sha256msg2", Op_DstSrcSSE},
    {"sha256rnds2", Op_DstSrcSSE}, // implicit xmm0
    {"shl", Op_Shift},
    {"shld", Op_UpdSrcShft},
    {"shr", Op_Shift},
    {"shrd", Op_UpdSrcShft},
    {"shufpd", Op_DstSrcImmS},
    {"shufps", Op_DstSrcImmS},
    {"sidt", Op_DstMemNT},
    {"sldt", Op_DstRMWNT},
    {"smsw", Op_DstRMWNT},
    {"sqrtpd", Op_DstSrcSSE},
    {"sqrtps", Op_DstSrcSSE},
    {"sqrtsd", Op_DstSrcSSE},
    {"sqrtss", Op_DstSrcSSE},
    {"stc", Op_Flags},
    {"std", Op_Flags},
    {"sti", Op_Flags},
    {"stmxcsr", Op_DstMemNT},
    {"stos", Op_stos},
    {"stosb", Op_stosX},
    {"stosd", Op_stosX},
#ifdef ASM_X86_64
    {"stosq", Op_stosX},
#endif
    {"stosw", Op_stosX},
    {"str", Op_DstMemNT}, // also r16
    {"sub", Op_UpdSrcF},
    {"subpd", Op_DstSrcSSE},
    {"subps", Op_DstSrcSSE},
#ifdef ASM_X86_64
    {"subq", Op_DstSrcSSE}, // ?
#endif
    {"subsd", Op_DstSrcSSE},
    {"subss", Op_DstSrcSSE},
#ifdef ASM_X86_64
    {"swapgs", Op_0}, // Not supported by DMD
#endif
    {"syscall", Op_0},
    {"sysenter", Op_0},
    {"sysexit", Op_0},
    {"sysret", Op_0},
#ifdef ASM_X86_64
    {"sysretq", Op_0}, // ?
#endif
    {"test", Op_SrcSrcF},
    {"ucomisd", Op_SrcSrcSSEF},
    {"ucomiss", Op_SrcSrcSSEF},
    {"ud2", Op_0},
    {"unpckhpd", Op_DstSrcSSE},
    {"unpckhps", Op_DstSrcSSE},
    {"unpcklpd", Op_DstSrcSSE},
    {"unpcklps", Op_DstSrcSSE},
    //        { vaddpd, XXXX },
    //        { vaddps, XXXX },
    //        { vaddsd, XXXX },
    //        { vaddss, XXXX },
    //        { vaddsubpd, XXXX },
    //        { vaddsubps, XXXX },
    //        { vaesdec, XXXX },
    //        { vaesdeclast, XXXX },
    //        { vaesenc, XXXX },
    //        { vaesenclast, XXXX },
    //        { vaesimc, XXXX },
    //        { vaeskeygenassist, XXXX },
    //        { vandnpd, XXXX },
    //        { vandnps, XXXX },
    //        { vandpd, XXXX },
    //        { vandps, XXXX },
    //        { vblendpd, XXXX },
    //        { vblendps, XXXX },
    //        { vblendvpd, XXXX },
    //        { vblendvps, XXXX },
    //        { vbroadcastf128, XXXX },
    //        { vbroadcastsd, XXXX },
    //        { vbroadcastss, XXXX },
    //        { vcmppd, XXXX },
    //        { vcmpps, XXXX },
    //        { vcmpsd, XXXX },
    //        { vcmpss, XXXX },
    //        { vcomisd, XXXX },
    //        { vcomiss, XXXX },
    //        { vcvtdq2pd, XXXX },
    //        { vcvtdq2ps, XXXX },
    //        { vcvtpd2dq, XXXX },
    //        { vcvtpd2ps, XXXX },
    //        { vcvtph2ps, XXXX },
    //        { vcvtps2dq, XXXX },
    //        { vcvtps2pd, XXXX },
    //        { vcvtps2ph, XXXX },
    //        { vcvtsd2si, XXXX },
    //        { vcvtsd2ss, XXXX },
    //        { vcvtsi2sd, XXXX },
    //        { vcvtsi2ss, XXXX },
    //        { vcvtss2sd, XXXX },
    //        { vcvtss2si, XXXX },
    //        { vcvttpd2dq, XXXX },
    //        { vcvttps2dq, XXXX },
    //        { vcvttsd2si, XXXX },
    //        { vcvttss2si, XXXX },
    //        { vdivpd, XXXX },
    //        { vdivps, XXXX },
    //        { vdivsd, XXXX },
    //        { vdivss, XXXX },
    //        { vdppd, XXXX },
    //        { vdpps, XXXX },
    {"verr", Op_SrcMemNTF},
    {"verw", Op_SrcMemNTF},
//        { vextractf128, XXXX },
//        { vextractps, XXXX },
//        { vfmadd132pd, XXXX },
//        { vfmadd132ps, XXXX },
//        { vfmadd132sd, XXXX },
//        { vfmadd132ss, XXXX },
//        { vfmadd213pd, XXXX },
//        { vfmadd213ps, XXXX },
//        { vfmadd213sd, XXXX },
//        { vfmadd213ss, XXXX },
//        { vfmadd231pd, XXXX },
//        { vfmadd231ps, XXXX },
//        { vfmadd231sd, XXXX },
//        { vfmadd231ss, XXXX },
//        { vfmaddsub132pd, XXXX },
//        { vfmaddsub132ps, XXXX },
//        { vfmaddsub213pd, XXXX },
//        { vfmaddsub213ps, XXXX },
//        { vfmaddsub231pd, XXXX },
//        { vfmaddsub231ps, XXXX },
//        { vfmsub132pd, XXXX },
//        { vfmsub132ps, XXXX },
//        { vfmsub132sd, XXXX },
//        { vfmsub132ss, XXXX },
//        { vfmsub213pd, XXXX },
//        { vfmsub213ps, XXXX },
//        { vfmsub213sd, XXXX },
//        { vfmsub213ss, XXXX },
//        { vfmsub231pd, XXXX },
//        { vfmsub231ps, XXXX },
//        { vfmsub231sd, XXXX },
//        { vfmsub231ss, XXXX },
//        { vfmsubadd132pd, XXXX },
//        { vfmsubadd132ps, XXXX },
//        { vfmsubadd213pd, XXXX },
//        { vfmsubadd213ps, XXXX },
//        { vfmsubadd231pd, XXXX },
//        { vfmsubadd231ps, XXXX },
//        { vhaddpd, XXXX },
//        { vhaddps, XXXX },
//        { vinsertf128, XXXX },
//        { vinsertps, XXXX },
//        { vlddqu, XXXX },
//        { vldmxcsr, XXXX },
//        { vmaskmovdqu, XXXX },
//        { vmaskmovpd, XXXX },
//        { vmaskmovps, XXXX },
//        { vmaxpd, XXXX },
//        { vmaxps, XXXX },
//        { vmaxsd, XXXX },
//        { vmaxss, XXXX },
//        { vminpd, XXXX },
//        { vminps, XXXX },
//        { vminsd, XXXX },
//        { vminss, XXXX },
//        { vmovapd, XXXX },
//        { vmovaps, XXXX },
//        { vmovd, XXXX },
//        { vmovddup, XXXX },
//        { vmovdqa, XXXX },
//        { vmovdqu, XXXX },
//        { vmovhlps, XXXX },
//        { vmovhpd, XXXX },
//        { vmovhps, XXXX },
//        { vmovlhps, XXXX },
//        { vmovlpd, XXXX },
//        { vmovlps, XXXX },
//        { vmovmskpd, XXXX },
//        { vmovmskps, XXXX },
//        { vmovntdq, XXXX },
//        { vmovntdqa, XXXX },
//        { vmovntpd, XXXX },
//        { vmovntps, XXXX },
//        { vmovq, XXXX },
//        { vmovsd, XXXX },
//        { vmovshdup, XXXX },
//        { vmovsldup, XXXX },
//        { vmovss, XXXX },
//        { vmovupd, XXXX },
//        { vmovups, XXXX },
//        { vmpsadbw, XXXX },
//        { vmulpd, XXXX },
//        { vmulps, XXXX },
//        { vmulsd, XXXX },
//        { vmulss, XXXX },
//        { vorpd, XXXX },
//        { vorps, XXXX },
//        { vpabsb, XXXX },
//        { vpabsd, XXXX },
//        { vpabsw, XXXX },
//        { vpackssdw, XXXX },
//        { vpacksswb, XXXX },
//        { vpackusdw, XXXX },
//        { vpackuswb, XXXX },
//        { vpaddb, XXXX },
//        { vpaddd, XXXX },
//        { vpaddq, XXXX },
//        { vpaddsb, XXXX },
//        { vpaddsw, XXXX },
//        { vpaddusb, XXXX },
//        { vpaddusw, XXXX },
//        { vpaddw, XXXX },
//        { vpalignr, XXXX },
//        { vpand, XXXX },
//        { vpandn, XXXX },
//        { vpavgb, XXXX },
//        { vpavgw, XXXX },
//        { vpblendvb, XXXX },
//        { vpblendw, XXXX },
//        { vpclmulqdq, XXXX },
//        { vpcmpeqb, XXXX },
//        { vpcmpeqd, XXXX },
//        { vpcmpeqq, XXXX },
//        { vpcmpeqw, XXXX },
//        { vpcmpestri, XXXX },
//        { vpcmpestrm, XXXX },
//        { vpcmpgtb, XXXX },
//        { vpcmpgtd, XXXX },
//        { vpcmpgtq, XXXX },
//        { vpcmpgtw, XXXX },
//        { vpcmpistri, XXXX },
//        { vpcmpistrm, XXXX },
//        { vperm2f128, XXXX },
//        { vpermilpd, XXXX },
//        { vpermilps, XXXX },
//        { vpextrb, XXXX },
//        { vpextrd, XXXX },
//        { vpextrq, XXXX },
//        { vpextrw, XXXX },
//        { vphaddd, XXXX },
//        { vphaddsw, XXXX },
//        { vphaddw, XXXX },
//        { vphminposuw, XXXX },
//        { vphsubd, XXXX },
//        { vphsubsw, XXXX },
//        { vphsubw, XXXX },
//        { vpinsrb, XXXX },
//        { vpinsrd, XXXX },
//        { vpinsrq, XXXX },
//        { vpinsrw, XXXX },
//        { vpmaddubsw, XXXX },
//        { vpmaddwd, XXXX },
//        { vpmaxsb, XXXX },
//        { vpmaxsd, XXXX },
//        { vpmaxsw, XXXX },
//        { vpmaxub, XXXX },
//        { vpmaxud, XXXX },
//        { vpmaxuw, XXXX },
//        { vpminsb, XXXX },
//        { vpminsd, XXXX },
//        { vpminsw, XXXX },
//        { vpminub, XXXX },
//        { vpminud, XXXX },
//        { vpminuw, XXXX },
//        { vpmovmskb, XXXX },
//        { vpmovsxbd, XXXX },
//        { vpmovsxbq, XXXX },
//        { vpmovsxbw, XXXX },
//        { vpmovsxdq, XXXX },
//        { vpmovsxwd, XXXX },
//        { vpmovsxwq, XXXX },
//        { vpmovzxbd, XXXX },
//        { vpmovzxbq, XXXX },
//        { vpmovzxbw, XXXX },
//        { vpmovzxdq, XXXX },
//        { vpmovzxwd, XXXX },
//        { vpmovzxwq, XXXX },
//        { vpmuldq, XXXX },
//        { vpmulhrsw, XXXX },
//        { vpmulhuw, XXXX },
//        { vpmulhw, XXXX },
//        { vpmulld, XXXX },
//        { vpmullw, XXXX },
//        { vpmuludq, XXXX },
//        { vpor, XXXX },
//        { vpsadbw, XXXX },
//        { vpshufb, XXXX },
//        { vpshufd, XXXX },
//        { vpshufhw, XXXX },
//        { vpshuflw, XXXX },
//        { vpsignb, XXXX },
//        { vpsignd, XXXX },
//        { vpsignw, XXXX },
//        { vpslld, XXXX },
//        { vpslldq, XXXX },
//        { vpsllq, XXXX },
//        { vpsllw, XXXX },
//        { vpsrad, XXXX },
//        { vpsraw, XXXX },
//        { vpsrld, XXXX },
//        { vpsrldq, XXXX },
//        { vpsrlq, XXXX },
//        { vpsrlw, XXXX },
//        { vpsubb, XXXX },
//        { vpsubd, XXXX },
//        { vpsubq, XXXX },
//        { vpsubsb, XXXX },
//        { vpsubsw, XXXX },
//        { vpsubusb, XXXX },
//        { vpsubusw, XXXX },
//        { vpsubw, XXXX },
//        { vptest, XXXX },
//        { vpunpckhbw, XXXX },
//        { vpunpckhdq, XXXX },
//        { vpunpckhqdq, XXXX },
//        { vpunpckhwd, XXXX },
//        { vpunpcklbw, XXXX },
//        { vpunpckldq, XXXX },
//        { vpunpcklqdq, XXXX },
//        { vpunpcklwd, XXXX },
//        { vpxor, XXXX },
//        { vrcpps, XXXX },
//        { vrcpss, XXXX },
//        { vroundpd, XXXX },
//        { vroundps, XXXX },
//        { vroundsd, XXXX },
//        { vroundss, XXXX },
//        { vshufpd, XXXX },
//        { vshufps, XXXX },
//        { vsqrtpd, XXXX },
//        { vsqrtps, XXXX },
//        { vsqrtsd, XXXX },
//        { vsqrtss, XXXX },
//        { vstmxcsr, XXXX },
//        { vsubpd, XXXX },
//        { vsubps, XXXX },
//        { vsubsd, XXXX },
//        { vsubss, XXXX },
//        { vucomisd, XXXX },
//        { vucomiss, XXXX },
//        { vunpckhpd, XXXX },
//        { vunpckhps, XXXX },
//        { vunpcklpd, XXXX },
//        { vunpcklps, XXXX },
//        { vxorpd, XXXX },
//        { vxorps, XXXX },
//        { vzeroall, XXXX },
//        { vzeroupper, XXXX },
#ifndef ASM_X86_64
    {"wait", Op_0},
#endif
    {"wbinvd", Op_0},
    //        { wrfsbase, XXXX },
    //        { wrgsbase, XXXX },
    {"wrmsr", Op_0},
    {"xadd", Op_UpdUpdF},
    {"xchg", Op_UpdUpd},
    {"xgetbv", Op_xgetbv},
    {"xlat", Op_xlat},
    {"xlatb", Op_0_AX},
    {"xor", Op_DstSrcF},
    {"xorpd", Op_DstSrcSSE},
    {"xorps", Op_DstSrcSSE},
#ifdef ASM_X86_64
    {"xorq", Op_DstSrcNT}, //?
#endif
    //        { xrstor, XXXX },
    //        { xrstor64, XXXX },
    //        { xsave, XXXX },
    //        { xsave64, XXXX },
    //        { xsaveopt, XXXX },
    //        { xsaveopt64, XXXX },
    //        { xsetbv, XXXX },
};

typedef enum {
  Default_Ptr = 0,
  Byte_Ptr = 1,
  Short_Ptr = 2,
  Int_Ptr = 4,
  QWord_Ptr = 8,
  Float_Ptr = 4,
  Double_Ptr = 8,
  Extended_Ptr = 10,
  Near_Ptr = 98,
  Far_Ptr = 99,
  N_PtrTypes
} PtrType;

static const int N_PtrNames = 8;
static const char *ptrTypeNameTable[N_PtrNames] = {
    "word", "dword", "qword", "float", "double", "extended", "near", "far"};

static Identifier *ptrTypeIdentTable[N_PtrNames];
static PtrType ptrTypeValueTable[N_PtrNames] = {
    Short_Ptr,  Int_Ptr,      QWord_Ptr, Float_Ptr,
    Double_Ptr, Extended_Ptr, Near_Ptr,  Far_Ptr};

typedef enum { Opr_Invalid, Opr_Immediate, Opr_Reg, Opr_Mem } OperandClass;

/* kill inlining if we reference a local? */

/* DMD seems to allow only one 'symbol' per operand .. include __LOCAL_SIZE */

/* DMD offset usage: <parm>[<reg>] seems to always be relative to EBP+8 .. even
   if naked.. */

// mov eax, 4
// mov eax, fs:4
// -- have to assume we know whether or not to use '$'

static Token eof_tok;
static Expression *Handled;
static Identifier *ident_seg;

struct AsmProcessor {
  typedef struct {
    int inBracket;
    int hasBracket;
    int hasNumber;
    int isOffset;

    Reg segmentPrefix;
    Reg reg;
    sinteger_t constDisplacement; // use to build up.. should be int constant in
                                  // the end..
    Expressions symbolDisplacement; // array of expressions or..
    Reg baseReg;
    Reg indexReg;
    int scale;

    OperandClass cls;
    PtrType dataSize;
    PtrType dataSizeHint; // DMD can use the type of a referenced variable
  } Operand;

  static const unsigned Max_Operands = 3;

  AsmStatement *stmt;
  Scope *sc;

  Token *token;
  std::ostringstream insnTemplate;

  AsmOp op;
  AsmOpInfo *opInfo;
  Operand operands[Max_Operands];
  Identifier *opIdent;
  Operand *operand;

  AsmProcessor(Scope *sc, AsmStatement *stmt) {
    this->sc = sc;
    this->stmt = stmt;
    token = stmt->tokens;

    opInfo = nullptr;

    if (!regInfo[0].ident) {
      char buf[8], *p;

      for (int i = 0; i < N_Regs; i++) {
        strncpy(buf, regInfo[i].name, sizeof(buf) - 1);
        for (p = buf; *p; p++) {
          *p = std::tolower(*p);
        }
        regInfo[i].gccName = std::string(buf, p - buf);
        if ((i <= Reg_ST || i > Reg_ST7) && i != Reg_EFLAGS) {
          regInfo[i].ident = Identifier::idPool(regInfo[i].name);
        }
      }

      for (int i = 0; i < N_PtrNames; i++) {
        ptrTypeIdentTable[i] = Identifier::idPool(ptrTypeNameTable[i]);
      }

      Handled = new Expression(Loc(), TOKvoid, sizeof(Expression));

      ident_seg = Identifier::idPool("seg");

      eof_tok.value = TOKeof;
      eof_tok.next = nullptr;
    }
  }

  void run() { parse(); }

  void nextToken() {
    if (token->next) {
      token = token->next;
    } else {
      token = &eof_tok;
    }
  }

  Token *peekToken() {
    if (token->next) {
      return token->next;
    }
    return &eof_tok;
  }

  void expectEnd() {
    if (token->value != TOKeof) {
      stmt->error("expected end of statement"); // %% extra at end...
    }
  }

  void parse() {
    op = parseOpcode();

    switch (op) {
    case Op_Align:
      doAlign();
      expectEnd();
      break;
    case Op_Even:
      doEven();
      expectEnd();
      break;
    case Op_Naked:
      doNaked();
      expectEnd();
      break;
    case Op_Invalid:
      break;
    default:
      if (op >= Op_db && op <= Op_de) {
        doData();
      } else {
        doInstruction();
      }
    }
  }

  AsmOp parseOpcode() {
    static const int N_ents = sizeof(opData) / sizeof(AsmOpEnt);

    switch (token->value) {
    case TOKalign:
      nextToken();
      return Op_Align;
    case TOKin:
      nextToken();
      opIdent = Id::___in;
      return Op_in;
    case TOKint32: // "int"
      nextToken();
      opIdent = Id::__int;
      return Op_SrcImm;
    case TOKout:
      nextToken();
      opIdent = Id::___out;
      return Op_out;
    case TOKidentifier:
      // search for mnemonic below
      break;
    default:
      stmt->error("expected opcode");
      return Op_Invalid;
    }

    opIdent = token->ident;
    const char *opcode = token->ident->string;

    nextToken();

    // %% okay to use bsearch?
    int i = 0, j = N_ents, k, l;
    do {
      k = (i + j) / 2;
      l = strcmp(opcode, opData[k].inMnemonic);
      if (!l) {
        return opData[k].asmOp;
      }
      if (l < 0) {
        j = k;
      } else {
        i = k + 1;
      }
    } while (i != j);

    stmt->error("unknown opcode '%s'", opcode);

    return Op_Invalid;
  }

  // need clobber information.. use information is good too...
  void doInstruction() {
    unsigned operand_i = 0;

    opInfo = &asmOpInfo[op];
    memset(operands, 0, sizeof(operands));

    if (token->value == TOKeof && op == Op_FMath0) {
// no iteration for x86 vs. single iteration for x64
#ifndef ASM_X86_64
      while (false)
#else
      for (operand_i = 0; operand_i < 1; operand_i++)
#endif
      {
        operand = &operands[operand_i];
        operand->reg = operand->baseReg = operand->indexReg =
            operand->segmentPrefix = Reg_Invalid;
        operand->cls = Opr_Reg;
        if (operand_i == 0) {
          operand->reg = Reg_ST;
        } else {
          operand->reg = Reg_ST1;
        }
        operand->hasNumber = 0;
        operand->constDisplacement = 0;
        parseOperand();

        if (matchOperands(operand_i)) {
          auto asmcode = new AsmCode(N_Regs);

          if (formatInstruction(operand_i, asmcode)) {
            stmt->asmcode = (code *)asmcode;
          }
        }
      }
      return;
    }

    while (token->value != TOKeof) {
      if (operand_i < Max_Operands) {
        operand = &operands[operand_i];
        operand->reg = operand->baseReg = operand->indexReg =
            operand->segmentPrefix = Reg_Invalid;
        parseOperand();
        operand_i++;
      } else {
        stmt->error("too many operands for instruction");
        break;
      }

      if (token->value == TOKcomma) {
        nextToken();
      } else if (token->value != TOKeof) {
        stmt->error("end of instruction expected, not '%s'", token->toChars());
        return;
      }
    }
    //  if (operand_i < opInfo->minOperands) {
    //      stmt->error("too few operands for instruction");
    //  }

    if (matchOperands(operand_i)) {
      auto asmcode = new AsmCode(N_Regs);

      if (formatInstruction(operand_i, asmcode)) {
        stmt->asmcode = (code *)asmcode;
      }
    }
  }

  void setAsmCode() {
    auto asmcode = new AsmCode(N_Regs);
    asmcode->insnTemplate = insnTemplate.str();
    Logger::cout() << "insnTemplate = " << asmcode->insnTemplate << '\n';
    stmt->asmcode = (code *)asmcode;
  }

  // note: doesn't update AsmOp op
  bool matchOperands(unsigned nOperands) {
    bool wrong_number = true;

    for (unsigned i = 0; i < nOperands; i++) {
      classifyOperand(&operands[i]);
    }

    while (1) {
      if (nOperands == opInfo->nOperands()) {
        wrong_number = false;
        /*  Cases in which number of operands is not
            enough for a match: Op_FCmp/Op_FCmp1,
            Op_FCmpP/Op_FCmpP1 */
        for (unsigned i = 0; i < nOperands; i++) {
          Operand *operand = &operands[i];

          switch (opInfo->operands[i] & Opr_ClassMask) {
          case OprC_Mem: // no FPMem currently
            if (operand->cls != Opr_Mem) {
              goto no_match;
            }
            break;
          case OprC_RFP:
            if (!(operand->reg >= Reg_ST && operand->reg <= Reg_ST7)) {
              goto no_match;
            }
            break;
          default:
            break;
          }
        }

        return true;
      }
    no_match:
      if (opInfo->linkType == Next_Form) {
        opInfo = &asmOpInfo[op = static_cast<AsmOp>(opInfo->link)];
      } else {
        break;
      }
    }
    if (wrong_number) {
      stmt->error("wrong number of operands");
    } else {
      stmt->error("wrong operand types");
    }
    return false;
  }

  // OSX and 32-bit Windows need an extra leading underscore when mangling a
  // symbol name.
  static bool prependExtraUnderscore() {
    return global.params.targetTriple.getOS() == llvm::Triple::MacOSX ||
           global.params.targetTriple.getOS() == llvm::Triple::Darwin ||
           (global.params.targetTriple.isOSWindows() &&
            global.params.targetTriple.isArch32Bit());
  }

  void addOperand(const char *fmt, AsmArgType type, Expression *e,
                  AsmCode *asmcode, AsmArgMode mode = Mode_Input) {
    if (sc->func->naked) {
      switch (type) {
      case Arg_Integer:
        if (e->type->isunsigned()) {
          insnTemplate << "$" << e->toUInteger();
        } else {
#ifndef ASM_X86_64
          insnTemplate << "$" << static_cast<sinteger_t>(e->toInteger());
#else
          insnTemplate << "$" << e->toInteger();
#endif
        }
        break;

      case Arg_Pointer:
        stmt->error("unsupported pointer reference to '%s' in naked asm",
                    e->toChars());
        break;

      case Arg_Memory:
        // Peel off one layer of explicitly taking the address, if present.
        if (e->op == TOKaddress) {
          e = static_cast<AddrExp *>(e)->e1;
        }

        if (e->op == TOKvar) {
          VarExp *v = (VarExp *)e;
          if (VarDeclaration *vd = v->var->isVarDeclaration()) {
            if (!vd->isDataseg()) {
              stmt->error("only global variables can be referenced by "
                          "identifier in naked asm");
              break;
            }

            // print out the mangle
            if (prependExtraUnderscore()) {
              insnTemplate << "_";
            }
            insnTemplate << mangle(vd);
            getIrGlobal(vd, true)->nakedUse = true;
            break;
          }
        }
        stmt->error("unsupported memory reference to '%s' in naked asm",
                    e->toChars());
        break;

      default:
        llvm_unreachable("Unsupported argument in asm.");
        break;
      }
    } else {
      insnTemplate << fmt << "<<" << (mode == Mode_Input ? "in" : "out")
                   << asmcode->args.size() << ">>";
      asmcode->args.push_back(AsmArg(type, e, mode));
    }
  }
  void addOperand2(const char *fmtpre, const char *fmtpost, AsmArgType type,
                   Expression *e, AsmCode *asmcode,
                   AsmArgMode mode = Mode_Input) {
    if (sc->func->naked) {
      // taken from above
      stmt->error("only global variables can be referenced by identifier in "
                  "naked asm");
      return;
    }

    insnTemplate << fmtpre << "<<" << (mode == Mode_Input ? "in" : "out")
                 << ">>" << fmtpost;
    asmcode->args.push_back(AsmArg(type, e, mode));
  }

  void addLabel(char *id) {
    // We need to delay emitting the actual function name, see
    // replace_func_name in asmstmt.cpp for details.
    printLabelName(insnTemplate, "<<func>>", id);
  }

  /* Determines whether the operand is a register, memory reference
     or immediate.  Immediate addresses are currently classified as
     memory.  This function is called before the exact instructions
     is known and thus, should not use opInfo. */
  void classifyOperand(Operand *operand) {
    operand->cls = classifyOperand1(operand);
  }

  OperandClass classifyOperand1(Operand *operand) {
    bool is_localsize = false;
    bool really_have_symbol = false;

    if (operand->symbolDisplacement.dim) {
      is_localsize =
          isLocalSize((Expression *)operand->symbolDisplacement.data[0]);
      really_have_symbol = !is_localsize;
    }

    if (operand->isOffset && !operand->hasBracket) {
      return Opr_Immediate;
    }

    if (operand->hasBracket ||
        really_have_symbol) // %% redo for 'offset' function
    {
      if (operand->reg != Reg_Invalid) {
        invalidExpression();
        return Opr_Invalid;
      }

      return Opr_Mem;
    }

    if (operand->reg != Reg_Invalid && operand->constDisplacement != 0) {
      invalidExpression();
      return Opr_Invalid;
    }

    if (operand->segmentPrefix != Reg_Invalid) {
      if (operand->reg != Reg_Invalid) {
        invalidExpression();
        return Opr_Invalid;
      }

      return Opr_Mem;
    }

    if (operand->reg != Reg_Invalid && !operand->hasNumber) {
      return Opr_Reg;
    }

    // should check immediate given (operand->hasNumber);
    //
    if (operand->hasNumber || is_localsize) {
      // size determination not correct if there are symbols Opr_Immediate
      if (operand->dataSize == Default_Ptr) {
        if (operand->constDisplacement < 0x100) {
          operand->dataSize = Byte_Ptr;
        } else if (operand->constDisplacement < 0x10000) {
          operand->dataSize = Short_Ptr;
        }
#ifndef ASM_X86_64
        else {
          operand->dataSize = Int_Ptr;
        }
#else
        else if (operand->constDisplacement <= 0xFFFFFFFF) {
          operand->dataSize = Int_Ptr;
        } else {
          // This could be possible since we are using 48 bits
          operand->dataSize = QWord_Ptr;
        }
#endif
      }
      return Opr_Immediate;
    }

    // probably a bug,?
    stmt->error("invalid operand");
    return Opr_Invalid;
  }

  void writeReg(Reg reg) { insnTemplate << "%" << regInfo[reg].gccName; }

  bool opTakesLabel() {
    switch (op) {
    case Op_Branch:
    case Op_CBranch:
    case Op_Loop:
      return true;
    default:
      return false;
    }
  }

  bool getTypeSuffix(TypeNeeded needed, PtrType ptrtype,
                     std::string &type_suffix) {
    switch (needed) {
    case Byte_NoType:
      return ptrtype == Byte_Ptr;
    case Word_Types:
      if (ptrtype == Byte_Ptr) {
        return false;
      }
    // drop through
    case Int_Types:
      switch (ptrtype) {
      case Byte_Ptr:
        type_suffix = 'b';
        break;
      case Short_Ptr:
        type_suffix = 'w';
        break;
      case Int_Ptr:
        type_suffix = 'l';
        break;
#ifdef ASM_X86_64
      case QWord_Ptr:
        type_suffix = 'q';
        break;
#endif
      default:
        // %% these may be too strict
        return false;
      }
      break;
    case FPInt_Types:
      switch (ptrtype) {
      case Short_Ptr:
        type_suffix = "";
        break;
      case Int_Ptr:
        type_suffix = 'l';
        break;
      case QWord_Ptr:
        type_suffix = "ll";
        break;
      default:
        return false;
      }
      break;
    case FP_Types:
      switch (ptrtype) {
      case Float_Ptr:
        type_suffix = 's';
        break;
      case Double_Ptr:
        type_suffix = 'l';
        break;
      case Extended_Ptr:
        // MS C runtime: real = 64-bit double
        if (global.params.targetTriple.isWindowsMSVCEnvironment()) {
          type_suffix = 'l';
        } else {
          type_suffix = 't';
        }
        break;
      default:
        return false;
      }
      break;
    default:
      return false;
    }
    return true;
  }

  // also set impl clobbers
  bool formatInstruction(int nOperands, AsmCode *asmcode) {
    const char *fmt;
    const char *mnemonic;
    std::string type_suffix;
    bool use_star;
    AsmArgMode mode;

    insnTemplate.str("");
    // %% todo: special case for something..
    if (opInfo->linkType == Out_Mnemonic) {
      mnemonic = alternateMnemonics[opInfo->link];
    } else {
      mnemonic = opIdent->string;
    }

    // handle two-operand form where second arg is ignored.
    // must be done before type_suffix detection
    if (op == Op_FidR_P || op == Op_fxch || op == Op_FfdRR_P) {
      if (operands[1].cls == Opr_Reg && operands[1].reg == Reg_ST) {
        nOperands = 1;
      } else {
        stmt->error("instruction allows only ST as second argument");
      }
    }

#ifdef ASM_X86_64
    if (op == Op_FCmpFlgP) {
      // Explicitly add %st as second argument to fucomip – it should
      // be implicit, but the GNU as shipping with Mac OS X (Apple Inc
      // version cctools-800~26, GNU assembler version 1.38) chokes on
      // it otherwise.
      assert(nOperands == 1);
      nOperands = 2;
      operands[1] = operands[0];
      memset(operands, 0, sizeof(Operand));
      operands[0].cls = Opr_Reg;
      operands[0].reg = Reg_ST;
    }
#endif

    if (opInfo->needsType) {
      PtrType exact_type = Default_Ptr;
      PtrType min_type = Default_Ptr;
      PtrType hint_type = Default_Ptr;

      /* Default types: This attempts to match the observed behavior of DMD
       */
      switch (opInfo->needsType) {
      case Int_Types:
        min_type = Byte_Ptr;
        break;
      case Word_Types:
        min_type = Short_Ptr;
        break;
      case FPInt_Types:
        if (op == Op_Fis_ST) { // integer math instructions
          min_type = Int_Ptr;
        } else { // compare, load, store
          min_type = Short_Ptr;
        }
        break;
      case FP_Types:
        min_type = Float_Ptr;
        break;
      }
      if (op == Op_push && operands[0].cls == Opr_Immediate) {
#ifndef ASM_X86_64
        min_type = Int_Ptr;
#else
        min_type = QWord_Ptr;
#endif
      }

      for (int i = 0; i < nOperands; i++) {
        if (hint_type == Default_Ptr && !(opInfo->operands[i] & Opr_NoType)) {
          hint_type = operands[i].dataSizeHint;
        }

        if ((opInfo->operands[i] & Opr_NoType) ||
            operands[i].dataSize == Default_Ptr) {
          continue;
        }
        if (operands[i].cls == Opr_Immediate) {
          min_type =
              operands[i].dataSize > min_type ? operands[i].dataSize : min_type;
        } else {
          exact_type =
              operands[i].dataSize; // could check for conflicting types
          break;
        }
      }

      bool type_ok;
      if (exact_type == Default_Ptr) {
        type_ok = getTypeSuffix(static_cast<TypeNeeded>(opInfo->needsType),
                                hint_type, type_suffix);
        if (!type_ok) {
          type_ok = getTypeSuffix(static_cast<TypeNeeded>(opInfo->needsType),
                                  min_type, type_suffix);
        }
      } else {
        type_ok = getTypeSuffix(static_cast<TypeNeeded>(opInfo->needsType),
                                exact_type, type_suffix);
      }

      if (!type_ok) {
        stmt->error("invalid operand size");
        return false;
      }
    } else if (op == Op_Branch) {
      if (operands[0].dataSize == Far_Ptr) { // %% type=Far_Ptr not set by
        // Seg:Ofss OTOH, we don't support
        // that..
        insnTemplate << 'l';
      }
    } else if (op == Op_FMath0 || op == Op_FdST0ST1) {
      operands[0].cls = Opr_Reg;
      operands[0].reg = Reg_ST1;
      operands[1].cls = Opr_Reg;
      operands[1].reg = Reg_ST;
      nOperands = 2;
    }

    switch (op) {
    case Op_SizedStack: {
      int mlen = strlen(mnemonic);
      if (mnemonic[mlen - 1] == 'd') {
        insnTemplate.write(mnemonic, mlen - 1);
      } else {
        insnTemplate << mnemonic << 'w';
      }
    } break;
    case Op_cmpsd:
    case Op_insX:
    case Op_lodsX:
    case Op_movsd:
    case Op_outsX:
    case Op_scasX:
    case Op_stosX: {
      int mlen = strlen(mnemonic);
      if (mnemonic[mlen - 1] == 'd') {
        insnTemplate.write(mnemonic, mlen - 1) << 'l';
      } else {
        insnTemplate << mnemonic;
      }
    } break;
    case Op_movsx:
    case Op_movzx: {
      char tc_1;
      int mlen = strlen(mnemonic);
      PtrType op1_size = operands[1].dataSize;
      if (op1_size == Default_Ptr) {
        op1_size = operands[1].dataSizeHint;
      }
      // Need type char for source arg
      switch (op1_size) {
      case Byte_Ptr:
      case Default_Ptr:
        tc_1 = 'b';
        break;
      case Short_Ptr:
        tc_1 = 'w';
        break;
      default:
        stmt->error("invalid operand size/type");
        return false;
      }
      assert(!type_suffix.empty());
      insnTemplate.write(mnemonic, mlen - 1) << tc_1 << type_suffix;
    } break;

    default:
      // special case fdiv, fsub: see dmd 840, ldc 256
      if ((strncmp(mnemonic, "fsub", 4) == 0 ||
           strncmp(mnemonic, "fdiv", 4) == 0) &&
          operands[0].reg != Reg_ST && op != Op_FMath) {
        // replace:
        //   f{sub,div}r{p,} <-> f{sub,div}{p,}
        if (mnemonic[4] == 'r') {
          insnTemplate.write(mnemonic, 4);
          insnTemplate.write(mnemonic + 5, strlen(mnemonic) - 5);
        } else {
          insnTemplate.write(mnemonic, 4) << "r";
          insnTemplate.write(mnemonic + 4, strlen(mnemonic) - 4);
        }
      } else {
        insnTemplate << mnemonic;
      }
      // the no-operand versions of floating point ops always pop
      if (op == Op_FMath0) {
        insnTemplate << "p";
      }
      insnTemplate << type_suffix;
      break;
    }

    switch (opInfo->implicitClobbers & Clb_DXAX_Mask) {
    case Clb_SizeAX:
    case Clb_EAX:
      asmcode->regs[Reg_EAX] = true;
      break;
    case Clb_SizeDXAX:
      asmcode->regs[Reg_EAX] = true;
      if (type_suffix != "b") {
        asmcode->regs[Reg_EDX] = true;
      }
      break;
#ifdef ASM_X86_64
    case Clb_SizeRDXRAX:
      asmcode->regs[Reg_RAX] = true;
      asmcode->regs[Reg_RDX] = true;
      break;
#endif
    default:
      // nothing
      break;
    }

    if (opInfo->implicitClobbers & Clb_DI) {
      asmcode->regs[Reg_EDI] = true;
    }
    if (opInfo->implicitClobbers & Clb_SI) {
      asmcode->regs[Reg_ESI] = true;
    }
    if (opInfo->implicitClobbers & Clb_CX) {
      asmcode->regs[Reg_ECX] = true;
    }
    if (opInfo->implicitClobbers & Clb_SP) {
      asmcode->regs[Reg_ESP] = true;
    }
    if (opInfo->implicitClobbers & Clb_ST) {
      /* Can't figure out how to tell GCC that an
         asm statement leaves an arg pushed on the stack.
         Maybe if the statment had and input or output
         operand it would work...  In any case, clobbering
         all FP prevents incorrect code generation. */
      asmcode->regs[Reg_ST] = true;
      asmcode->regs[Reg_ST1] = true;
      asmcode->regs[Reg_ST2] = true;
      asmcode->regs[Reg_ST3] = true;
      asmcode->regs[Reg_ST4] = true;
      asmcode->regs[Reg_ST5] = true;
      asmcode->regs[Reg_ST6] = true;
      asmcode->regs[Reg_ST7] = true;
    }
    if (opInfo->implicitClobbers & Clb_Flags) {
      asmcode->regs[Reg_EFLAGS] = true;
    }
    if (op == Op_cpuid) {
      asmcode->regs[Reg_EAX] = true;
      asmcode->regs[Reg_EBX] = true;
      asmcode->regs[Reg_ECX] = true;
      asmcode->regs[Reg_EDX] = true;
    }

    insnTemplate << ' ';
    for (int i__ = 0; i__ < nOperands; i__++) {
      int i;
      if (i__ != 0) {
        insnTemplate << ", ";
      }
      fmt = "$";

      switch (op) {
      case Op_mul:
        // gas won't accept the two-operand form; skip to the source
        // operand
        i__ = 1;
      // drop through
      case Op_bound:
      case Op_enter:
        i = i__;
        break;
      default:
        i = nOperands - 1 - i__; // operand = & operands[ nOperands - 1 - i ];
        break;
      }
      operand = &operands[i];

      switch (operand->cls) {
      case Opr_Immediate:
        // for implementing offset:
        // $var + $7 // fails
        // $var + 7  // ok
        // $7 + $var // ok

        // DMD doesn't seem to allow this
        /*
        if ( opTakesLabel() )  tho... (near ptr <Number> would be abs?)
            fmt = "$a"; // GAS won't accept "jmp $42"; must be "jmp 42"
        (rel) or
        "jmp *42" (abs)
        */
        if (opTakesLabel()) {
          // "relative addressing not allowed in branch instructions" ..
          stmt->error("integer constant not allowed in branch instructions");
          return false;
        }

        if (operand->symbolDisplacement.dim &&
            isLocalSize((Expression *)operand->symbolDisplacement.data[0])) {
          // handle __LOCAL_SIZE, which in this constant, is an immediate
          // should do this in slotexp..
          addOperand("$", Arg_LocalSize,
                     (Expression *)operand->symbolDisplacement.data[0],
                     asmcode);
          if (operand->constDisplacement) {
            insnTemplate << '+';
          } else {
            break;
          }
        }

        if (operand->symbolDisplacement.dim) {
          fmt = "$a";
          addOperand("$", Arg_Pointer,
                     (Expression *)operand->symbolDisplacement.data[0],
                     asmcode);

          if (operand->constDisplacement) {
            insnTemplate << '+';
          } else {
            // skip the addOperand(fmt, Arg_Integer...) below
            break;
          }
        }
        addOperand(fmt, Arg_Integer, newIntExp(operand->constDisplacement),
                   asmcode);
        break;
      case Opr_Reg:
        if (opInfo->operands[i] & Opr_Dest) {
          Reg clbr_reg = static_cast<Reg>(regInfo[operand->reg].baseReg);
          if (clbr_reg != Reg_Invalid) {
            asmcode->regs[clbr_reg] = true;
          }
        }
        if (opTakesLabel()) {
          insnTemplate << '*';
        }
        writeReg(operand->reg);
        /*
        insnTemplate << "%";
        insnTemplate << regInfo[operand->reg].name;
        */
        break;
      case Opr_Mem:
        // better: use output operands for simple variable references
        if ((opInfo->operands[i] & Opr_Update) == Opr_Update) {
          mode = Mode_Update;
        } else if (opInfo->operands[i] & Opr_Dest) {
          mode = Mode_Output;
        } else {
          mode = Mode_Input;
        }

        use_star = opTakesLabel();

        if (Logger::enabled()) {
          Logger::cout() << "Opr_Mem\n";
          LOG_SCOPE
          Logger::cout() << "baseReg: " << operand->baseReg << '\n';
          Logger::cout() << "segmentPrefix: " << operand->segmentPrefix << '\n';
          Logger::cout() << "constDisplacement: " << operand->constDisplacement
                         << '\n';
          for (unsigned i = 0; i < operand->symbolDisplacement.dim; i++) {
            Expression *expr =
                static_cast<Expression *>(operand->symbolDisplacement.data[i]);
            Logger::cout() << "symbolDisplacement[" << i
                           << "] = " << expr->toChars() << '\n';
          }
        }
        if (operand->segmentPrefix != Reg_Invalid) {
          if (op != Op_Branch) {
            writeReg(operand->segmentPrefix);
            insnTemplate << ':';
          } else {
            stmt->error("Cannot generate a segment prefix for a "
                        "branching instruction");
          }
        }
        if ((operand->segmentPrefix != Reg_Invalid &&
             operand->symbolDisplacement.dim == 0) ||
            operand->constDisplacement) {
          insnTemplate << operand->constDisplacement;
          if (operand->symbolDisplacement.dim) {
            insnTemplate << '+';
          }
          operand->constDisplacement = 0;
          // addOperand(fmt, Arg_Integer,
          // newIntExp(operand->constDisplacement),
          // asmcode);
          if (opInfo->operands[i] & Opr_Dest) {
            asmcode->clobbersMemory = 1;
          }
        }
        if (operand->symbolDisplacement.dim) {
          Expression *e = (Expression *)operand->symbolDisplacement.data[0];
          Declaration *decl = nullptr;

          if (e->op == TOKvar) {
            decl = ((VarExp *)e)->var;
          }

          if (operand->baseReg != Reg_Invalid && decl && !decl->isDataseg()) {

            // Use the offset from frame pointer

            /* GCC doesn't give the front end access to stack offsets
               when optimization is turned on (3.x) or at all (4.x).

               Try to convert var[EBP] (or var[ESP] for naked funcs) to
               a memory expression that does not require us to know
               the stack offset.
            */

            if (operand->indexReg == Reg_Invalid && decl->isVarDeclaration() &&
#ifndef ASM_X86_64
                ((operand->baseReg == Reg_EBP && !sc->func->naked) ||
                 (operand->baseReg == Reg_ESP && sc->func->naked)))
#else
                (((operand->baseReg == Reg_EBP ||
                   operand->baseReg == Reg_RBP) &&
                  !sc->func->naked) ||
                 ((operand->baseReg == Reg_ESP ||
                   operand->baseReg == Reg_RSP) &&
                  sc->func->naked)))
#endif
            {

              e = new AddrExp(Loc(), e);
              e->type = decl->type->pointerTo();

              operand->constDisplacement = 0;
              operand->baseReg = Reg_Invalid;

              addOperand(fmt, Arg_Memory, e, asmcode, mode);

            } else {
              // FIXME: what is this ?
              addOperand2("${", ":a}", Arg_FrameRelative, e, asmcode);
            }
            if (opInfo->operands[i] & Opr_Dest) {
              asmcode->clobbersMemory = 1;
            }
          } else {
            // Plain memory reference to variable or reference to label.

            /* If in a reg, DMD moves to memory.. even with -O, so we'll
               do the
               same
               by always using the "m" contraint.

               In order to get the correct output for function and label
               symbols,
               the %an format must be used with the "p" constraint.
            */
            if (isDollar(e)) {
              stmt->error("dollar labels are not supported");
              asmcode->dollarLabel = 1;
            } else if (e->op == TOKdsymbol) {
              LabelDsymbol *lbl = (LabelDsymbol *)((DsymbolExp *)e)->s;
              stmt->isBranchToLabel = lbl;

              use_star = false;
              addLabel(lbl->ident->toChars());
            } else if ((decl && decl->isCodeseg())) // if function or label
            {
              use_star = false;
              // simply write out the mangle
              if (prependExtraUnderscore()) {
                insnTemplate << "_";
              }
              insnTemplate << mangle(decl);
              //              addOperand2("${", ":c}", Arg_Pointer, e,
              //              asmcode);
            } else {
              if (use_star) {
                insnTemplate << '*';
                use_star = false;
              }

              if (!sc->func->naked) // no addrexp in naked asm please :)
              {
                Type *tt = e->type->pointerTo();
                e = new AddrExp(Loc(), e);
                e->type = tt;
              }

              addOperand(fmt, Arg_Memory, e, asmcode, mode);
            }
          }
        }
        if (use_star) {
          insnTemplate << '*';
        }
        if (operand->baseReg != Reg_Invalid ||
            operand->indexReg != Reg_Invalid) {
          insnTemplate << '(';
          if (operand->baseReg != Reg_Invalid) {
            writeReg(operand->baseReg);
          }
          if (operand->indexReg != Reg_Invalid) {
            insnTemplate << ',';
            writeReg(operand->indexReg);
            if (operand->scale) {
              insnTemplate << "," << operand->scale;
            }
          }
          insnTemplate << ')';
          if (opInfo->operands[i] & Opr_Dest) {
            asmcode->clobbersMemory = 1;
          }
        }
        break;
      case Opr_Invalid:
        return false;
      }
    }

    asmcode->insnTemplate = insnTemplate.str();
    Logger::cout() << "insnTemplate = " << asmcode->insnTemplate << '\n';
    return true;
  }

  bool isIntExp(Expression *exp) { return exp->op == TOKint64; }
  bool isRegExp(Expression *exp) { return exp->op == TOKmod; } // ewww.%%
  bool isLocalSize(Expression *exp) {
    // cleanup: make a static var
    return exp->op == TOKidentifier &&
           ((IdentifierExp *)exp)->ident == Id::__LOCAL_SIZE;
  }
  bool isDollar(Expression *exp) {
    return exp->op == TOKidentifier &&
           ((IdentifierExp *)exp)->ident == Id::dollar;
  }

  Expression *newRegExp(int regno) {
    auto e = new IntegerExp(regno);
    e->op = TOKmod;
    return e;
  }

#ifndef ASM_X86_64
  Expression *newIntExp(int v /* %% type */) {
    // Only handles 32-bit numbers as this is IA-32.
    return new IntegerExp(stmt->loc, v, Type::tint32);
  }
#else
  Expression *newIntExp(long long v /* %% type */) {
    // Handle 64 bit
    return new IntegerExp(stmt->loc, v, Type::tint64);
  }
#endif

  void slotExp(Expression *exp) {
    /*
      if offset, make a note

      if integer, add to immediate
      if reg:
          if not in bracket, set reg (else error)
          if in bracket:
         if not base, set base
         if not index, set index
         else, error
      if symbol:
        set symbol field
     */

    bool is_offset = false;
    if (exp->op == TOKaddress) {
      exp = ((AddrExp *)exp)->e1;
      is_offset = true;
    }

    if (isIntExp(exp)) {
      if (is_offset) {
        invalidExpression();
      }
      operand->constDisplacement += exp->toInteger();
      if (!operand->inBracket) {
        operand->hasNumber = 1;
      }
    } else if (isRegExp(exp)) {
      if (is_offset) {
        invalidExpression();
      }
      if (!operand->inBracket) {
        if (operand->reg == Reg_Invalid) {
          operand->reg = static_cast<Reg>(exp->toInteger());
        } else {
          stmt->error("too many registers in operand (use brackets)");
        }
      } else {
        if (operand->baseReg == Reg_Invalid) {
          operand->baseReg = static_cast<Reg>(exp->toInteger());
        } else if (operand->indexReg == Reg_Invalid) {
          operand->indexReg = static_cast<Reg>(exp->toInteger());
          operand->scale = 1;
        } else {
          stmt->error("too many registers memory operand");
        }
      }
    } else if (exp->op == TOKvar) {
      VarDeclaration *v = ((VarExp *)exp)->var->isVarDeclaration();

      if (v && v->storage_class & STCfield) {
        operand->constDisplacement += v->offset;
        if (!operand->inBracket) {
          operand->hasNumber = 1;
        }
      } else {
        if (v && v->type->isscalar()) {
          // DMD doesn't check Tcomplex*, and counts Tcomplex32 as
          // Tfloat64
          TY ty = v->type->toBasetype()->ty;
          operand->dataSizeHint =
              ty == Tfloat80 || ty == Timaginary80
                  ? Extended_Ptr
                  : static_cast<PtrType>(v->type->size(Loc()));
        }

        if (!operand->symbolDisplacement.dim) {
          if (is_offset && !operand->inBracket) {
            operand->isOffset = 1;
          }
          operand->symbolDisplacement.push(exp);
        } else {
          stmt->error("too many symbols in operand");
        }
      }
    } else if (exp->op == TOKidentifier || exp->op == TOKdsymbol) {
      // %% localsize could be treated as a simple constant..
      // change to addSymbolDisp(e)
      if (!operand->symbolDisplacement.dim) {
        operand->symbolDisplacement.push(exp);
      } else {
        stmt->error("too many symbols in operand");
      }
    } else if (exp == Handled) {
      // nothing
    } else {
      stmt->error("invalid operand");
    }
  }

  void invalidExpression() {
    // %% report operand number
    stmt->error("invalid expression");
  }

  Expression *intOp(TOK op, Expression *e1, Expression *e2) {
    Expression *e;
    if (isIntExp(e1) && (!e2 || isIntExp(e2))) {
      switch (op) {
      case TOKadd:
        if (e2) {
          e = new AddExp(stmt->loc, e1, e2);
        } else {
          e = e1;
        }
        break;
      case TOKmin:
        if (e2) {
          e = new MinExp(stmt->loc, e1, e2);
        } else {
          e = new NegExp(stmt->loc, e1);
        }
        break;
      case TOKmul:
        e = new MulExp(stmt->loc, e1, e2);
        break;
      case TOKdiv:
        e = new DivExp(stmt->loc, e1, e2);
        break;
      case TOKmod:
        e = new ModExp(stmt->loc, e1, e2);
        break;
      case TOKshl:
        e = new ShlExp(stmt->loc, e1, e2);
        break;
      case TOKshr:
        e = new ShrExp(stmt->loc, e1, e2);
        break;
      case TOKushr:
        e = new UshrExp(stmt->loc, e1, e2);
        break;
      case TOKnot:
        e = new NotExp(stmt->loc, e1);
        break;
      case TOKtilde:
        e = new ComExp(stmt->loc, e1);
        break;
      case TOKoror:
        e = new OrOrExp(stmt->loc, e1, e2);
        break;
      case TOKandand:
        e = new AndAndExp(stmt->loc, e1, e2);
        break;
      case TOKor:
        e = new OrExp(stmt->loc, e1, e2);
        break;
      case TOKand:
        e = new AndExp(stmt->loc, e1, e2);
        break;
      case TOKxor:
        e = new XorExp(stmt->loc, e1, e2);
        break;
      case TOKequal:
      case TOKnotequal:
        e = new EqualExp(op, stmt->loc, e1, e2);
        break;
      case TOKgt:
      case TOKge:
      case TOKlt:
      case TOKle:
        e = new CmpExp(op, stmt->loc, e1, e2);
        break;
      default:
        llvm_unreachable("Unknown integer operation.");
      }
      e = e->semantic(sc);
      return e->ctfeInterpret();
    }
    stmt->error("expected integer operand(s) for '%s'", Token::tochars[op]);
    return newIntExp(0);
  }

  void parseOperand() {
    Expression *exp = parseAsmExp();
    slotExp(exp);
    if (isRegExp(exp)) {
      operand->dataSize = static_cast<PtrType>(regInfo[exp->toInteger()].size);
    }
  }

  Expression *parseAsmExp() { return parseCondExp(); }

  Expression *parseCondExp() {
    Expression *exp = parseLogOrExp();
    if (token->value == TOKquestion) {
      nextToken();
      Expression *exp2 = parseCondExp();
      if (token->value != TOKcolon) {
        return exp;
      }
      nextToken();
      Expression *exp3 = parseCondExp();
      exp = exp->toUInteger() ? exp2 : exp3;
    }
    return exp;
  }

  Expression *parseLogOrExp() {
    Expression *exp = parseLogAndExp();
    while (token->value == TOKoror) {
      nextToken();
      Expression *exp2 = parseLogAndExp();
      if (isIntExp(exp) && isIntExp(exp2)) {
        exp = intOp(TOKandand, exp, exp2);
      } else {
        stmt->error("bad integral operand");
      }
    }
    return exp;
  }

  Expression *parseLogAndExp() {
    Expression *exp = parseIncOrExp();
    while (token->value == TOKoror) {
      nextToken();
      Expression *exp2 = parseIncOrExp();
      if (isIntExp(exp) && isIntExp(exp2)) {
        exp = intOp(TOKoror, exp, exp2);
      } else {
        stmt->error("bad integral operand");
      }
    }
    return exp;
  }

  Expression *parseIncOrExp() {
    Expression *exp = parseXOrExp();
    while (token->value == TOKor) {
      nextToken();
      Expression *exp2 = parseXOrExp();
      if (isIntExp(exp) && isIntExp(exp2)) {
        exp = intOp(TOKor, exp, exp2);
      } else {
        stmt->error("bad integral operand");
      }
    }
    return exp;
  }

  Expression *parseXOrExp() {
    Expression *exp = parseAndExp();
    while (token->value == TOKxor) {
      nextToken();
      Expression *exp2 = parseAndExp();
      if (isIntExp(exp) && isIntExp(exp2)) {
        exp = intOp(TOKxor, exp, exp2);
      } else {
        stmt->error("bad integral operand");
      }
    }
    return exp;
  }

  Expression *parseAndExp() {
    Expression *exp = parseEqualExp();
    while (token->value == TOKand) {
      nextToken();
      Expression *exp2 = parseEqualExp();
      if (isIntExp(exp) && isIntExp(exp2)) {
        exp = intOp(TOKand, exp, exp2);
      } else {
        stmt->error("bad integral operand");
      }
    }
    return exp;
  }

  Expression *parseEqualExp() {
    Expression *exp = parseRelExp();
    while (1) {
      switch (token->value) {
      case TOKequal:
      case TOKnotequal: {
        TOK tok = token->value;
        nextToken();
        Expression *exp2 = parseRelExp();
        if (isIntExp(exp) && isIntExp(exp2)) {
          exp = intOp(tok, exp, exp2);
        } else {
          stmt->error("bad integral operand");
        }
      }
      default:
        return exp;
      }
    }
    return exp;
  }

  Expression *parseRelExp() {
    Expression *exp = parseShiftExp();
    while (1) {
      switch (token->value) {
      case TOKgt:
      case TOKge:
      case TOKlt:
      case TOKle: {
        TOK tok = token->value;
        nextToken();
        Expression *exp2 = parseShiftExp();
        if (isIntExp(exp) && isIntExp(exp2)) {
          exp = intOp(tok, exp, exp2);
        } else {
          stmt->error("bad integral operand");
        }
      }
      default:
        return exp;
      }
    }
    return exp;
  }

  Expression *parseShiftExp() {
    Expression *e1 = parseAddExp();
    Expression *e2;

    while (1) {
      TOK tv = token->value;
      switch (tv) {
      case TOKshl:
      case TOKshr:
      case TOKushr:
        nextToken();
        e2 = parseAddExp();
        e1 = intOp(tv, e1, e2);
        continue;
      default:
        break;
      }
      break;
    }
    return e1;
  }

  Expression *parseAddExp() {
    Expression *e1 = parseMultExp();
    Expression *e2;

    while (1) {
      TOK tv = token->value;
      switch (tv) {
      case TOKadd:
        nextToken();
        e2 = parseMultExp();
        if (isIntExp(e1) && isIntExp(e2)) {
          e1 = intOp(tv, e1, e2);
        } else {
          slotExp(e1);
          slotExp(e2);
          e1 = Handled;
        }
        continue;
      case TOKmin:
        // Note: no support for symbol address difference
        nextToken();
        e2 = parseMultExp();
        if (isIntExp(e1) && isIntExp(e2)) {
          e1 = intOp(tv, e1, e2);
        } else {
          slotExp(e1);
          e2 = intOp(TOKmin, e2, nullptr); // verifies e2 is an int
          slotExp(e2);
          e1 = Handled;
        }
        continue;
      default:
        break;
      }
      break;
    }
    return e1;
  }

  bool tryScale(Expression *e1, Expression *e2) {
    Expression *et;
    if (isIntExp(e1) && isRegExp(e2)) {
      et = e1;
      e1 = e2;
      e2 = et;
      goto do_scale;
    } else if (isRegExp(e1) && isIntExp(e2)) {
    do_scale:
      if (!operand->inBracket) {
        invalidExpression(); // maybe should allow, e.g. DS:EBX+EAX*4
      }

      if (operand->scale || operand->indexReg != Reg_Invalid) {
        invalidExpression();
        return true;
      }

      operand->indexReg = static_cast<Reg>(e1->toInteger());
      operand->scale = e2->toInteger();
      switch (operand->scale) {
      case 1:
      case 2:
      case 4:
      case 8:
        // ok; do nothing
        break;
      default:
        stmt->error("invalid index register scale '%d'", operand->scale);
        return true;
      }

      return true;
    }
    return false;
  }

  Expression *parseMultExp() {
    Expression *e1 = parseBrExp();
    Expression *e2;

    while (1) {
      TOK tv = token->value;
      switch (tv) {
      case TOKmul:
        nextToken();
        e2 = parseMultExp();
        if (isIntExp(e1) && isIntExp(e2)) {
          e1 = intOp(tv, e1, e2);
        } else if (tryScale(e1, e2)) {
          e1 = Handled;
        } else {
          invalidExpression();
        }
        continue;
      case TOKdiv:
      case TOKmod:
        nextToken();
        e2 = parseMultExp();
        e1 = intOp(tv, e1, e2);
        continue;
      default:
        break;
      }
      break;
    }
    return e1;
  }

  Expression *parseBrExp() {
    // %% check (why is bracket lower precends..)
    // 3+4[eax] -> 3 + (4 [EAX]) ..

    // only one bracked allowed, so this doesn't quite handle
    // the spec'd syntax
    Expression *e;

    if (token->value == TOKlbracket) {
      e = Handled;
    } else {
      e = parseUnaExp();
    }

    // DMD allows multiple bracket expressions.
    while (token->value == TOKlbracket) {
      nextToken();

      operand->inBracket = operand->hasBracket = 1;
      slotExp(parseAsmExp());
      operand->inBracket = 0;

      if (token->value == TOKrbracket) {
        nextToken();
      } else {
        stmt->error("missing ']'");
      }
    }

    return e;
  }

  PtrType isPtrType(Token *tok) {
    switch (tok->value) {
    case TOKint8:
      return Byte_Ptr;
    case TOKint16:
      return Short_Ptr;
    case TOKint32:
      return Int_Ptr;
#ifndef ASM_X86_64
// 'long ptr' isn't accepted?
#else
    case TOKint64:
      return QWord_Ptr;
#endif
    case TOKfloat32:
      return Float_Ptr;
    case TOKfloat64:
      return Double_Ptr;
    case TOKfloat80:
      return Extended_Ptr;
    case TOKidentifier:
      for (int i = 0; i < N_PtrNames; i++) {
        if (tok->ident == ptrTypeIdentTable[i]) {
          return ptrTypeValueTable[i];
        }
      }
      break;
    default:
      break;
    }
    return Default_Ptr;
  }

  Expression *parseUnaExp() {
    Expression *e = nullptr;
    PtrType ptr_type;

    // First, check for type prefix.
    if (token->value != TOKeof && peekToken()->value == TOKidentifier &&
        peekToken()->ident == Id::ptr) {

      ptr_type = isPtrType(token);
      if (ptr_type != Default_Ptr) {
        if (operand->dataSize == Default_Ptr) {
          operand->dataSize = ptr_type;
        } else {
          stmt->error("multiple specifications of operand size");
        }
      } else {
        stmt->error("unknown operand size '%s'", token->toChars());
      }
      nextToken();
      nextToken();
      return parseAsmExp();
    }

    TOK tv = token->value;
    switch (tv) {
    case TOKidentifier:
      if (token->ident == ident_seg) {
        nextToken();
        stmt->error("'seg' not supported");
        e = parseAsmExp();
      } else if (token->ident == Id::offset || token->ident == Id::offsetof) {
        if (token->ident == Id::offset && !global.params.useDeprecated) {
          stmt->error("offset deprecated, use offsetof");
        }
        nextToken();
        e = parseAsmExp();
        e = new AddrExp(stmt->loc, e);
      } else {
        // primary exp
        break;
      }
      return e;
    case TOKadd:
    case TOKmin:
    case TOKnot:
    case TOKtilde:
      nextToken();
      e = parseUnaExp();
      return intOp(tv, e, nullptr);
    default:
      // primary exp
      break;
    }
    return parsePrimaryExp();
  }

  Expression *parsePrimaryExp() {
    Expression *e;
    Identifier *ident = nullptr;

    // get rid of short/long prefixes for branches
    if (opTakesLabel() &&
        (token->value == TOKint16 || token->value == TOKint64)) {
      nextToken();
    }

    switch (token->value) {
    case TOKint32v:
    case TOKuns32v:
    case TOKint64v:
    case TOKuns64v:
// semantic here?
#ifndef ASM_X86_64
      // %% for tok64 really should use 64bit type
      e = new IntegerExp(stmt->loc, token->uns64value, Type::tint32);
#else
      e = new IntegerExp(stmt->loc, token->uns64value, Type::tint64);
#endif
      nextToken();
      break;
    case TOKfloat32v:
    case TOKfloat64v:
    case TOKfloat80v:
      // %% need different types?
      e = new RealExp(stmt->loc, token->float80value, Type::tfloat80);
      nextToken();
      break;
    case TOKidentifier: {
      ident = token->ident;
      nextToken();

      if (ident == Id::__LOCAL_SIZE) {
        return new IdentifierExp(stmt->loc, ident);
      }
      if (ident == Id::dollar) {
      do_dollar:
        return new IdentifierExp(stmt->loc, ident);
      } else {
        e = new IdentifierExp(stmt->loc, ident);
      }

      // If this is more than one component ref, it gets complicated:
      // *(&Field +
      // n)
      // maybe just do one step at a time..
      // simple case is Type.f -> VarDecl(field)
      // actually, DMD only supports on level...
      // X.y+Y.z[EBX] is supported, tho..
      // %% doesn't handle properties (check%%)
      while (token->value == TOKdot) {
        nextToken();
        if (token->value == TOKidentifier) {
          e = new DotIdExp(stmt->loc, e, token->ident);
          nextToken();
        } else {
          stmt->error("expected identifier");
          return Handled;
        }
      }

      // check for reg first then dotexp is an error?
      if (e->op == TOKidentifier) {
        for (int i = 0; i < N_Regs; i++) {
          if (ident == regInfo[i].ident) {
            if (static_cast<Reg>(i) == Reg_ST && token->value == TOKlparen) {
              nextToken();
              switch (token->value) {
              case TOKint32v:
              case TOKuns32v:
              case TOKint64v:
              case TOKuns64v:
                if (token->uns64value < 8) {
                  e = newRegExp(static_cast<Reg>(Reg_ST + token->uns64value));
                } else {
                  stmt->error("invalid floating point register index");
                  e = Handled;
                }
                nextToken();
                if (token->value == TOKrparen) {
                  nextToken();
                } else {
                  stmt->error("expected ')'");
                }
                return e;
              default:
                break;
              }
              invalidExpression();
              return Handled;
            }
            if (token->value == TOKcolon) {
              nextToken();
              if (operand->segmentPrefix != Reg_Invalid) {
                stmt->error("too many segment prefixes");
              } else if (i >= Reg_CS && i <= Reg_GS) {
                operand->segmentPrefix = static_cast<Reg>(i);
              } else {
                stmt->error("'%s' is not a segment register", ident->string);
              }
              return parseAsmExp();
            }
            return newRegExp(static_cast<Reg>(i));
          }
        }
      }

      if (e->op == TOKidentifier) {
        // DMD uses labels secondarily to other symbols, so check
        // if IdentifierExp::semantic won't find anything.
        Dsymbol *scopesym;
        if (!sc->search(stmt->loc, ident, &scopesym)) {
          if (LabelDsymbol *labelsym = sc->func->searchLabel(ident)) {
            e = new DsymbolExp(stmt->loc, labelsym);
            if (opTakesLabel()) {
              return e;
            }
            return new AddrExp(stmt->loc, e);
          }
        }
      }

      e = e->semantic(sc);
      e = e->optimize(WANTvalue);

      // Special case for floating point constant declarations.
      if (e->op == TOKfloat64) {
        Dsymbol *sym = sc->search(stmt->loc, ident, nullptr);
        if (sym) {
          VarDeclaration *v = sym->isVarDeclaration();
          if (v) {
            Expression *ve = new VarExp(stmt->loc, v);
            ve->type = e->type;
            e = ve;
          }
        }
      }
      return e;
    } break;
    case TOKdollar:
      nextToken();
      ident = Id::dollar;
      goto do_dollar;
      break;
    default:
      if (op == Op_FMath0 || op == Op_FdST0ST1 || op == Op_FMath) {
        return Handled;
      }
      // DMD does not emit an error message here.
      // invalidExpression();
      return Handled;
    }
    return e;
  }

  void doAlign() {
    // .align bits vs. bytes...
    // apparently a.out platforms use bits instead of bytes...

    // parse primary: DMD allows 'MyAlign' (const int) but not '2+2'
    // GAS is padding with NOPs last time I checked.
    Expression *e = parseAsmExp()->ctfeInterpret();
    uinteger_t align = e->toUInteger();

    if ((align & (align - 1)) == 0) {
// FIXME: This printf is not portable. The use of `align` varies from system to
// system;
// on i386 using a.out, .align `n` will align on a 2^`n` boundary instead of an
// `n` boundary
#ifdef HAVE_GAS_BALIGN_AND_P2ALIGN
      insnTemplate << ".balign\t" << align;
#else
      insnTemplate << ".align\t" << align;
#endif
    } else {
      stmt->error("alignment must be a power of 2, not %u",
                  static_cast<unsigned>(align));
    }

    setAsmCode();
  }

  void doEven() {
// .align for GAS is in bits, others probably use bytes..
#ifdef HAVE_GAS_BALIGN_AND_P2ALIGN
    insnTemplate << ".align\t2";
#else
    insnTemplate << ".align\t2";
#endif
    setAsmCode();
  }

  void doNaked() {
    // %% can we assume sc->func != 0?
    sc->func->naked = 1;
  }

  void doData() {
// FIXME: data instructions not implemented.
#if 0
            static const char * directives[] = { ".byte", ".short", ".long", ".long",
                                                 "", "", ""
                                               };

                machine_mode mode;

                insnTemplate->writestring(static_cast<char*>(directives[op - Op_db]));
                insnTemplate->writebyte(' ');

                do {
                    // DMD is pretty strict here, not even constant expressions are allowed..
                    switch (op) {
                    case Op_db:
                    case Op_ds:
                    case Op_di:
                    case Op_dl:
                    if (token->value == TOKint32v || token->value == TOKuns32v ||
                        token->value == TOKint64v || token->value == TOKuns64v) {
                        // As per usual with GNU, assume at least 32-bit host
                        if (op != Op_dl)
                        insnTemplate->printf("%u", (d_uns32) token->uns64value);
                        else {
                        // Output two .longS.  GAS has .quad, but would have to rely on 'L' format ..
                        // just need to use HOST_WIDE_INT_PRINT_DEC
                        insnTemplate->printf("%u,%u",
                            (d_uns32) token->uns64value, (d_uns32) (token->uns64value >> 32));
                        }
                    } else {
                        stmt->error("expected integer constant");
                    }
                    break;
                    case Op_df:
                    mode = SFmode;
                    goto do_float;
                    case Op_dd:
                    mode = DFmode;
                    goto do_float;
                    case Op_de:
#ifndef TARGET_80387
#define XFmode TFmode
#endif
                    mode = XFmode; // not TFmode
                    // drop through
                    do_float:
                    if (token->value == TOKfloat32v || token->value == TOKfloat64v ||
                        token->value == TOKfloat80v) {
                        long words[3];
                        real_to_target(words, & token->float80value.rv(), mode);
                        // don't use directives..., just use .long like GCC
                        insnTemplate->printf(".long\t%u", words[0]);
                        if (mode != SFmode)
                        insnTemplate->printf(",%u", words[1]);
                        // DMD outputs 10 bytes, so we need to switch to .short here
                        if (mode == XFmode)
                        insnTemplate->printf("\n\t.short\t%u", words[2]);
                    } else {
                        stmt->error("expected float constant");
                    }
                    break;
                    default:
                    abort();
                    }

                    nextToken();
                    if (token->value == TOKcomma) {
                    insnTemplate->writebyte(',');
                    nextToken();
                    } else if (token->value == TOKeof) {
                    break;
                    } else {
                    stmt->error("expected comma");
                    }
                } while (1);

                setAsmCode();
#endif
  }
};

#if D_GCC_VER < 40
// struct rtx was modified for c++; this macro from rtl.h needs to
// be modified accordingly.
#undef XEXP
#define XEXP(RTX, N) (RTL_CHECK2(RTX, N, 'e', 'u').rt_rtx)
#endif

// FIXME
#define HOST_WIDE_INT long
bool getFrameRelativeValue(LLValue *decl, HOST_WIDE_INT *result) {
  llvm_unreachable("getFrameRelativeValue not implemented.");
  // FIXME
  //     // Using this instead of DECL_RTL for struct args seems like a
  //     // good way to get hit by a truck because it may not agree with
  //     // non-asm access, but asm code wouldn't know what GCC does
  //     anyway. */
  //     rtx r = DECL_INCOMING_RTL(decl);
  //     rtx e1, e2;
  //
  //     // Local variables don't have DECL_INCOMING_RTL
  //     if (r == NULL_RTX)
  //  r = DECL_RTL(decl);
  //
  //     if (r != NULL_RTX && GET_CODE(r) == MEM /* && r->frame_related */
  //     ) {
  //  r = XEXP(r, 0);
  //  if (GET_CODE(r) == PLUS) {
  //      e1 = XEXP(r, 0);
  //      e2 = XEXP(r, 1);
  //      if (e1 == virtual_incoming_args_rtx && GET_CODE(e2) ==
  //      CONST_INT) {
  //      *result = INTVAL(e2) + 8; // %% 8 is 32-bit specific...
  //      return true;
  //      } else if (e1 == virtual_stack_vars_rtx && GET_CODE(e2) ==
  //      CONST_INT)
  //      {
  //      *result = INTVAL(e2); // %% 8 is 32-bit specific...
  //      return true;
  //      }
  //  } else if (r == virtual_incoming_args_rtx) {
  //      *result = 8;
  //      return true; // %% same as above
  //  }
  //  // shouldn't have virtual_stack_vars_rtx by itself
  //     }
  //
  return false;
}

struct AsmParser : public AsmParserCommon {
  void run(Scope *sc, AsmStatement *asmst) override {
    AsmProcessor ap(sc, asmst);
    ap.run();
  }

  std::string getRegName(int i) override { return regInfo[i].gccName; }
};
}
