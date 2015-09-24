/**
 * This module contains functions and structures required for exception
 * handling which are shared by platform-specific implementations.
 */
module ldc.eh.common;

// debug = EH_personality;
// debug = EH_personality_verbose;

import core.memory : GC;
import core.stdc.stdio;
import core.stdc.stdlib;
import core.stdc.stdarg;

// D runtime function
extern(C) int _d_isbaseof(ClassInfo oc, ClassInfo c);

// error and exit
extern(C) void fatalerror(in char* format, ...)
{
    va_list args;
    va_start(args, format);
    fprintf(stderr, "Fatal error in EH code: ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    abort();
}

// ------------------------
//    Reading DWARF data
// ------------------------

version (Win64) {} else
{
    extern(C)
    {
        alias void* _Unwind_Context_Ptr;
        ptrdiff_t _Unwind_GetRegionStart(_Unwind_Context_Ptr context);
        ptrdiff_t _Unwind_GetTextRelBase(_Unwind_Context_Ptr context);
        ptrdiff_t _Unwind_GetDataRelBase(_Unwind_Context_Ptr context);
    }
}

enum _DW_EH_Format : int
{
    DW_EH_PE_absptr  = 0x00,  // The Value is a literal pointer whose size is determined by the architecture.
    DW_EH_PE_uleb128 = 0x01,  // Unsigned value is encoded using the Little Endian Base 128 (LEB128)
    DW_EH_PE_udata2  = 0x02,  // A 2 bytes unsigned value.
    DW_EH_PE_udata4  = 0x03,  // A 4 bytes unsigned value.
    DW_EH_PE_udata8  = 0x04,  // An 8 bytes unsigned value.
    DW_EH_PE_sleb128 = 0x09,  // Signed value is encoded using the Little Endian Base 128 (LEB128)
    DW_EH_PE_sdata2  = 0x0A,  // A 2 bytes signed value.
    DW_EH_PE_sdata4  = 0x0B,  // A 4 bytes signed value.
    DW_EH_PE_sdata8  = 0x0C,  // An 8 bytes signed value.

    DW_EH_PE_pcrel   = 0x10,  // Value is relative to the current program counter.
    DW_EH_PE_textrel = 0x20,  // Value is relative to the beginning of the .text section.
    DW_EH_PE_datarel = 0x30,  // Value is relative to the beginning of the .got or .eh_frame_hdr section.
    DW_EH_PE_funcrel = 0x40,  // Value is relative to the beginning of the function.
    DW_EH_PE_aligned = 0x50,  // Value is aligned to an address unit sized boundary.

    DW_EH_PE_indirect = 0x80,

    DW_EH_PE_omit    = 0xff   // Indicates that no value is present.
}

ubyte* get_uleb128(ubyte* addr, ref size_t res)
{
    res = 0;
    size_t bitsize = 0;

    // read as long as high bit is set
    while(*addr & 0x80)
    {
        res |= (*addr & 0x7f) << bitsize;
        bitsize += 7;
        addr += 1;
        if (bitsize >= size_t.sizeof*8)
            fatalerror("tried to read uleb128 that exceeded size of size_t");
    }
    // read last
    if (bitsize != 0 && *addr >= 1L << size_t.sizeof*8 - bitsize)
        fatalerror("tried to read uleb128 that exceeded size of size_t");
    res |= (*addr) << bitsize;

    return addr + 1;
}

ubyte* get_sleb128(ubyte* addr, ref ptrdiff_t res)
{
    res = 0;
    size_t bitsize = 0;

    // read as long as high bit is set
    while (*addr & 0x80)
    {
        res |= (*addr & 0x7f) << bitsize;
        bitsize += 7;
        addr += 1;
        if (bitsize >= size_t.sizeof*8)
            fatalerror("tried to read sleb128 that exceeded size of size_t");
    }
    // read last
    if (bitsize != 0 && *addr >= 1L << size_t.sizeof*8 - bitsize)
        fatalerror("tried to read sleb128 that exceeded size of size_t");
    res |= (*addr) << bitsize;

    // take care of sign
    if (bitsize < size_t.sizeof*8 && ((*addr) & 0x40))
        res |= cast(ptrdiff_t)(-1) ^ ((1 << (bitsize+7)) - 1);

    return addr + 1;
}

size_t get_size_of_encoded_value(ubyte encoding)
{
    if (encoding == _DW_EH_Format.DW_EH_PE_omit)
        return 0;

    switch (encoding & 0x07)
    {
        case _DW_EH_Format.DW_EH_PE_absptr:
            return size_t.sizeof;
        case _DW_EH_Format.DW_EH_PE_udata2:
            return 2;
        case _DW_EH_Format.DW_EH_PE_udata4:
            return 4;
        case _DW_EH_Format.DW_EH_PE_udata8:
            return 8;
        default:
            fatalerror("Unsupported DWARF Exception Header value format: unknown encoding");
    }
    assert(0);
}

ubyte* get_encoded_value(ubyte* addr, ref size_t res, ubyte encoding, void* context)
{
    ubyte *old_addr = addr;

    if (encoding == _DW_EH_Format.DW_EH_PE_aligned)
        goto Lerr;

    switch (encoding & 0x0f)
    {
      case _DW_EH_Format.DW_EH_PE_absptr:
          res = cast(size_t)*cast(ubyte**)addr;
          addr += size_t.sizeof;
          break;

      case _DW_EH_Format.DW_EH_PE_uleb128:
          addr = get_uleb128(addr, res);
          break;

      case _DW_EH_Format.DW_EH_PE_sleb128:
          ptrdiff_t r;
          addr = get_sleb128(addr, r);
          r = cast(size_t)res;
          break;

      case _DW_EH_Format.DW_EH_PE_udata2:
          res = *cast(ushort*)addr;
          addr += 2;
          break;
      case _DW_EH_Format.DW_EH_PE_udata4:
          res = *cast(uint*)addr;
          addr += 4;
          break;
      case _DW_EH_Format.DW_EH_PE_udata8:
          res = cast(size_t)*cast(ulong*)addr;
          addr += 8;
          break;

      case _DW_EH_Format.DW_EH_PE_sdata2:
          res = *cast(short*)addr;
          addr += 2;
          break;
      case _DW_EH_Format.DW_EH_PE_sdata4:
          res = *cast(int*)addr;
          addr += 4;
          break;
      case _DW_EH_Format.DW_EH_PE_sdata8:
          res = cast(size_t)*cast(long*)addr;
          addr += 8;
          break;

      default:
          goto Lerr;
    }

    switch (encoding & 0x70)
    {
        case _DW_EH_Format.DW_EH_PE_absptr:
            break;
        case _DW_EH_Format.DW_EH_PE_pcrel:
            res += cast(size_t)old_addr;
            break;
        case _DW_EH_Format.DW_EH_PE_funcrel:
            version(Win64) fatalerror("Not yet implemented."); else
            res += cast(size_t)_Unwind_GetRegionStart(context);
            break;
        case _DW_EH_Format.DW_EH_PE_textrel:
            version(Win64) fatalerror("Not yet implemented."); else
            res += cast(size_t)_Unwind_GetTextRelBase(context);
            break;
        case _DW_EH_Format.DW_EH_PE_datarel:
            version(Win64) fatalerror("Not yet implemented."); else
            res += cast(size_t)_Unwind_GetDataRelBase(context);
            break;
        default:
            goto Lerr;
    }

    if (encoding & _DW_EH_Format.DW_EH_PE_indirect)
        res = cast(size_t)*cast(void**)res;

    return addr;

Lerr:
    fatalerror("Unsupported DWARF Exception Header value format");
    return addr;
}

ptrdiff_t get_base_of_encoded_value(ubyte encoding, void* context)
{
    if (encoding == _DW_EH_Format.DW_EH_PE_omit)
        return 0;

    with (_DW_EH_Format) switch (encoding & 0x70) {
        case DW_EH_PE_absptr:
        case DW_EH_PE_pcrel:
        case DW_EH_PE_aligned:
            return 0;

      version(Win64) {} else
      {
        case DW_EH_PE_textrel:
            return _Unwind_GetTextRelBase (context);
        case DW_EH_PE_datarel:
            return _Unwind_GetDataRelBase (context);
        case DW_EH_PE_funcrel:
            return _Unwind_GetRegionStart (context);
      }

        default:
            fatalerror("Unsupported encoding type to get base from.");
            assert(0);
    }
}

void _d_getLanguageSpecificTables(ubyte* data, ref ubyte* callsite, ref ubyte* action, ref ubyte* classinfo_table, ref ubyte ciEncoding)
{
    if (data is null)
    {
        debug(EH_personality) printf("language specific data was null\n");
        callsite = null;
        action = null;
        classinfo_table = null;
        return;
    }
    debug(EH_personality) printf("  - LSDA: %p\n", data);

    //TODO: Do proper DWARF reading here
    if (*data++ != _DW_EH_Format.DW_EH_PE_omit)
        fatalerror("DWARF header has unexpected format 1");

    ciEncoding = *data++;
    if (ciEncoding == _DW_EH_Format.DW_EH_PE_omit)
        fatalerror("Language Specific Data does not contain Types Table");
    version (ARM) version (linux) {
        with (_DW_EH_Format) {
            ciEncoding = DW_EH_PE_pcrel | DW_EH_PE_indirect;
        }
    }

    size_t cioffset;
    data = get_uleb128(data, cioffset);
    classinfo_table = data + cioffset;

    if (*data++ != _DW_EH_Format.DW_EH_PE_udata4)
        fatalerror("DWARF header has unexpected format 2");
    size_t callsitelength;
    data = get_uleb128(data, callsitelength);
    action = data + callsitelength;

    callsite = data;

    debug(EH_personality) printf("  - callsite: %p, action: %p, classinfo_table: %p, ciEncoding: %d\n", callsite, action, classinfo_table, ciEncoding);
}



// -----------------------------
//    Stack of finally blocks
// -----------------------------

struct ActiveCleanupBlock {
    /// Link to the next active finally block.
    ActiveCleanupBlock* outerBlock;

    /// The exception that caused this cleanup block to be entered.
    Object dObject;

    /// The CFA (stack address, roughly) when this cleanup block was entered, as
    /// reported by libunwind.
    ///
    /// Used to determine when this pad is reached again when unwinding from
    /// somewhere within it. Note that this must somehow be related to the
    /// stack, not the instruction pointer, to properly support recursive
    /// chaining.
    ptrdiff_t cfaAddr;
}

/// Stack of active finally blocks (i.e. cleanup landing pads) that were entered
/// because of exception unwinding. Used for exception chaining.
///
/// Note that sometimes a landing pad is both a catch and a cleanup. This
/// happens for example when there is a try/finally nested inside a try/catch
/// in the same function, or has been inlined into one. Whether a catch will
/// actually execute (which terminates this strand of unwinding) or not is
/// determined by the program code and cannot be known inside the personality
/// routine. Thus, we always push such a block even before entering a catch(),
/// and have the user code call _d_eh_enter_catch() once the (possible) cleanup
/// part is done so we can pop it again. In theory, this could be optimized a
/// bit, because we only need to do it for landing pads which have this double
/// function, which we could possibly figure out form the DWARF tables. However,
/// since this makes generating the code for popping it non-trivial, this is not
/// currently done.
ActiveCleanupBlock* innermostCleanupBlock = null;

/// innermostCleanupBlock is per-stack, not per-thread, and as such needs to be
/// swapped out on fiber context switches.
extern(C) void* _d_eh_swapContext(void* newContext) nothrow
{
    auto old = innermostCleanupBlock;
    innermostCleanupBlock = cast(ActiveCleanupBlock*)newContext;
    return old;
}

/// During the search phase of unwinding, points to the currently active cleanup
/// block (i.e. somewhere in the innermostCleanupBlock linked list, but possibly
/// not at the beginning if the search for the catch block has already continued
/// past that).
///
/// Note that this and searchPhaseCurrentCleanupBlock can just be a single
/// variable because there can never be more than one search phase running per
/// thread.
ActiveCleanupBlock* searchPhaseCurrentCleanupBlock = null;

/// During the search phase, keeps track of the type of the dynamic type of the
/// currently thrown exception (might change according to exception chaining
/// rules).
ClassInfo searchPhaseClassInfo = null;

void pushCleanupBlockRecord(ptrdiff_t cfaAddr, Object dObject)
{
    auto acb = cast(ActiveCleanupBlock*)malloc(ActiveCleanupBlock.sizeof);
    if (!acb)
    {
        // TODO: Allocate some statically to avoid problem with unwinding out of
        // memory errors. A fairly small amount of memory should suffice for
        // most applications unless people want to unwind through very deeply
        // recursive code with many finally blocks.
        fatalerror("Could not allocate memory for exception chaining.");
    }
    acb.cfaAddr = cfaAddr;
    acb.dObject = dObject;
    acb.outerBlock = innermostCleanupBlock;
    innermostCleanupBlock = acb;

    // We need to be sure that an in-flight exception is kept alive while
    // executing a finally block. This is not automatically the case if the
    // finally block always throws, because the compiler then does not need to
    // keep a reference to the object extracted from the landing pad around as
    // there is no _d_eh_resume_unwind() call.
    GC.addRoot(cast(void*)dObject);
}

void popCleanupBlockRecord()
{
    if (!innermostCleanupBlock)
    {
        fatalerror("No cleanup block record found, should have been pushed " ~
            "before entering the finally block.");
    }
    // Remove the cleanup block we installed for this handler.
    auto acb = innermostCleanupBlock;
    GC.removeRoot(cast(void*)acb.dObject);
    innermostCleanupBlock = acb.outerBlock;
    free(acb);
}



/// This is the implementation of the personality function, which is called by
/// libunwind twice per frame (search phase, unwind phase).
///
/// It is responsible to figure out whether we need to stop unwinding because of
/// a catch block or if there is a finally block to execute by reading the DWARF
/// EH tables.
extern(C) auto eh_personality_common(NativeContext)(ref NativeContext nativeContext)
{
    //
    // First, we need to find the Language-Specific Data table for this frame
    // and extract our information tables (the "language-specific" part is a bit
    // of a misnomer, in reality this is generated by LLVM).
    //
    // The callsite and action tables do not contain static-length data and will
    // be parsed as needed.
    //

    ubyte* callsite_table;
    ubyte* action_table;
    ubyte* classinfo_table; // points past the end of the table
    ubyte classinfo_table_encoding;
    ubyte* data = nativeContext.getLanguageSpecificData();
    _d_getLanguageSpecificTables(data, callsite_table, action_table, classinfo_table, classinfo_table_encoding);

    if (!callsite_table)
        return nativeContext.continueUnwind();

    //
    // Now, we need to figure out if the address of the current instruction
    // in this frame corresponds to a block which has an associated landing pad.
    //

    // The instruction pointer (ip) will point to the next instruction after
    // whatever made execution leave this frame, so substract 1 for the range
    // comparison below.
    immutable ptrdiff_t ip = nativeContext.getIP() - 1;

    // The table entries are all relative to the start address of the region.
    immutable ptrdiff_t region_start = nativeContext.getRegionStart();

    // The address of the landing pad to jump to (null if no match).
    ptrdiff_t landingPadAddr;

    // The offset in the action table corresponding to the first action for this
    // landing pad (will be zero if there are none).
    size_t actionTableStartOffset;

    ubyte* callsite_walker = callsite_table;
    while (true)
    {
        // if we've gone through the list and found nothing...
        if (callsite_walker >= action_table)
            return nativeContext.continueUnwind();

        immutable block_start_offset = *cast(uint*)callsite_walker;
        immutable block_size = *(cast(uint*)callsite_walker + 1);
        landingPadAddr = *(cast(uint*)callsite_walker + 2);
        callsite_walker = get_uleb128(callsite_walker + 3 * uint.sizeof, actionTableStartOffset);

        debug(EH_personality_verbose)
        {
            printf("  - ip=%llx %d %d %llx\n", ip, block_start_offset,
                block_size, landingPadAddr);
        }

        // since the list is sorted, as soon as we're past the ip
        // there's no handler to be found
        if (ip < region_start + block_start_offset)
            return nativeContext.continueUnwind();

        // if we've found our block, exit
        if (ip < region_start + block_start_offset + block_size)
            break;
    }

    debug(EH_personality)
    {
        printf("  - Found correct landing pad and actionTableStartOffset %d\n",
            actionTableStartOffset);
    }

    // There is no landing pad for this part of the frame, continue with the next level.
    if (!landingPadAddr)
        return nativeContext.continueUnwind();

    // We have a landing pad, adjust by region start address.
    landingPadAddr += region_start;

    immutable bool isSearchPhase = nativeContext.isSearchPhase();

    //
    // We have at least a finally landing pad in this scope. First, check if we
    // have arrived at the scope a previous exception was thrown in. In this
    // case, we need to chain exception_struct.exception_object to it or replace
    // it with the former.
    //
    immutable ptrdiff_t currentCfaAddr = nativeContext.getCfaAddress();
    ref ActiveCleanupBlock* acb()
    {
        return isSearchPhase ? searchPhaseCurrentCleanupBlock : innermostCleanupBlock;
    }

    while (acb)
    {
        debug(EH_personality)
        {
            printf("  - Current CFA: %p, Previous CFA: %p\n",
                currentCfaAddr, acb.cfaAddr);
        }

        // If the next active cleanup block is somewhere further up the stack,
        // there is nothing to do/check.
        // Note: assumes the stack grows downwards.
        if (currentCfaAddr < acb.cfaAddr)
            break;

        Object thrownDObject = nativeContext.getThrownObject();
        auto currentClassInfo = isSearchPhase ? searchPhaseClassInfo : thrownDObject.classinfo;
        if (_d_isbaseof(currentClassInfo, Error.classinfo) && !cast(Error)acb.dObject)
        {
            // The currently unwound Throwable is an Error but the previous one
            // is not, so replace the latter with the former.
            debug(EH_personality)
            {
                printf(" ++ Replacing %p (%s) by %p (%s)\n",
                    acb.dObject,
                    acb.dObject.classinfo.name.ptr,
                    thrownDObject,
                    thrownDObject.classinfo.name.ptr);
            }

            if (!isSearchPhase)
            {
                (cast(Error)thrownDObject).bypassedException =
                    cast(Throwable)acb.dObject;
            }
        }
        else
        {
            // We are just unwinding an Exception or there was already an Error,
            // so append this Throwable to the end of the previous chain.
            if (isSearchPhase)
            {
                debug(EH_personality)
                {
                    printf(" ++ Setting up classinfo to chain %s to %p (%s, classinfo at %p)\n",
                        searchPhaseClassInfo.name.ptr, acb.dObject,
                        acb.dObject.classinfo.name.ptr, acb.dObject.classinfo);
                }
                searchPhaseClassInfo = acb.dObject.classinfo;
            }
            else
            {
                auto lastChainElem = cast(Throwable)acb.dObject;
                while (lastChainElem.next)
                {
                    lastChainElem = lastChainElem.next;
                }

                auto thisThrowable = cast(Throwable)thrownDObject;
                if (lastChainElem is thisThrowable)
                {
                    // We would need to chain an exception to itself. This can
                    // happen if somebody throws the same exception object twice.
                    // It is questionable whether this is supposed to work in the
                    // first place, but core.demangle does it when generating the
                    // backtrace for its internal exceptions during demangling a
                    // symbol as part of the default trace handler (it uses the
                    // .init value for the exception class instead of allocating
                    // new instances).
                    debug(EH_personality)
                    {
                        printf(" ++ Not chaining %p (%s) to itself\n",
                            thisThrowable, thisThrowable.classinfo.name.ptr);
                    }
                }
                else
                {
                    debug(EH_personality)
                    {
                        printf(" ++ Chaining %p (%s) to %p (%s)\n",
                            thisThrowable, thisThrowable.classinfo.name.ptr,
                            lastChainElem, lastChainElem.classinfo.name.ptr);
                    }
                    lastChainElem.next = thisThrowable;
                }

                nativeContext.overrideThrownObject(acb.dObject);
            }
        }

        // In both cases, we've executed one level of chaining.
        auto outer = acb.outerBlock;
        if (!isSearchPhase)
        {
            GC.removeRoot(cast(void*)acb.dObject);
            free(acb);
        }
        acb = outer;
    }

    //
    // Exception chaining is now done. Let's figure out what we have to do in
    // this frame.
    //

    // If there are no actions, this is a cleanup landing pad.
    if (!actionTableStartOffset)
    {
        return nativeContext.installFinallyContext(landingPadAddr);
    }

    // We have at least some attached actions. Figure out whether any of them
    // match the type of the current exception.
    immutable ci_size = get_size_of_encoded_value(classinfo_table_encoding);
    debug(EH_personality) printf("  - ci_size: %td, ci_encoding: %d\n", ci_size, classinfo_table_encoding);

    ubyte* action_walker = action_table + actionTableStartOffset - 1;
    while (true)
    {
        ptrdiff_t ti_offset;
        action_walker = get_sleb128(action_walker, ti_offset);
        debug(EH_personality) printf("  - ti_offset: %tx\n", ti_offset);

        // it is intentional that we not modify action_walker here
        // next_action_offset is from current action_walker position
        ptrdiff_t next_action_offset;
        get_sleb128(action_walker, next_action_offset);

        // negative are 'filters' which we don't use
        if (!(ti_offset >= 0))
            fatalerror("Filter actions are unsupported");

        // zero means cleanup, which we require to be the last action
        if (ti_offset == 0)
        {
            if (!(next_action_offset == 0))
                fatalerror("Cleanup action must be last in chain");
            return nativeContext.installFinallyContext(landingPadAddr);
        }

        if (!nativeContext.skipCatchComparison())
        {
            ClassInfo catchClassInfo = nativeContext.getCatchClassInfo(
                classinfo_table - ti_offset * ci_size, classinfo_table_encoding);
            ClassInfo exceptionClassInfo = isSearchPhase ?
                searchPhaseClassInfo : nativeContext.getThrownObject().classinfo;

            debug(EH_personality)
            {
                printf("  - Comparing catch %s to exception %s\n",
                    catchClassInfo.name.ptr, exceptionClassInfo.name.ptr);
            }
            if (_d_isbaseof(exceptionClassInfo, catchClassInfo))
            {
                return nativeContext.installCatchContext(ti_offset, landingPadAddr);
            }
        }

        debug(EH_personality) printf("  - Type mismatch, next action offset: %tx\n", next_action_offset);

        if (next_action_offset == 0)
            return nativeContext.continueUnwind();
        action_walker += next_action_offset;
    }
}
