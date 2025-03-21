override LINK_SHARED := $(SHARED)

ifdef IN_LDC
# need OS for the conditions below
include ../../../../dmd/osmodel.mak
endif

ifneq (,$(LINK_SHARED))
    ifndef IN_LDC
        # TODO: enable tests on Windows
        ifeq (windows,$(OS))
            TESTS:=link linkD linkDR loadDR
        else
            TESTS:=link load linkD linkDR loadDR host finalize dynamiccast \
                   link_linkdep load_linkdep link_loaddep load_loaddep load_13414
        endif
    else
        # LDC: enable ~all tests on Windows too
        TESTS:=link load linkD linkDR loadDR host finalize dynamiccast \
               link_linkdep link_loaddep load_loaddep load_13414
        # FIXME: `load_linkdep` needs a non-dummy `getDependencies()` in rt.sections_elf_shared,
        #        not implemented yet on Darwin and Windows
        ifneq (osx,$(OS))
            ifneq (windows,$(OS))
                TESTS+=load_linkdep
            endif
        endif

        ifeq (windows,$(OS))
            TESTS += dll_gc_proxy_teardown dll_gc_proxy_teardown_nounload
        endif
    endif
endif

# there are extra tests for Windows, not requiring a druntime DLL
ifeq (windows,$(OS))
    TESTS+=loadlibwin dllrefcount dllgc dynamiccast
endif

include ../common.mak

abs_root := $(abspath $(ROOT))
# on posix we link directly to the .so
# on windows you have to link the .lib which references the .dll
for_linking := $(if $(filter windows,$(OS)),.lib,$(DOTDLL))

ifeq (windows,$(OS)) # extra tests on Windows

ifndef IN_LDC
ifeq ($(SHARED),1)
# dmd -shared does not (yet) imply -visibility=public
$(ROOT)/%$(DOTDLL): private extra_dflags += -visibility=public

extra_dflags += -version=SharedRuntime
PATH := $(dir $(DRUNTIMESO));$(PATH)
endif
endif

# LDC: this test is designed for .exe & .dll with separate druntimes
$(ROOT)/dllgc$(DOTEXE) $(ROOT)/dllgc$(DOTDLL): private extra_dflags += -link-defaultlib-shared=false -dllimport=none

# LDC addition: test teardown with separate druntimes, with the DLL using the .exe GC
dgpt = dll_gc_proxy_teardown dll_gc_proxy_teardown_nounload
$(dgpt:%=$(OBJDIR)/%$(DOTDLL)): private extra_ldflags.d += -L/EXPORT:gc_setProxy -L/EXPORT:gc_clrProxy
$(dgpt:%=$(OBJDIR)/%$(DOTDLL)): private extra_dflags += -version=DLL
$(dgpt:%=$(OBJDIR)/%$(DOTEXE)) $(dgpt:%=$(OBJDIR)/%$(DOTDLL)): private extra_dflags += -link-defaultlib-shared=false -dllimport=none
$(dgpt:%=$(OBJDIR)/%.done): $(OBJDIR)/%.done: $(OBJDIR)/%$(DOTDLL)

$(OBJDIR)/dll_gc_proxy_teardown_nounload%: private extra_dflags += -version=NoUnload
$(OBJDIR)/dll_gc_proxy_teardown_nounload$(DOTDLL): dll_gc_proxy_teardown.d
	$(LINK.d) -shared $< $(extra_sources) $(extra_ldlibs.d) $(LDLIBS.d) $(OUTPUT_OPTION.d)
$(OBJDIR)/dll_gc_proxy_teardown_nounload$(DOTEXE): dll_gc_proxy_teardown.d
	$(LINK.d) $< $(extra_sources) $(extra_ldlibs.d) $(LDLIBS.d) $(OUTPUT_OPTION.d)

$(ROOT)/dllgc.done: $(ROOT)/dllgc$(DOTDLL)
$(ROOT)/dllgc$(DOTDLL): extra_dflags += -version=DLL
endif # Windows

$(ROOT)/dynamiccast.done: $(ROOT)/%.done: $(ROOT)/%$(DOTEXE) $(ROOT)/%$(DOTDLL)
	@echo Testing $*
	$(RM) $(ROOT)/dynamiccast_end{bar,main}
	$(TIMELIMIT)$<
	test -f $(ROOT)/dynamiccast_endbar
	test -f $(ROOT)/dynamiccast_endmain
	@touch $@
$(ROOT)/dynamiccast$(DOTEXE): private extra_ldlibs.d += $(LINKDL)
$(ROOT)/dynamiccast$(DOTDLL): private extra_dflags += -version=DLL

# Avoid a race condition that I sometimes hit with make -j8.
# Maybe temporary file collisions when invoking the linker?
$(OBJDIR)/dynamiccast$(DOTEXE): $(OBJDIR)/dynamiccast$(DOTDLL)

$(OBJDIR)/link$(DOTEXE): $(OBJDIR)/lib$(for_linking) lib.d
$(OBJDIR)/link$(DOTEXE): private extra_ldlibs.d += $(abs_root)/lib$(for_linking)

$(ROOT)/liblinkdep$(DOTDLL): $(OBJDIR)/lib$(for_linking) lib.d
$(ROOT)/liblinkdep$(DOTDLL): private extra_ldlibs.d += $(abs_root)/lib$(for_linking)

$(ROOT)/libloaddep$(DOTDLL): private extra_ldlibs.d += $(LINKDL)

$(ROOT)/link_linkdep$(DOTEXE): $(ROOT)/liblinkdep$(for_linking) liblinkdep.d
$(ROOT)/link_linkdep$(DOTEXE): private extra_ldlibs.d += $(abs_root)/liblinkdep$(for_linking)

# dlopens lib.so through libloaddep
$(ROOT)/link_loaddep.done: $(ROOT)/lib$(DOTDLL)
$(ROOT)/link_loaddep$(DOTEXE): $(ROOT)/libloaddep$(for_linking) libloaddep.d utils.di
$(ROOT)/link_loaddep$(DOTEXE): private extra_ldlibs.d += $(abs_root)/libloaddep$(for_linking)

# dlopens liblinkdep.so
$(ROOT)/load_linkdep.done: $(ROOT)/liblinkdep$(DOTDLL)
$(ROOT)/load_linkdep$(DOTEXE): utils.di
$(ROOT)/load_linkdep$(DOTEXE): private extra_ldlibs.d += $(LINKDL)

# dlopens libloaddep.so and runs code that will dlopen lib.so
$(ROOT)/load_loaddep.done: $(ROOT)/lib$(DOTDLL) $(ROOT)/libloaddep$(DOTDLL)
$(ROOT)/load_loaddep$(DOTEXE): utils.di
$(ROOT)/load_loaddep$(DOTEXE): private extra_ldlibs.d += $(LINKDL)

$(ROOT)/load.done: $(ROOT)/lib$(DOTDLL)
$(ROOT)/load$(DOTEXE): utils.di
$(ROOT)/load$(DOTEXE): private extra_ldlibs.d += $(LINKDL)

$(ROOT)/finalize.done: $(ROOT)/lib$(DOTDLL)
$(ROOT)/finalize$(DOTEXE): utils.di
$(ROOT)/finalize$(DOTEXE): private extra_ldlibs.d += $(LINKDL)

$(ROOT)/load_13414.done: $(ROOT)/lib_13414$(DOTDLL)
$(ROOT)/load_13414$(DOTEXE): utils.di
$(ROOT)/load_13414$(DOTEXE): private extra_ldlibs.d += $(LINKDL)

ifeq (windows,$(OS))
    CC := cl
    OUTPUT_FLAG := /Fe
    # we additionally specify the .obj output path (/Fo) to prevent collisions
    extra_cflags += /Fo$(OBJDIR)/
endif

# $(LINKDL) == -L-ldl => $(ldl) == -ldl
ldl := $(LINKDL:-L%=%)

$(ROOT)/linkD$(DOTEXE): $(ROOT)/lib$(for_linking)
$(ROOT)/linkD$(DOTEXE): private extra_ldlibs += $(abs_root)/lib$(for_linking)

$(ROOT)/linkDR.done: $(ROOT)/lib$(DOTDLL)
$(ROOT)/linkDR$(DOTEXE): $(DRUNTIME_DEP) utils.h
$(ROOT)/linkDR$(DOTEXE): private extra_ldlibs += $(ldl) $(druntime_for_linking)
ifneq ($(OS),windows)
$(ROOT)/linkDR$(DOTEXE): private extra_ldflags += -Wl,-rpath,$(druntimeso_dir)
endif

$(ROOT)/loadDR.done: $(ROOT)/lib$(DOTDLL) $(DRUNTIMESO)
$(ROOT)/loadDR.done: private run_args = $(DRUNTIMESO)
$(ROOT)/loadDR$(DOTEXE): utils.h
$(ROOT)/loadDR$(DOTEXE): private extra_ldlibs += $(ldl)
ifeq ($(IS_MUSL),1)
$(ROOT)/loadDR$(DOTEXE): private extra_ldflags += -Wl,-rpath,$(druntimeso_dir)
endif

$(ROOT)/host.done: $(DRUNTIMESO) $(ROOT)/plugin1$(DOTDLL) $(ROOT)/plugin2$(DOTDLL)
$(ROOT)/host.done: private run_args = $(DRUNTIMESO)
$(ROOT)/host$(DOTEXE): utils.h
$(ROOT)/host$(DOTEXE): private extra_ldlibs += $(ldl)

$(ROOT)/plugin1$(DOTDLL) $(ROOT)/plugin2$(DOTDLL): $(ROOT)/plugin$(DOTDLL)
	cp $< $@

########## default rule for building a shared library ##########

$(ROOT)/%$(DOTDLL): %.d $(DMD_DEP) $(DRUNTIME_DEP)
	$(LINK.d) -shared $< $(extra_ldlibs.d) $(LDLIBS.d) $(OUTPUT_OPTION.d)

$(ROOT)/%$(for_linking): $(ROOT)/%$(DOTDLL) ;
