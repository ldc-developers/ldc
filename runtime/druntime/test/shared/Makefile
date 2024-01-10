# LDC: druntime DLL supported on Windows
LINK_SHARED:=1

include ../common.mak # affected by LINK_SHARED!

.PHONY: all clean

ifeq (windows,$(OS))

# LDC: required for executables to implicitly dllimport DLL data symbols
DFLAGS+=-dllimport=all

TESTS:=loadlibwin dllrefcount dllgc dynamiccast
ifeq ($(SHARED),1)
DFLAGS+=-version=SharedRuntime
endif

$(ROOT)/%.done: $(ROOT)/%$(DOTEXE)
	@echo Testing $*
	$(QUIET)$(TIMELIMIT)$< $(RUN_ARGS)
	@touch $@

$(ROOT)/dynamiccast.done: $(ROOT)/%.done: $(ROOT)/%$(DOTEXE)
	@echo Testing $*
	$(QUIET)rm -f $(ROOT)/dynamiccast_end{bar,main}
	$(QUIET)$(TIMELIMIT)$< $(RUN_ARGS)
	$(QUIET)test -f $(ROOT)/dynamiccast_endbar
	$(QUIET)test -f $(ROOT)/dynamiccast_endmain
	@touch $@

$(ROOT)/dllrefcount$(DOTEXE): $(SRC)/dllrefcount.d
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $<

$(ROOT)/loadlibwin$(DOTEXE): $(SRC)/loadlibwin.d
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $<

# LDC: this test is designed for .exe & .dll with separate druntimes
$(ROOT)/dllgc$(DOTEXE): DFLAGS+=-link-defaultlib-shared=false -dllimport=none

$(ROOT)/dllgc$(DOTEXE): $(SRC)/dllgc.d
	$(QUIET)$(DMD) $(DFLAGS) -version=DLL -shared -of$(ROOT)/dllgc$(DOTDLL) $<
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $<

$(ROOT)/dynamiccast$(DOTEXE): $(SRC)/dynamiccast.d
	$(QUIET)$(DMD) $(DFLAGS) -version=DLL -shared -of$(ROOT)/dynamiccast$(DOTDLL) $<
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $<

# LDC addition: test teardown with separate druntimes, with the DLL using the .exe GC
dll_gc_proxy_teardown: DFLAGS+=-link-defaultlib-shared=false -dllimport=none
dll_gc_proxy_teardown: $(SRC)/dll_gc_proxy_teardown.d
	$(QUIET)$(DMD) $(DFLAGS) -shared -L/EXPORT:gc_setProxy -L/EXPORT:gc_clrProxy -version=DLL -of$(ROOT)/dll_gc_proxy_teardown$(DOTDLL) $<
	$(QUIET)$(DMD) $(DFLAGS) -shared -L/EXPORT:gc_setProxy -L/EXPORT:gc_clrProxy -version=DLL -version=NoUnload -of$(ROOT)/dll_gc_proxy_teardown_nounload$(DOTDLL) $<
	$(QUIET)$(DMD) $(DFLAGS) -of$(ROOT)/load_dll_gc_proxy_teardown$(DOTEXE) $<
	$(QUIET)$(DMD) $(DFLAGS) -version=NoUnload -of$(ROOT)/load_dll_gc_proxy_teardown_nounload$(DOTEXE) $<
	$(QUIET)$(ROOT)/load_dll_gc_proxy_teardown$(DOTEXE)
	$(QUIET)$(ROOT)/load_dll_gc_proxy_teardown_nounload$(DOTEXE)

# LDC: include Posix tests too
endif # Windows

TESTS+=link load linkD linkDR loadDR dynamiccast
TESTS+=link_linkdep link_loaddep load_loaddep load_13414
# LDC: disable 3 tests on Mac, 1 on Windows
ifneq ($(OS),osx)
	# * `host` loads two modules with the same name, which is currently disallowed
	#   by the (potentially overly eager) module collision detection on OS X.
	# * `finalize` fails starting with macOS 10.13, as .dylibs with TLS can't be
	#   unloaded anymore (https://github.com/ldc-developers/ldc/issues/3002).
	# * FIXME: `load_linkdep`
	#   it might fail because of unimplemented `getDependencies()` in rt.sections_elf_shared
	ifneq (,$(findstring win,$(OS)))
		# LDC FIXME: disable `load_linkdep` on Windows - needs `getDependencies()`
		TESTS+=host finalize
	else
		TESTS+=host finalize load_linkdep
	endif
endif

EXPORT_DYNAMIC=$(if $(findstring $(OS),linux freebsd dragonflybsd),-L--export-dynamic,)
NO_AS_NEEDED=$(if $(findstring $(OS),linux freebsd dragonflybsd),-L--no-as-needed,)

all: $(addprefix $(ROOT)/,$(addsuffix .done,$(TESTS))) $(if $(findstring $(OS),windows),dll_gc_proxy_teardown,)

$(ROOT)/loadDR.done $(ROOT)/host.done: RUN_ARGS:=$(DRUNTIMESO)

$(ROOT)/dynamiccast.done: CLEANUP:=rm $(ROOT)/dynamiccast_endmain$(DOTEXE) $(ROOT)/dynamiccast_endbar$(DOTEXE)

$(ROOT)/%.done: $(ROOT)/%$(DOTEXE)
	@echo Testing $*
	$(QUIET)$(TIMELIMIT)$< $(RUN_ARGS)
	$(CLEANUP)
	@touch $@

$(ROOT)/link$(DOTEXE): $(SRC)/link.d $(ROOT)/lib$(DOTDLL) $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< -L$(ROOT)/lib$(DOTDLL)

$(ROOT)/link_linkdep$(DOTEXE): $(SRC)/link_linkdep.d $(ROOT)/lib$(DOTDLL) $(ROOT)/liblinkdep$(DOTDLL) $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< $(LINKFLAGS) -L$(ROOT)/liblinkdep$(DOTDLL) -L$(ROOT)/lib$(DOTDLL)

$(ROOT)/load_linkdep$(DOTEXE): $(SRC)/load_linkdep.d $(ROOT)/lib$(DOTDLL) $(ROOT)/liblinkdep$(DOTDLL) $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< $(LINKFLAGS) $(LINKDL)

$(ROOT)/link_loaddep$(DOTEXE): $(SRC)/link_loaddep.d $(ROOT)/lib$(DOTDLL) $(ROOT)/libloaddep$(DOTDLL) $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< $(LINKFLAGS) -L$(ROOT)/libloaddep$(DOTDLL)

$(ROOT)/load_loaddep$(DOTEXE): $(SRC)/load_loaddep.d $(ROOT)/lib$(DOTDLL) $(ROOT)/libloaddep$(DOTDLL) $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< $(LINKFLAGS) $(LINKDL)

$(ROOT)/load$(DOTEXE) $(ROOT)/finalize$(DOTEXE): $(ROOT)/%$(DOTEXE): $(SRC)/%.d $(ROOT)/lib$(DOTDLL) $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< $(LINKDL)

$(ROOT)/load_13414$(DOTEXE): $(ROOT)/%$(DOTEXE): $(SRC)/%.d $(ROOT)/lib_13414$(DOTDLL) $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< $(LINKDL)

$(ROOT)/dynamiccast$(DOTEXE): $(SRC)/dynamiccast.d $(ROOT)/dynamiccast$(DOTDLL) $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $(SRC)/dynamiccast.d $(LINKDL)

$(ROOT)/dynamiccast$(DOTDLL): $(SRC)/dynamiccast.d $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< -version=DLL -fPIC -shared $(LINKDL)

# TODO: cl.exe flags
$(ROOT)/linkD$(DOTEXE): $(SRC)/linkD.c $(ROOT)/lib$(DOTDLL) $(DRUNTIMESO)
	$(QUIET)$(CC) $(CFLAGS) -o $@ $< $(ROOT)/lib$(DOTDLL) $(LDL) -pthread

$(ROOT)/linkDR$(DOTEXE): $(SRC)/linkDR.c $(ROOT)/lib$(DOTDLL) $(DRUNTIMESO)
	$(QUIET)$(CC) $(CFLAGS) -o $@ $< $(DRUNTIMESO) $(LDL) -pthread

$(ROOT)/loadDR$(DOTEXE): $(SRC)/loadDR.c $(ROOT)/lib$(DOTDLL) $(DRUNTIMESO)
	$(QUIET)$(CC) $(CFLAGS) -o $@ $< $(LDL) -pthread

$(ROOT)/host$(DOTEXE): $(SRC)/host.c $(ROOT)/plugin1$(DOTDLL) $(ROOT)/plugin2$(DOTDLL)
	$(QUIET)$(CC) $(CFLAGS) -o $@ $< $(LDL) -pthread

$(ROOT)/liblinkdep$(DOTDLL): $(ROOT)/lib$(DOTDLL)
$(ROOT)/liblinkdep$(DOTDLL): DFLAGS+=-L$(ROOT)/lib$(DOTDLL)

$(ROOT)/plugin1$(DOTDLL) $(ROOT)/plugin2$(DOTDLL): $(SRC)/plugin.d $(DRUNTIMESO)
	$(QUIET)$(DMD) -fPIC -shared $(DFLAGS) -of$@ $<

$(ROOT)/%$(DOTDLL): $(SRC)/%.d $(DRUNTIMESO)
	$(QUIET)$(DMD) -fPIC -shared $(DFLAGS) -of$@ $< $(LINKDL)

clean:
	rm -rf $(GENERATED)
