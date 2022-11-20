LINK_SHARED:=1

include ../common.mak

TESTS:=link load linkD linkDR loadDR dynamiccast
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

.PHONY: all clean

# LDC: add Windows tests
ifeq (,$(findstring win,$(OS)))

all: $(addprefix $(ROOT)/,$(addsuffix .done,$(TESTS)))
LIB_EXT:=so
DLL_EXT:=so

$(ROOT)/loadDR.done $(ROOT)/host.done: RUN_ARGS:=$(DRUNTIMESO)

else
# Windows

# required for executables to implicitly dllimport DLL data symbols
DFLAGS+=-dllimport=all

all: $(addprefix $(ROOT)/,$(addsuffix .done,$(TESTS))) loadlibwin dllrefcount dllgc dll_gc_proxy_teardown
LIB_EXT:=lib
DLL_EXT:=dll
CC:=cl.exe

$(ROOT)/loadDR.done $(ROOT)/host.done: RUN_ARGS:=$(subst .lib,.dll,$(DRUNTIMESO))

dllrefcount:
	$(DMD) $(DFLAGS) src/dllrefcount.d -of$(ROOT)/dllrefcount.exe
	$(ROOT)/dllrefcount.exe

loadlibwin:
	$(DMD) $(DFLAGS) src/loadlibwin.d -of$(ROOT)/loadlibwin.exe
	$(ROOT)/loadlibwin.exe

dllgc:
	$(DMD) $(DFLAGS) -version=DLL -shared -of$(ROOT)/dllgc.dll src/dllgc.d
	$(DMD) $(DFLAGS) -of$(ROOT)/loaddllgc.exe src/dllgc.d
	$(ROOT)/loaddllgc.exe

# LDC: this test is designed for .exe & .dll with separate druntimes
dllgc: DFLAGS+=-link-defaultlib-shared=false -dllimport=none

# LDC addition: test teardown with separate druntimes, with the DLL using the .exe GC
dll_gc_proxy_teardown: DFLAGS+=-link-defaultlib-shared=false -dllimport=none
dll_gc_proxy_teardown:
	$(DMD) $(DFLAGS) -shared -L/EXPORT:gc_setProxy -L/EXPORT:gc_clrProxy -version=DLL -of$(ROOT)/dll_gc_proxy_teardown.dll src/dll_gc_proxy_teardown.d
	$(DMD) $(DFLAGS) -shared -L/EXPORT:gc_setProxy -L/EXPORT:gc_clrProxy -version=DLL -version=NoUnload -of$(ROOT)/dll_gc_proxy_teardown_nounload.dll src/dll_gc_proxy_teardown.d
	$(DMD) $(DFLAGS) -of$(ROOT)/load_dll_gc_proxy_teardown.exe src/dll_gc_proxy_teardown.d
	$(DMD) $(DFLAGS) -version=NoUnload -of$(ROOT)/load_dll_gc_proxy_teardown_nounload.exe src/dll_gc_proxy_teardown.d
	$(ROOT)/load_dll_gc_proxy_teardown.exe
	$(ROOT)/load_dll_gc_proxy_teardown_nounload.exe

# end Windows
endif

$(ROOT)/dynamiccast.done: CLEANUP:=rm $(ROOT)/dynamiccast_endmain $(ROOT)/dynamiccast_endbar

$(ROOT)/%.done: $(ROOT)/%
	@echo Testing $*
	$(QUIET)$(TIMELIMIT)$< $(RUN_ARGS)
	$(CLEANUP)
	@touch $@

$(ROOT)/link: $(SRC)/link.d $(ROOT)/lib.$(DLL_EXT) $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< -L$(ROOT)/lib.$(LIB_EXT)

$(ROOT)/link_linkdep: $(SRC)/link_linkdep.d $(ROOT)/lib.$(DLL_EXT) $(ROOT)/liblinkdep.$(DLL_EXT) $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< $(LINKFLAGS) -L$(ROOT)/liblinkdep.$(LIB_EXT) -L$(ROOT)/lib.$(LIB_EXT)

$(ROOT)/load_linkdep: $(SRC)/load_linkdep.d $(ROOT)/lib.$(DLL_EXT) $(ROOT)/liblinkdep.$(DLL_EXT) $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< $(LINKFLAGS) $(LINKDL)

$(ROOT)/link_loaddep: $(SRC)/link_loaddep.d $(ROOT)/lib.$(DLL_EXT) $(ROOT)/libloaddep.$(DLL_EXT) $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< $(LINKFLAGS) -L$(ROOT)/libloaddep.$(LIB_EXT)

$(ROOT)/load_loaddep: $(SRC)/load_loaddep.d $(ROOT)/lib.$(DLL_EXT) $(ROOT)/libloaddep.$(DLL_EXT) $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< $(LINKFLAGS) $(LINKDL)

$(ROOT)/load $(ROOT)/finalize: $(ROOT)/%: $(SRC)/%.d $(ROOT)/lib.$(DLL_EXT) $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< $(LINKDL)

$(ROOT)/load_13414: $(ROOT)/%: $(SRC)/%.d $(ROOT)/lib_13414.$(DLL_EXT) $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< $(LINKDL)

$(ROOT)/dynamiccast: $(SRC)/dynamiccast.d $(ROOT)/dynamiccast.$(DLL_EXT) $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $(SRC)/dynamiccast.d $(LINKDL)

$(ROOT)/dynamiccast.$(DLL_EXT): $(SRC)/dynamiccast.d $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< -version=DLL -fPIC -shared $(LINKDL)

ifeq (,$(findstring win,$(OS)))

$(ROOT)/linkD: $(SRC)/linkD.c $(ROOT)/lib.$(DLL_EXT) $(DRUNTIMESO)
	$(QUIET)$(CC) $(CFLAGS) -o $@ $< $(ROOT)/lib.$(DLL_EXT) $(LDL) -pthread

$(ROOT)/linkDR: $(SRC)/linkDR.c $(ROOT)/lib.$(DLL_EXT) $(DRUNTIMESO)
	$(QUIET)$(CC) $(CFLAGS) -o $@ $< $(DRUNTIMESO) $(LDL) -pthread

$(ROOT)/loadDR: $(SRC)/loadDR.c $(ROOT)/lib.$(DLL_EXT) $(DRUNTIMESO)
	$(QUIET)$(CC) $(CFLAGS) -o $@ $< $(LDL) -pthread

$(ROOT)/host: $(SRC)/host.c $(ROOT)/plugin1.$(DLL_EXT) $(ROOT)/plugin2.$(DLL_EXT)
	$(QUIET)$(CC) $(CFLAGS) -o $@ $< $(LDL) -pthread

else
# Windows

$(ROOT)/linkD: $(SRC)/linkD.c $(ROOT)/lib.$(DLL_EXT) $(DRUNTIMESO)
	$(QUIET)$(CC) $(CFLAGS) /Fo$@.obj /Fe$@.exe $< $(ROOT)/lib.$(LIB_EXT)

$(ROOT)/linkDR: $(SRC)/linkDR.c $(ROOT)/lib.$(DLL_EXT) $(DRUNTIMESO)
	$(QUIET)$(CC) $(CFLAGS) /Fo$@.obj /Fe$@.exe $< $(DRUNTIMESO)

$(ROOT)/loadDR: $(SRC)/loadDR.c $(ROOT)/lib.$(DLL_EXT) $(DRUNTIMESO)
	$(QUIET)$(CC) $(CFLAGS) /Fo$@.obj /Fe$@.exe $<

$(ROOT)/host: $(SRC)/host.c $(ROOT)/plugin1.$(DLL_EXT) $(ROOT)/plugin2.$(DLL_EXT)
	$(QUIET)$(CC) $(CFLAGS) /Fo$@.obj /Fe$@.exe $<

# end Windows
endif

$(ROOT)/liblinkdep.$(DLL_EXT): $(ROOT)/lib.$(DLL_EXT)
$(ROOT)/liblinkdep.$(DLL_EXT): DFLAGS+=-L$(ROOT)/lib.$(LIB_EXT)

$(ROOT)/plugin1.$(DLL_EXT) $(ROOT)/plugin2.$(DLL_EXT): $(SRC)/plugin.d $(DRUNTIMESO)
	$(QUIET)$(DMD) -fPIC -shared $(DFLAGS) -of$@ $<

$(ROOT)/%.$(DLL_EXT): $(SRC)/%.d $(DRUNTIMESO)
	$(QUIET)$(DMD) -fPIC -shared $(DFLAGS) -of$@ $< $(LINKDL)

clean:
	rm -rf $(GENERATED)
