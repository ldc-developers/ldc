LINK_SHARED:=1

include ../common.mak

TESTS:=link load linkD linkDR loadDR
# LDC: disable 3 tests on Mac
ifneq ($(OS),osx)
	# * `finalize` fails starting with macOS 10.13, as .dylibs with TLS can't be
	#   unloaded anymore (https://github.com/ldc-developers/ldc/issues/3002).
	# * `host` loads two modules with the same name, which is currently disallowed
	#   by the (potentially overly eager) module collision detection on OS X.
	# * FIXME: `load_linkdep`
	#   it might fail because of unimplemented `getDependencies()` in rt.sections_elf_shared
	ifneq (,$(findstring win,$(OS)))
		# LDC FIXME: disable `load_linkdep` on Windows - needs `getDependencies()`
		TESTS+=finalize host
	else
		TESTS+=finalize host load_linkdep
	endif
endif
TESTS+=link_linkdep link_loaddep load_loaddep load_13414

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

all: $(addprefix $(ROOT)/,$(addsuffix .done,$(TESTS))) loadlibwin dllrefcount dllgc
LIB_EXT:=lib
DLL_EXT:=dll
CC:=cl.exe

$(ROOT)/loadDR.done $(ROOT)/host.done: RUN_ARGS:=$(subst .lib,.dll,$(DRUNTIMESO))

dllrefcount:
	$(DMD) $(DFLAGS) src/dllrefcount.d -of$(ROOT)/dllrefcount.exe
	$(ROOT)/dllrefcount.exe
	rm $(ROOT)/dllrefcount.*

loadlibwin:
	$(DMD) $(DFLAGS) src/loadlibwin.d -of$(ROOT)/loadlibwin.exe
	$(ROOT)/loadlibwin.exe
	rm $(ROOT)/loadlibwin.*

dllgc:
	$(DMD) $(DFLAGS) -version=DLL -shared -of$(ROOT)/dllgc.dll src/dllgc.d
	$(DMD) $(DFLAGS) -of$(ROOT)/loaddllgc.exe src/dllgc.d
	$(ROOT)/loaddllgc.exe
	rm $(ROOT)/loaddllgc.* $(ROOT)/dllgc.*

# LDC: this test is designed for .exe & .dll with separate druntimes
dllgc: DFLAGS+=-link-defaultlib-shared=false

# end Windows
endif

$(ROOT)/%.done: $(ROOT)/%
	@echo Testing $*
	$(QUIET)$(TIMELIMIT)$< $(RUN_ARGS)
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
	$(QUIET)$(CC) $(CFLAGS) /Fe$@ $< $(ROOT)/lib.$(LIB_EXT)

$(ROOT)/linkDR: $(SRC)/linkDR.c $(ROOT)/lib.$(DLL_EXT) $(DRUNTIMESO)
	$(QUIET)$(CC) $(CFLAGS) /Fe$@ $< $(DRUNTIMESO)

$(ROOT)/loadDR: $(SRC)/loadDR.c $(ROOT)/lib.$(DLL_EXT) $(DRUNTIMESO)
	$(QUIET)$(CC) $(CFLAGS) /Fe$@ $<

$(ROOT)/host: $(SRC)/host.c $(ROOT)/plugin1.$(DLL_EXT) $(ROOT)/plugin2.$(DLL_EXT)
	$(QUIET)$(CC) $(CFLAGS) /Fe$@ $<

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
