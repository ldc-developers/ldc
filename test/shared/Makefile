LINK_SHARED:=1

include ../common.mak

TESTS:=link load linkD linkDR loadDR
# LDC: disable 2 tests on Mac
ifneq ($(OS),osx)
	# * `finalize` fails starting with macOS 10.13, as .dylibs with TLS can't be
	#   unloaded anymore (https://github.com/ldc-developers/ldc/issues/3002).
	# * `host` loads two modules with the same name, which is currently disallowed
	#   by the (potentially overly eager) module collision detection on OS X.
	TESTS+=finalize host
endif
TESTS+=link_linkdep load_linkdep link_loaddep load_loaddep load_13414

EXPORT_DYNAMIC=$(if $(findstring $(OS),linux freebsd dragonflybsd),-L--export-dynamic,)
NO_AS_NEEDED=$(if $(findstring $(OS),linux freebsd dragonflybsd),-L--no-as-needed,)

.PHONY: all clean
all: $(addprefix $(ROOT)/,$(addsuffix .done,$(TESTS)))

$(ROOT)/loadDR.done $(ROOT)/host.done: RUN_ARGS:=$(DRUNTIMESO)

$(ROOT)/%.done: $(ROOT)/%
	@echo Testing $*
	$(QUIET)$(TIMELIMIT)$< $(RUN_ARGS)
	@touch $@

$(ROOT)/link: $(SRC)/link.d $(ROOT)/lib.so $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< -L$(ROOT)/lib.so

$(ROOT)/link_linkdep: $(SRC)/link_linkdep.d $(ROOT)/lib.so $(ROOT)/liblinkdep.so $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< $(LINKFLAGS) -L$(ROOT)/liblinkdep.so -L$(ROOT)/lib.so

$(ROOT)/load_linkdep: $(SRC)/load_linkdep.d $(ROOT)/lib.so $(ROOT)/liblinkdep.so $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< $(LINKFLAGS) $(LINKDL)

$(ROOT)/link_loaddep: $(SRC)/link_loaddep.d $(ROOT)/lib.so $(ROOT)/libloaddep.so $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< $(LINKFLAGS) -L$(ROOT)/libloaddep.so

$(ROOT)/load_loaddep: $(SRC)/load_loaddep.d $(ROOT)/lib.so $(ROOT)/libloaddep.so $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< $(LINKFLAGS) $(LINKDL)

$(ROOT)/load $(ROOT)/finalize: $(ROOT)/%: $(SRC)/%.d $(ROOT)/lib.so $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< $(LINKDL)

$(ROOT)/load_13414: $(ROOT)/%: $(SRC)/%.d $(ROOT)/lib_13414.so $(DRUNTIMESO)
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $< $(LINKDL)

$(ROOT)/linkD: $(SRC)/linkD.c $(ROOT)/lib.so $(DRUNTIMESO)
	$(QUIET)$(CC) $(CFLAGS) -o $@ $< $(ROOT)/lib.so $(LDL) -pthread

$(ROOT)/linkDR: $(SRC)/linkDR.c $(ROOT)/lib.so $(DRUNTIMESO)
	$(QUIET)$(CC) $(CFLAGS) -o $@ $< $(DRUNTIMESO) $(LDL) -pthread

$(ROOT)/loadDR: $(SRC)/loadDR.c $(ROOT)/lib.so $(DRUNTIMESO)
	$(QUIET)$(CC) $(CFLAGS) -o $@ $< $(LDL) -pthread

$(ROOT)/host: $(SRC)/host.c $(ROOT)/plugin1.so $(ROOT)/plugin2.so
	$(QUIET)$(CC) $(CFLAGS) -o $@ $< $(LDL) -pthread

$(ROOT)/liblinkdep.so: $(ROOT)/lib.so
$(ROOT)/liblinkdep.so: DFLAGS+=-L$(ROOT)/lib.so

$(ROOT)/plugin1.so $(ROOT)/plugin2.so: $(SRC)/plugin.d $(DRUNTIMESO)
	$(QUIET)$(DMD) -fPIC -shared $(DFLAGS) -of$@ $<

$(ROOT)/%.so: $(SRC)/%.d $(DRUNTIMESO)
	$(QUIET)$(DMD) -fPIC -shared $(DFLAGS) -of$@ $< $(LINKDL)

clean:
	rm -rf $(GENERATED)
