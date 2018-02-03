include ../common.mak

ifneq (,$(findstring Win,$(OS)))
TESTS:=
else
TESTS:=fiber_guard_page
endif

.PHONY: all clean
all: $(addprefix $(ROOT)/,$(addsuffix .done,$(TESTS)))

# segfault || bus error (OSX)
$(ROOT)/fiber_guard_page.done: $(ROOT)/%.done : $(ROOT)/%
	@echo Testing $*
	$(QUIET)$(TIMELIMIT)$(ROOT)/$* $(RUN_ARGS); rc=$$?; [ $$rc -eq 139 ] || [ $$rc -eq 138 ]
	@touch $@

$(ROOT)/%: $(SRC)/%.d
	$(QUIET)$(DMD) $(DFLAGS) -of$@ $<

clean:
	rm -rf $(GENERATED)
