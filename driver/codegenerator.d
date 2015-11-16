//===-- codegenerator.d ---------------------------------------------------===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//

module driver.codegenerator;

import ddmd.dmodule;
import ddmd.dscope;
import ddmd.globals;
import ddmd.id;
import ddmd.identifier;
import ddmd.parse;
import ddmd.tokens;

extern (C++) __gshared Module g_entrypointModule = null;
extern (C++) __gshared Module g_dMainModule = null;

/// Callback to generate a C main() function, invoked by the frontend.
extern (C++) void genCmain(Scope *sc) {
  if (g_entrypointModule) {
    return;
  }

  /* The D code to be generated is provided as D source code in the form of a
   * string.
   * Note that Solaris, for unknown reasons, requires both a main() and an
   * _main()
   */
  static __gshared const(char)[] code =
  q{
      pragma(LDC_profile_instr, false):
      extern(C)
      {
          int _d_run_main(int argc, char **argv, void* mainFunc);
          int _Dmain(char[][] args);
          int main(int argc, char **argv)
          {
              return _d_run_main(argc, argv, &_Dmain);
          }
          version (Solaris) int _main(int argc, char** argv) { return main(argc, argv); }
      }
      pragma(LDC_no_moduleinfo);
  };

  Identifier id = Id.entrypoint;
  auto m = new Module("__entrypoint.d", id, 0, 0);

  scope Parser p = new Parser(m, code.ptr, code.length, 0);
  p.scanloc = Loc();
  p.nextToken();
  m.members = p.parseModule();
  assert(p.token.value == TOKeof);

  bool v = global.params.verbose;
  global.params.verbose = false;
  m.importedFrom = m;
  m.importAll(null);
  m.semantic();
  m.semantic2();
  m.semantic3();
  global.params.verbose = v;

  g_entrypointModule = m;
  g_dMainModule = sc._module;
}

