/**
REQUIRED_ARGS: -ftime-trace -ftime-trace-file=- -ftime-trace-granularity=0
TRANSFORM_OUTPUT: sanitize_timetrace
TEST_OUTPUT:
---
Code generation,
Codegen: function add, object.add
Codegen: function fun, object.fun
Codegen: module object, object
Ctfe: add(4, 8), add(4, 8)
Ctfe: call add, object.add(4, 8)
Generate IR, object
Import object.object, object.object
Optimize, $r:.*ftimetrace_0\.o(bj)?$
Parse: Module object, object
Parsing,
Prune object file cache,
Sema1: Module object, object
Sema2: add, object.add
Sema2: fun, object.fun
Sema3: add, object.add
Sema3: fun, object.fun
Semantic analysis,
Write file(s), $r:.*ftimetrace_0\.o(bj)?$
---
*/

module object; // Don't clutter time trace output with object.d

void fun()
{
    enum z = add(4, 8);
}

int add(int x, int y)
{
    return x + y;
}
