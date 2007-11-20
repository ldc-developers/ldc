module phobos;

import
std.array,
std.ctype,
std.intrinsic,
std.math,
std.moduleinit,
std.outofmemory,
std.stdint,
std.stdio,
std.stdarg,
std.uni,
std.utf,

//std.format,
//std.string,

std.c.fenv,
std.c.locale,
std.c.math,
std.c.process,
std.c.stdarg,
std.c.stddef,
std.c.stdio,
std.c.stdlib,
std.c.string,
std.c.time;

version(linux) {
    import
    std.c.linux.linux,
    std.c.linux.linuxextern,
    std.c.linux.pthread,
    std.c.linux.socket;
}
