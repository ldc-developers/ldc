module phobos;

import
std.array,
std.base64,
std.ctype,
std.format,
std.intrinsic,
std.math,
std.moduleinit,
std.outofmemory,
std.stdarg,
std.stdint,
std.stdio,
std.string,
std.thread,
std.traits,
std.uni,
std.utf,

std.c.fenv,
std.c.locale,
std.c.math,
std.c.process,
std.c.stdarg,
std.c.stddef,
std.c.stdio,
std.c.stdlib,
std.c.string,
std.c.time,
std.file,
std.date,
std.socket,
std.zlib,
std.cstream;

version(linux) {
    import
    std.c.linux.linux,
    std.c.linux.linuxextern,
    std.c.linux.pthread,
    std.c.linux.socket;
}
