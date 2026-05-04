module extrafiles.sectiondefs;
import core.attribute;

version(Windows)
    enum PlatformEntryName(string Name) = "." ~ Name ~ "$N";
else version (OSX)
    enum PlatformEntryName(string Name) = "__DATA," ~ Name;
else
    enum PlatformEntryName(string Name) = Name;

@section(PlatformEntryName!"myInts") @assumeUsed
__gshared int anInt = 9;

@section(PlatformEntryName!"my8Ints") @assumeUsed
__gshared int[8] an8Int = [64, 72, 9, 81, 21, 59, 45, 2];
