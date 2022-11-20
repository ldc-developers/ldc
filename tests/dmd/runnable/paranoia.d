/*
EXTRA_SOURCES: extra-files/paranoia.d
PERMUTE_ARGS:

ARG_SETS: -version=Single
ARG_SETS: -version=Double

Windows targets adapted for LDC:
ARG_SETS(windows): -version=Extended ../../dmd/root/longdouble.d
ARG_SETS(windows): -version=ExtendedSoft ../../dmd/root/longdouble.d

ARG_SETS(linux osx): -version=Extended
*/

module test.runnable.paranoia;
