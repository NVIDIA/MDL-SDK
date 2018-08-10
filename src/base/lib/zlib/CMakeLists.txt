# name of the target and the resulting library
set(PROJECT_NAME base-lib-zlib)

# collect sources
set(PROJECT_SOURCES 
    "zlib_wrapper.cpp"
    "adler32.c"
    "compress.c"
    "crc32.c"
    "deflate.c"
    "infback.c"
    "inffast.c"
    "inflate.c"
    "inftrees.c"
    "trees.c"
    "uncompr.c"
    "zutil.c"
    "gzread.c"
    "gzwrite.c"
    "gzclose.c"
    "gzlib.c"
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    )
