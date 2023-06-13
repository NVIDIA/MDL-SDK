//*****************************************************************************
// Copyright 2023 NVIDIA Corporation. All rights reserved.
//*****************************************************************************

#include "mdl_assert.h"

#include <stdio.h>
#include <stdlib.h>

// Reporting function for failed mdl_assert().
void mdl_assert_fct( const char *file, int line, const char* fct, const char *msg) {
    fprintf( stderr, 
             "ASSERT: MDL Distiller Plugin: failed in %s %u, function %s: \"%s\".\n",
             file, line, fct, msg);
    fflush(stderr);
    abort();
}

namespace mi {
namespace mdl {

/// Reporting function for failed MDL_ASSERT() macro in mdl/ tree. calls this function to report an assertion failure.
/// The function will usually not return.
/// \param exp   the expression that failed
/// \param file  name of the file containing the assertion
/// \param line  line number of assertion in file
void report_assertion_failure(
    char const   *exp,
    char const   *file,
    unsigned int line) 
{
    fprintf(stderr, "ASSERT: MDL Distiller Plugin: failed in %s %u: \"%s\"\n", file, line, exp);
    fflush(stderr);
    abort();
}


}  // mdl
}  // mi
