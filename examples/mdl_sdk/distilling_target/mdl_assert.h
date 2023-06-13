//*****************************************************************************
// Copyright 2023 NVIDIA Corporation. All rights reserved.
//*****************************************************************************
/// \file mdl_assert.h
/// \brief Assert macro
///
//*****************************************************************************

#pragma once

#include <iostream>

// remap internal ASSERT to our asset here, needs to come before and 
// other header gets to include the internal base/lib/lig/i_log_assert.h file
#define BASE_LIB_LOG_I_LOG_ASSERT_H
#define ASSERT(m,x) mdl_assert(x)
#define DEBUG_ASSERT(m,x) mdl_assert(x)

/// If possible, lets the asserts support function names in their message.
#if defined(__FUNCSIG__)
#  define MDL_ASSERT_FUNCTION __FUNCSIG__
#elif defined( __cplusplus) && defined(__GNUC__) && defined(__GNUC_MINOR__) \
        && ((__GNUC__ << 16) + __GNUC_MINOR__ >= (2 << 16) + 6)
#  define MDL_ASSERT_FUNCTION    __PRETTY_FUNCTION__
#else
#  if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#    define MDL_ASSERT_FUNCTION    __func__
#  else
#    define MDL_ASSERT_FUNCTION    ("unknown")
#  endif
#endif

#ifdef NDEBUG
#define mdl_assert(expr) (static_cast<void>(0)) // valid but void null stmt
#define mdl_assert_msg(expr, msg) (static_cast<void>(0)) // valid but void null stmt
#else
#define mdl_assert(expr)  \
    (void)((expr) || (mdl_assert_fct( __FILE__, __LINE__, MDL_ASSERT_FUNCTION, #expr),0))
#define mdl_assert_msg(expr, msg) \
    (void)((expr) || (mdl_assert_fct( __FILE__, __LINE__, MDL_ASSERT_FUNCTION, msg),0))
#endif // NDEBUG

void mdl_assert_fct (const char *file, int line, const char* fct, const char *msg);
