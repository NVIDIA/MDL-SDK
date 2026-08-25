/******************************************************************************
 * Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

// examples/mdl_sdk/shared/utils/main_utf8.h
//
// Methods to deal with UTF8-encoded command-line arguments.
//
// Be carefule with additional dependencies. This file is also used standalone without linking
// the entire shared project.

#ifndef EXAMPLE_SHARED_UTILS_MAIN_UTF8_H
#define EXAMPLE_SHARED_UTILS_MAIN_UTF8_H

#include <mi/base.h>

#include <cassert>

#ifdef MI_PLATFORM_WINDOWS

// Use main_utf8(int,char*[]) as entry point.
#define MAIN_UTF8 main_utf8

// Define wmain() to translate command-line arguments to UTF8 and pass them to main_utf().
// Set console output page to UTF8.
#define COMMANDLINE_TO_UTF8 \
    int wmain( int argc, wchar_t* argv[]) { \
        char** argv_utf8 = new char*[argc]; \
        for( int i = 0; i < argc; i++) { \
            LPWSTR warg = argv[i]; \
            DWORD size = WideCharToMultiByte( CP_UTF8, 0, warg, -1, NULL, 0, NULL, NULL); \
            assert( size > 0); \
            argv_utf8[i] = new char[size]; \
            DWORD result = WideCharToMultiByte( \
                CP_UTF8, 0, warg, -1, argv_utf8[i], size, NULL, NULL); \
            assert( result > 0); \
        } \
        SetConsoleOutputCP( CP_UTF8); \
        int result = main_utf8( argc, argv_utf8); \
        for( int i = 0; i < argc; i++) \
            delete[] argv_utf8[i]; \
        delete[] argv_utf8; \
        return result; \
    }

#else // MI_PLATFORM_WINDOWS

// Use main(int,char*[]) as entry point.
#define MAIN_UTF8 main

// No conversion/code page changes needed.
#define COMMANDLINE_TO_UTF8

#endif // MI_PLATFORM_WINDOWS

#endif // EXAMPLE_SHARED_UTILS_MAIN_UTF8_H
