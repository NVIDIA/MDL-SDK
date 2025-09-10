/******************************************************************************
 * Copyright (c) 2011-2025, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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

#ifndef IO_IMAGE_IMAGE_TEST_SHARED_H
#define IO_IMAGE_IMAGE_TEST_SHARED_H

#include <cstdlib>
#include <string>

#include <base/system/test/i_test_case.h>

void MI_CHECK_IMG_DIFF( const char* image1, const char* image2, bool strict = true)
{

    // The path containing idiff is provided by the CTest environment.
#ifdef MI_PLATFORM_WINDOWS
    std::string command = "idiff.exe";
#else
    std::string command = "idiff";
#endif


#if defined( MI_PLATFORM_LINUX) && defined( MI_ARCH_ARM_64) && defined( NDEBUG)
    command += strict ? " -fail 0.00071" : " -fail 0.01 -failpercent 0.01";
#else
    command += strict ? " -fail 0.000016" : " -fail 0.01 -failpercent 0.01";
#endif
    command += " -a ";
    command += image1;
    command += " ";
    command += image2;
    std::cout << std::flush;
    int result = system( command.c_str());
#ifdef MI_PLATFORM_WINDOWS
    MI_CHECK_MSG( result == 0, "image comparison failed");
#else
    int exitcode = WEXITSTATUS( result);
    MI_CHECK_MSG( exitcode != 127, "failed to execute idiff");
    MI_CHECK_MSG( exitcode == 0, "image comparison failed");
#endif
}

void MI_CHECK_IMG_DIFF( const std::string& image1, const std::string& image2, bool strict = true)
{
    MI_CHECK_IMG_DIFF( image1.c_str(), image2.c_str(), strict);
}

#endif // IO_IMAGE_IMAGE_TEST_SHARED_H

