/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
 ******************************************************************************/
#pragma once

#include <mi/mdl_sdk.h>
#include <string>
#include <map>

#if defined(DEBUG)
/// Helper macro. Checks whether the expression is true and if not prints a message and exits.
#define check_success3(expr, error_code /* See I18N_ERROR_CODE */, extra_optional_error_string) \
do { \
	if( !(expr)) { \
		fprintf( \
            stderr, \
            "Error in file %s, line %u: \"%s\".\n", \
            __FILE__, __LINE__, #expr); \
        i18n::Errors::ouput_error(error_code, extra_optional_error_string);\
		exit( EXIT_FAILURE); \
	} \
} while( false)
#else
#define check_success3(expr, error_code /* See I18N_ERROR_CODE */, extra_optional_error_string) \
do { \
	if( !(expr)) { \
        i18n::Errors::ouput_error(error_code, extra_optional_error_string);\
		exit( EXIT_FAILURE); \
	} \
} while( false)
#endif

#define check_success2(expr1, expr2) check_success3(expr1, expr2, "")
#define check_success(expr1) check_success2(expr1, i18n::Errors::ERR_UNKNOW)

namespace i18n
{
    /// Wrapper for handling application errors
    ///
    class Errors
    {
    public:
        typedef enum {
              ERR_UNKNOW = 0
            , ERR_PARSING_ARGUMENTS
            , ERR_INIT_FAILURE
            , ERR_MODULE_PATH_FAILURE
            , ERR_UNIT_TEST
            , ERR_ARCHIVE_FAILURE

        } I18N_ERROR_CODE;

    private:
        static std::map<I18N_ERROR_CODE, std::string> the_error_table;

    public:
        static void ouput_error(
              I18N_ERROR_CODE code
            , const std::string & extra_optional_error_string="");
    };

} // namespace i18n
