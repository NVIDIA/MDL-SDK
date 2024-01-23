/******************************************************************************
 * Copyright (c) 2007-2024, NVIDIA CORPORATION. All rights reserved.
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
/// \file
/// \brief regression test infrastructure
///
/// - mi_src_path()           resolve a relative path inside $MI_SRC
/// - mi_data_path()          resolve a relative path inside $MI_DATA
/// - mi_large_data_path()    resolve a relative path inside $MI_LARGE_DATA

#ifndef BASE_SYSTEM_TEST_ENVIRONMENT_H
#define BASE_SYSTEM_TEST_ENVIRONMENT_H

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace MI { namespace TEST {

inline char const * get_env_variable(char const * name)
{
    using namespace std;
    char const * const val( getenv(name) );
    if (val) return val;
    runtime_error err(string("required environment variable ${") + name + "} is not set");
    throw err;
}

inline std::string get_env_path(char const * varname, std::string const & path)
{
    std::string full_path( get_env_variable(varname) );
    full_path.append("/").append(path);
    std::replace( full_path.begin(), full_path.end()
#ifdef WIN_NT
                  , '/', '\\'
#else
                  , '\\', '/'
#endif
                  );
    return full_path;
}

inline std::string mi_src_path(std::string const & path)
{
    return get_env_path("MI_SRC", path);
}

inline std::string mi_data_path(std::string const & path)
{
    return get_env_path("MI_DATA", path);
}

inline std::string mi_large_data_path(std::string const & path)
{
    return get_env_path("MI_LARGE_DATA", path);
}

}} // MI::TEST

#endif // BASE_SYSTEM_TEST_ENVIRONMENT_H

