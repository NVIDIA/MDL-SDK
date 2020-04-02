/***************************************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************************************/

#ifndef BASE_LIB_PATH_PATH_MODULE_IMPL_H
#define BASE_LIB_PATH_PATH_MODULE_IMPL_H

#include "i_path.h"

#include <mi/base/lock.h>

namespace MI {

namespace PATH {

/// Implementation class of the PATH module
class Path_module_impl : public Path_module
{
public:
    // methods of SYSTEM::IModule

    bool init();

    void exit();

    // methods of Path_module

    mi::Sint32 set_search_path( Kind kind, const Search_path& paths);

    const Search_path& get_search_path( Kind kind) const;

    void clear_search_path( Kind kind);

    size_t get_path_count( Kind kind) const;

    mi::Sint32 set_path( Kind kind, size_t index, const Path& path);

    const Path& get_path( Kind kind, size_t index) const;

    mi::Sint32 add_path( Kind kind, const Path& path);

    mi::Sint32 remove_path( Kind kind, const Path& path);

    std::string search( Kind kind, const std::string& file_name) const;

private:
    /// On Windows, replaces all slashes by backslashes.
    /// On all other platforms, returns the string unchanged.
    static std::string normalize( const std::string& s);

    /// Lock for m_search_paths.
    mutable mi::base::Lock m_lock;

    /// The stored search paths. Needs m_lock.
    Search_path m_search_paths[N_KINDS];

    /// An empty string (for const reference return values).
    static const Path s_empty_string;
};

} // namespace PATH

} // namespace MI

#endif // BASE_LIB_PATH_PATH_MODULE_IMPL_H
