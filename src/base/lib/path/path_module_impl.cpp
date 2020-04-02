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

#include "pch.h"

#include "path_module_impl.h"

#include <boost/algorithm/string/replace.hpp>
#include <mi/base/config.h>
#include <base/system/main/module_registration.h>
#include <base/hal/disk/disk.h>
#include <base/hal/hal/i_hal_ospath.h>

#include <base/system/main/access_module.h>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/config/config.h>
#include <base/util/registry/i_config_registry.h>

namespace MI {

namespace PATH {

const Path_module::Path Path_module_impl::s_empty_string;

// Register the module.
static SYSTEM::Module_registration<Path_module_impl> s_module( M_PATH, "PATH");

Module_registration_entry* Path_module::get_instance()
{
    return s_module.init_module( s_module.get_name());
}

bool Path_module_impl::init()
{
    return true;
}

void Path_module_impl::exit()
{
}

mi::Sint32 Path_module_impl::set_search_path( Kind kind, const Search_path& paths)
{
    // Check for config option to allow non-existing directories as search paths
    // as needed for some dedicated customer workflows.
    SYSTEM::Access_module<CONFIG::Config_module> config_module( false);
    bool allow_invalid_search_paths = false;
    config_module->get_configuration().get_value(
        "allow_invalid_search_paths", allow_invalid_search_paths);

    size_t n = paths.size();
    mi::Sint32 return_code = 0;
    for( size_t i = 0; i < n; ++i) {
        if( paths[i].empty())
            return -2;
        if( !DISK::is_directory( paths[i].c_str())) {
            if( !allow_invalid_search_paths)
                return -2;
            else
                return_code = -2;
        }
    }

    mi::base::Lock::Block block( &m_lock);
    Search_path& search_path = m_search_paths[kind];
    search_path.resize( n);
    for( size_t i = 0; i < n; ++i)
        search_path[i] = normalize( paths[i]);
    return return_code;
}

const Path_module::Search_path& Path_module_impl::get_search_path( Kind kind) const
{
    mi::base::Lock::Block block( &m_lock);
    return m_search_paths[kind];
}

void Path_module_impl::clear_search_path( Kind kind)
{
    mi::base::Lock::Block block( &m_lock);
    m_search_paths[kind].clear();
}

size_t Path_module_impl::get_path_count( Kind kind) const
{
    mi::base::Lock::Block block( &m_lock);
    return m_search_paths[kind].size();
}

mi::Sint32 Path_module_impl::set_path( Kind kind, size_t index, const Path& path)
{
    // Check for config option to allow non-existing directories as search paths
    // as needed for some dedicated customer workflows.
    SYSTEM::Access_module<CONFIG::Config_module> config_module( false);
    bool allow_invalid_search_paths = false;
    config_module->get_configuration().get_value(
        "allow_invalid_search_paths", allow_invalid_search_paths);

    if( path.empty())
        return -2;
    mi::Sint32 return_code = 0;
    if( !DISK::is_directory( path.c_str())) {
        if( !allow_invalid_search_paths)
            return -2;
        else
            return_code = -2;
    }

    mi::base::Lock::Block block( &m_lock);
    Search_path& search_path = m_search_paths[kind];
    if( index >= search_path.size())
        return -3;

    search_path[index] = normalize( path);
    return return_code;
}

const Path_module::Path& Path_module_impl::get_path( Kind kind, size_t index) const
{
    mi::base::Lock::Block block( &m_lock);
    const Search_path& search_path = m_search_paths[kind];
    if( index >= search_path.size())
        return s_empty_string;
    return search_path[index];
}

mi::Sint32 Path_module_impl::add_path( Kind kind, const Path& path)
{
    // Check for config option to allow non-existing directories as search paths
    // as needed for some dedicated customer workflows.
    SYSTEM::Access_module<CONFIG::Config_module> config_module( false);
    bool allow_invalid_search_paths = false;
    config_module->get_configuration().get_value(
        "allow_invalid_search_paths", allow_invalid_search_paths);

    if( path.empty())
        return -2;
    mi::Sint32 return_code = 0;
    if( !DISK::is_directory( path.c_str())) {
        if( !allow_invalid_search_paths)
            return -2;
        else
            return_code = -2;
    }

    mi::base::Lock::Block block( &m_lock);
    m_search_paths[kind].push_back( normalize( path));
    return return_code;
}

mi::Sint32 Path_module_impl::remove_path( Kind kind, const Path& path)
{
    mi::base::Lock::Block block( &m_lock);

    Search_path& search_path = m_search_paths[kind];
    size_t n = search_path.size();

    const std::string normalized_path = normalize( path);
    for( size_t i = 0; i < n; ++i)
        if( search_path[i] == normalized_path) {
            search_path.erase( search_path.begin() + i);
            return 0;
        }

    return -2;
}

std::string Path_module_impl::search(
    Kind kind, const std::string& file_name) const
{
    const std::string& normalized_file_name = normalize( file_name);

    if( DISK::is_path_absolute( normalized_file_name))
        return DISK::access( normalized_file_name.c_str()) ? normalized_file_name : s_empty_string;

    mi::base::Lock::Block block( &m_lock);

    const Search_path& search_path = m_search_paths[kind];
    size_t n = search_path.size();

    for( size_t i = 0; i < n; ++i) {
        const std::string& joined_file_name
            = HAL::Ospath::join( search_path[i], normalized_file_name);
        if( DISK::access( joined_file_name.c_str()))
            return joined_file_name;
    }

    return s_empty_string;
}

std::string Path_module_impl::normalize( const std::string& s)
{
#ifdef MI_PLATFORM_WINDOWS
    return boost::replace_all_copy( s, "/", "\\");
#else
    return s;
#endif
}

} // namespace PATH

} // namespace MI
