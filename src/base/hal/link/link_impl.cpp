/***************************************************************************************************
 * Copyright (c) 2007-2020, NVIDIA CORPORATION. All rights reserved.
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

/// \file
/// \brief

#include "pch.h"

#include "link_impl.h"

#include <mi/base/config.h>
#include <base/system/main/module_registration.h>
#include <base/lib/log/i_log_logger.h>

#ifndef MI_PLATFORM_WINDOWS
#include <dlfcn.h>
#else
#include <mi/base/miwindows.h>
#include <base/util/string_utils/i_string_utils.h>
#endif

#ifdef MI_PLATFORM_LINUX
#include <base/system/main/access_module.h>
#include <base/util/registry/i_config_registry.h>
#include <base/lib/config/config.h>
#endif

namespace MI {

namespace LINK {

Library_impl::Library_impl( void* handle)
  : m_handle( handle)
{
}

Library_impl::~Library_impl()
{
#ifndef MI_PLATFORM_WINDOWS
    dlclose( m_handle);
#else // MI_PLATFORM_WINDOWS
    FreeLibrary( (HMODULE)m_handle);
#endif // MI_PLATFORM_WINDOWS
}

void* Library_impl::get_symbol( const char* symbol_name)
{
#ifndef MI_PLATFORM_WINDOWS
    void* symbol = dlsym( m_handle, symbol_name);
    return dlerror() != 0 ? 0 : symbol;
#else // MI_PLATFORM_WINDOWS
    return GetProcAddress( (HMODULE)m_handle, symbol_name);
#endif // MI_PLATFORM_WINDOWS
}

std::string Library_impl::get_filename( const char* symbol_name)
{
#ifndef MI_PLATFORM_WINDOWS
    void* symbol = dlsym( m_handle, symbol_name);
    if( !symbol)
        return "";
    Dl_info dl_info;
    if( !dladdr(symbol, &dl_info))
        return "";
    if( !dl_info.dli_fname)
        return "";
    return dl_info.dli_fname;
#else // MI_PLATFORM_WINDOWS
    TCHAR filename[MAX_PATH];
    if( !GetModuleFileName( (HMODULE)m_handle, filename, MAX_PATH))
        return "";
    return filename;
#endif // MI_PLATFORM_WINDOWS
}

ILibrary* Link_module_impl::load_library( const char* path)
{
#ifndef MI_PLATFORM_WINDOWS

#ifdef MI_PLATFORM_MACOSX
    int dlopen_flags = RTLD_LAZY;
#else // MI_PLATFORM_MACOSX
    int dlopen_flags = RTLD_LAZY;
    SYSTEM::Access_module<CONFIG::Config_module> config_module( false);
    bool use_rtld_deepbind_for_plugins = false;
    config_module->get_configuration().get_value(
        "use_rtld_deepbind_for_plugins", use_rtld_deepbind_for_plugins);
    if( use_rtld_deepbind_for_plugins)
        dlopen_flags |= RTLD_DEEPBIND;
#endif // MI_PLATFORM_MACOSX
    void* handle = dlopen( path, dlopen_flags);
    if( !handle) {
        LOG::mod_log->warning( M_LINK, LOG::Mod_log::C_PLUGIN, 14,
            "Loading %s: %s", path, dlerror());
        return 0;
    }

#else // MI_PLATFORM_WINDOWS

    const std::wstring& wpath( MI::STRING::utf8_to_wchar( path));
    void* handle = LoadLibraryW( wpath.c_str());

    if( !handle) {
        LPTSTR buffer = 0;
        LPCTSTR message = TEXT("unknown failure");
        DWORD error_code = GetLastError();
        if( FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS, 0, error_code,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &buffer, 0, 0))
            message = buffer;
        LOG::mod_log->warning( M_LINK, LOG::Mod_log::C_PLUGIN,
#ifdef UNICODE
            "Loading %s: %ls", path, message);
#else // UNICODE
            "Loading %s: %s", path, message);
#endif // UNICODE
        if( buffer)
            LocalFree( buffer);
        return 0;
    }

#endif // MI_PLATFORM_WINDOWS

    return new Library_impl( handle);
}

static SYSTEM::Module_registration<Link_module_impl> s_module( M_LINK, "LINK");

Module_registration_entry* Link_module::get_instance()
{
    return s_module.init_module( s_module.get_name());
}

} // namespace LINK

} // namespace MI
