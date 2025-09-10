/***************************************************************************************************
 * Copyright (c) 2008-2025, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief Implementation of IPlugin_configuration
 **
 ** Implements the IPlugin_configuration interface
 **/

#include "pch.h"

#include "neuray_plugin_api_impl.h"
#include "neuray_plugin_configuration_impl.h"

#include <api/api/mdl/mdl_neuray_impl.h>


#include <filesystem>
#include <vector>

#include <base/hal/disk/disk_utils.h>
#include <base/lib/plug/i_plug.h>


namespace fs = std::filesystem;

namespace MI {

namespace NEURAY {

Plugin_configuration_impl::Plugin_configuration_impl( mi::neuraylib::INeuray* neuray)
  : m_neuray( neuray)
{
    m_plug_module.set();

    auto* neuray_impl = static_cast<MDL::Neuray_impl*>( neuray);
    mi::base::Handle<mi::neuraylib::IPlugin_api> plugin_api(
        neuray_impl->get_plugin_api());
    m_plug_module->set_plugin_api( plugin_api.get());
}

Plugin_configuration_impl::~Plugin_configuration_impl()
{
    m_plug_module->set_plugin_api( nullptr);
    m_plug_module.reset();

    m_neuray = nullptr;
}

mi::Sint32 Plugin_configuration_impl::load_plugin_library( const char* path)
{
    if( !path)
        return -1;

    mi::neuraylib::INeuray::Status status = m_neuray->get_status();
    if( status != mi::neuraylib::INeuray::PRE_STARTING)
        return -1;

    return m_plug_module->load_library( path) ? 0 : -1;
}

mi::Sint32 Plugin_configuration_impl::load_plugins_from_directory( const char* path)
{
    if( !path)
        return -1;

    mi::neuraylib::INeuray::Status status = m_neuray->get_status();
    if( status != mi::neuraylib::INeuray::PRE_STARTING)
        return -1;

    try {

        fs::path directory( fs::u8path( path));
        if( !fs::is_directory( directory))
            return -1;

        std::vector<fs::path> filenames;
        for( const auto& entry: fs::directory_iterator( directory))
            if( fs::is_regular_file( entry.path()))
                filenames.push_back( entry.path());
        std::sort( filenames.begin(), filenames.end());

        mi::Sint32 result = 0;

        for( const auto& filename: filenames) {
            std::string extension = DISK::to_string( filename.extension());
#if defined(MI_PLATFORM_WINDOWS)
            if( extension != ".dll")
                continue;
#elif defined(MI_PLATFORM_LINUX)
            if( extension != ".so")
                continue;
#elif defined(MI_PLATFORM_MACOSX)
            if( extension != ".so" && extension != ".dylib")
                continue;
#else
#error Unsupported platform
#endif

            if( load_plugin_library( DISK::to_string( filename).c_str()) != 0)
                result = -1;
        }

        return result;

    } catch( ...) {
        return -1;
    }
}

mi::Size Plugin_configuration_impl::get_plugin_length() const
{
    return m_plug_module->get_plugin_count();
}

mi::base::IPlugin_descriptor* Plugin_configuration_impl::get_plugin_descriptor(
    mi::Size index) const
{
    return m_plug_module->get_plugin( index);
}

mi::Sint32 Plugin_configuration_impl::start()
{
    return 0;
}

mi::Sint32 Plugin_configuration_impl::shutdown()
{
    return 0;
}

} // namespace NEURAY

} // namespace MI

