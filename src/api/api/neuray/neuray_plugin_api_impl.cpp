/***************************************************************************************************
 * Copyright (c) 2010-2022, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Source for the IPlugin_api implementation.
 **/

#include "pch.h"

#include "neuray_plugin_api_impl.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/ineuray.h>
#include <mi/neuraylib/version.h>

#include <cstring>
#include <base/lib/log/i_log_module.h>
#include <base/lib/plug/i_plug.h>

namespace MI {

namespace NEURAY {

Plugin_api_impl::Plugin_api_impl( mi::neuraylib::INeuray* neuray)
  : m_neuray( neuray)
{
}

Plugin_api_impl::~Plugin_api_impl()
{
    m_neuray = nullptr;
}

mi::Sint32 Plugin_api_impl::start()
{
    m_plug_module.set();


    return 0;
}

mi::Sint32 Plugin_api_impl::shutdown()
{

    m_plug_module.reset();

    return 0;
}

mi::Uint32 Plugin_api_impl::get_interface_version() const
{
    return MI_NEURAYLIB_API_VERSION;
}

const char* Plugin_api_impl::get_version() const
{
    return m_neuray->get_version();
}

mi::base::IInterface* Plugin_api_impl::get_api_component(
    const mi::base::Uuid& uuid) const
{
    return m_neuray->get_api_component( uuid);
}

mi::Sint32 Plugin_api_impl::register_api_component(
    const mi::base::Uuid& uuid, mi::base::IInterface* api_component)
{
    return m_neuray->register_api_component( uuid, api_component);
}

mi::Sint32 Plugin_api_impl::unregister_api_component( const mi::base::Uuid& uuid)
{
    return m_neuray->unregister_api_component( uuid);
}

bool Plugin_api_impl::is_valid_api_plugin(
    const char* type, const char* name, const char* filename)
{
    if( !type)
        return false;

    // current version
    if( 0 == strcmp( type, MI_NEURAYLIB_PLUGIN_TYPE))
        return true;

    if( false /*(0 == strcmp( type, "version_goes_here"))*/) {
        LOG::mod_log->warning( M_NEURAY_API, LOG::Mod_log::C_PLUGIN,
            "API plugin of name \"%s\" from library \"%s\" has different plugin type "
            "\"%s\". If you encounter problems with this plugin, you may want to use a "
            "version of the plugin that has been compiled for the currently supported "
            "plugin type \"%s\".",
            name, filename, type, MI_NEURAYLIB_PLUGIN_TYPE);
        return true;
    }

    // unsupported versions (previous or future)
    if( (0 == strncmp( type, "neuray API", 10)) || (0 == strncmp( type, "API", 3))) {
        LOG::mod_log->error( M_NEURAY_API, LOG::Mod_log::C_PLUGIN,
            "API plugin of name \"%s\" from library \"%s\" has unsupported plugin type "
            "\"%s\". Please use a version of the plugin that has been compiled for the "
            "currently supported plugins type \"%s\".",
            name, filename, type, MI_NEURAYLIB_PLUGIN_TYPE);
        return false;
    }

    return false;
}

} // namespace NEURAY

} // namespace MI

