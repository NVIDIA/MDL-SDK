/***************************************************************************************************
 * Copyright (c) 2010-2021, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Header for the IPlugin implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_PLUGIN_IMPL_H
#define API_API_NEURAY_NEURAY_PLUGIN_IMPL_H

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/base/lock.h>
#include <mi/base/plugin.h>
#include <mi/neuraylib/iplugin_api.h>

#include <vector>
#include <boost/core/noncopyable.hpp>
#include <base/system/main/access_module.h>

namespace mi { namespace neuraylib { class INeuray; } }

namespace MI {

namespace PLUG { class Plug_module; }

namespace NEURAY {

class Neuray_impl;

class Plugin_api_impl
  : public mi::base::Interface_implement<mi::neuraylib::IPlugin_api>,
    public boost::noncopyable
{
public:
    /// Constructor of Plugin_api_impl
    ///
    /// \param neuray   The neuray instance which contains this Plugin_api_impl
    Plugin_api_impl( mi::neuraylib::INeuray* neuray);

    /// Destructor of Plugin_api_impl
    ~Plugin_api_impl();

    // public API methods

    mi::Uint32 get_interface_version() const;

    const char* get_version() const;

    mi::base::IInterface* get_api_component( const mi::base::Uuid& interface_id) const;

    mi::Sint32 register_api_component(
        const mi::base::Uuid& uuid, mi::base::IInterface* api_component);

    mi::Sint32 unregister_api_component( const mi::base::Uuid& uuid);

    // internal methods

    /// Starts this API component.
    ///
    /// The implementation of INeuray::start() calls the #start() method of each API component.
    /// This method performs the API component's specific part of the library start.
    ///
    /// \return            0, in case of success, -1 in case of failure.
    mi::Sint32 start();

    /// Shuts down this API component.
    ///
    /// The implementation of INeuray::shutdown() calls the #shutdown() method of each API
    /// component. This method performs the API component's specific part of the library shutdown.
    ///
    /// \return           0, in case of success, -1 in case of failure
    mi::Sint32 shutdown();

private:

    /// Helper function to detect valid API plugin type names.
    ///
    /// Returns \c true if the plugin type is known and supported.
    /// Prints a suitable warning message for known supported but outdated plugins and an
    /// error message for
    ///
    /// \param type        Type of the plugin, should be MI_NEURAYLIB_PLUGIN_TYPE.
    /// \param name        Name of the plugin, for diagnostics.
    /// \param filename    Filename of the DSO, for diagnostics.
    /// \return            \c true if the plugin type is known and supported, \c false otherwise.
    ///                    Logs a warning for supported but outdated API plugin types, and an
    ///                    error for unsupported API plugin types.
    static bool is_valid_api_plugin( const char* type, const char* name, const char* filename);

    /// Pointer to Neuray_impl
    mi::neuraylib::INeuray* m_neuray;

    /// Access to the PLUG module
    SYSTEM::Access_module<PLUG::Plug_module> m_plug_module;

    /// Lock for #m_plugins.
    mi::base::Lock m_plugins_lock;

    typedef std::vector<mi::base::Handle<mi::base::IPlugin_descriptor> > Plugin_vector;

    /// The registered API plugins. Needs #m_plugins_lock.
    Plugin_vector m_plugins;

};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_PLUGIN_IMPL_H
