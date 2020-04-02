/***************************************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef BASE_LIB_PLUG_PLUG_IMPL_H
#define BASE_LIB_PLUG_PLUG_IMPL_H

#include "i_plug.h"

#include <string>
#include <vector>

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/base/lock.h>
#include <mi/base/plugin.h>

#include <base/system/main/access_module.h>
#include <base/hal/link/i_link.h>

namespace MI {

namespace PLUG {

/// Implementation of #mi::base::IPlugin_descriptor.
class Plugin_descriptor_impl
  : public mi::base::Interface_implement<mi::base::IPlugin_descriptor>,
    public STLEXT::Non_copyable
{
public:
    /// Constructor
    ///
    /// \param library     The library this plugin belongs to.
    /// \param plugin      The plugin. Takes over ownership. Must not be \c NULL.
    /// \param path        The plugin library path.
    Plugin_descriptor_impl(
        LINK::ILibrary* library, mi::base::Plugin* plugin, const char* path)
      : m_library( library, mi::base::DUP_INTERFACE), m_plugin( plugin), m_path( path) { }

    /// Destructor
    ///
    /// Destroys the wrapped plugin.
    ~Plugin_descriptor_impl() { m_plugin->release(); }

    // public API methods

    mi::base::Plugin* get_plugin() const { return m_plugin; }

    const char* get_plugin_library_path() const { return m_path.c_str(); }

    // internal methods

private:
    mi::base::Handle<LINK::ILibrary> m_library;
    mi::base::Plugin*                m_plugin;
    std::string                    m_path;
};

/// Implementation of the Plug_module interface
class Plug_module_impl : public Plug_module
{
public:

    // interface methods

    bool load_library( const char* path);

    size_t get_plugin_count();

    mi::base::IPlugin_descriptor* get_plugin( size_t index);

    mi::base::IPlugin_descriptor* get_plugin( const char* name);

    void set_plugin_api( mi::neuraylib::IPlugin_api* plugin_api);

    mi::neuraylib::IPlugin_api* get_plugin_api() const;


    // methods of SYSTEM::IModule

    bool init();

    void exit();

private:
    /// Lock for #m_plugins.
    mi::base::Lock m_lock;

    /// The loaded plugins. Needs #m_lock.
    std::vector<mi::base::Handle<mi::base::IPlugin_descriptor> > m_plugins;

    /// The LINK module.
    SYSTEM::Access_module<LINK::Link_module> m_link_module;

    /// The plugin API of the Iray/DiCE API.
    mi::base::Handle<mi::neuraylib::IPlugin_api> m_plugin_api;

};

} // namespace PLUG

} // namespace MI

#endif // BASE_LIB_PLUG_PLUG_IMPL_H

