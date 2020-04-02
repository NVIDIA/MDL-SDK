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

#include "pch.h"

#include "plug_impl.h"

#include <cstring>
#include <mi/neuraylib/iplugin_api.h>
#include <base/system/main/module_registration.h>
#include <base/lib/log/i_log_logger.h>

namespace MI {

namespace PLUG {

static SYSTEM::Module_registration<Plug_module_impl> s_module( M_PLUG, "PLUG");

SYSTEM::Module_registration_entry* Plug_module::get_instance()
{
    return s_module.init_module( s_module.get_name());
}

bool Plug_module_impl::init()
{
    m_link_module.set();
    return m_link_module.get_status() == SYSTEM::MODULE_STATUS_INITIALIZED;
}

void Plug_module_impl::exit()
{
    m_plugins.clear();
    m_link_module.reset();
}

bool Plug_module_impl::load_library( const char* path)
{
    ASSERT( M_PLUG, path);

    mi::base::Lock::Block block( &m_lock);

    // Load library
    LOG::mod_log->debug(
        M_PLUG, LOG::Mod_log::C_PLUGIN, "Attempting to load library \"%s\".", path);
    mi::base::Handle<LINK::ILibrary> library( m_link_module->load_library( path));
    if( !library)
        return false;

    // Update path (use canonical name in neuray).
    std::string s = library->get_filename( "mi_plugin_factory");
    std::string path_str = !s.empty() ? s : path;
    path = 0; // avoid accidental use

    // Do not load a library more than once
    for( size_t i = m_plugins.size(); i > 0; --i)
        if( strcmp( m_plugins[i-1]->get_plugin_library_path(), path_str.c_str()) == 0) {
            LOG::mod_log->error(
                M_PLUG, LOG::Mod_log::C_PLUGIN, 4, "Library %s: already loaded", path_str.c_str());
            return false;
        }

    LOG::mod_log->info( M_PLUG, LOG::Mod_log::C_PLUGIN, "Loaded library \"%s\".", path_str.c_str());

    // Retrieve factory symbol
    mi::base::Plugin_factory* factory
        = (mi::base::Plugin_factory*) library->get_symbol( "mi_plugin_factory");
    if( !factory) {
        LOG::mod_log->error( M_PLUG, LOG::Mod_log::C_PLUGIN, 1,
            "Library %s: symbol \"mi_plugin_factory\" not found.", path_str.c_str());
        return false;
    }

    // Invoke plugin factory for all plugins
    std::vector<mi::base::Handle<mi::base::IPlugin_descriptor> > new_plugins;
    for( size_t i = 0; true; ++i) {

        mi::base::Plugin* plugin = factory( i, NULL);
        if( !plugin)
            break;

        mi::Sint32 plugin_system_version = plugin->get_plugin_system_version();
        if( plugin_system_version != mi::base::Plugin::s_version) {
            LOG::mod_log->error( M_PLUG, LOG::Mod_log::C_PLUGIN, 2,
                "Library \"%s\": found plugin with unsupported plugin system version %d, "
                "ignoring plugin.", path_str.c_str(), plugin_system_version);
            plugin->release();
            continue;
        }

        mi::base::Handle<mi::base::IPlugin_descriptor> descriptor( // takes over plugin
            new Plugin_descriptor_impl( library.get(), plugin, path_str.c_str()));
        m_plugins.push_back( descriptor);
        new_plugins.push_back( descriptor);

    }

    return true;
}

size_t Plug_module_impl::get_plugin_count()
{
    mi::base::Lock::Block block( &m_lock);
    return m_plugins.size();
}

mi::base::IPlugin_descriptor* Plug_module_impl::get_plugin( const char* name)
{
    mi::base::Lock::Block block( &m_lock);
    for( size_t i = 0; i < m_plugins.size(); ++i)
        if( strcmp( m_plugins[i]->get_plugin()->get_name(), name) == 0) {
            m_plugins[i]->retain();
            return m_plugins[i].get();
        }
    return 0;
}

mi::base::IPlugin_descriptor* Plug_module_impl::get_plugin( size_t index)
{
    mi::base::Lock::Block block( &m_lock);
    if( index >= m_plugins.size())
        return 0;
    m_plugins[index]->retain();
    return m_plugins[index].get();
}

void Plug_module_impl::set_plugin_api( mi::neuraylib::IPlugin_api* plugin_api)
{
    m_plugin_api = make_handle_dup( plugin_api);
}

mi::neuraylib::IPlugin_api* Plug_module_impl::get_plugin_api() const
{
    if( m_plugin_api)
        m_plugin_api->retain();
    return m_plugin_api.get();
}


} // namespace PLUG

} // namespace MI

