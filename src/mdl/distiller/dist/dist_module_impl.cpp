/******************************************************************************
 * Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include "dist_module_impl.h"

#include <mi/base/interface_implement.h>

#include <base/system/main/module_registration.h>
#include <base/system/main/access_module.h>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/plug/i_plug.h>

#include <sstream>

using std::pair;
using std::make_pair;
using std::string;

namespace MI {
namespace DIST {

bool convert_string_to_category( const char* in, LOG::Mod_log::Category& out) {
    mi_static_assert( 14 == LOG::Mod_log::NUM_OF_CATEGORIES);

    if( !in)
        return false;

    if( strcmp( in, "MAIN"    ) == 0) { out = LOG::Mod_log::C_MAIN;      return true; }
    if( strcmp( in, "NETWORK" ) == 0) { out = LOG::Mod_log::C_NETWORK;   return true; }
    if( strcmp( in, "MEMORY"  ) == 0) { out = LOG::Mod_log::C_MEMORY;    return true; }
    if( strcmp( in, "DATABASE") == 0) { out = LOG::Mod_log::C_DATABASE;  return true; }
    if( strcmp( in, "DISK"    ) == 0) { out = LOG::Mod_log::C_DISK;      return true; }
    if( strcmp( in, "PLUGIN"  ) == 0) { out = LOG::Mod_log::C_PLUGIN;    return true; }
    if( strcmp( in, "RENDER"  ) == 0) { out = LOG::Mod_log::C_RENDER;    return true; }
    if( strcmp( in, "GEOMETRY") == 0) { out = LOG::Mod_log::C_GEOMETRY;  return true; }
    if( strcmp( in, "IMAGE"   ) == 0) { out = LOG::Mod_log::C_IMAGE;     return true; }
    if( strcmp( in, "IO"      ) == 0) { out = LOG::Mod_log::C_IO;        return true; }
    if( strcmp( in, "ERRTRACE") == 0) { out = LOG::Mod_log::C_ERRTRACE;  return true; }
    if( strcmp( in, "MISC"    ) == 0) { out = LOG::Mod_log::C_MISC;      return true; }
    if( strcmp( in, "DISTRACE") == 0) { out = LOG::Mod_log::C_DISKTRACE; return true; }
    if( strcmp( in, "COMPILER") == 0) { out = LOG::Mod_log::C_COMPILER;  return true; }
    return false;
}

/// This logger forwards all messages to the LOG module.
class Forwarding_logger : public mi::base::Interface_implement<mi::base::ILogger>
{
public:
    void message( mi::base::Message_severity level, const char* category, const mi::base::Message_details&, const char* message);
};

void Forwarding_logger::message(
    mi::base::Message_severity level, const char* module_category,
    const mi::base::Message_details& details, const char* message)
{
    // Split module_category into module name and category enum
    string module = "?";
    string category;
    if( module_category) {
        const char* colon = strchr( module_category, ':');
        if (colon) {
            module = string( module_category).substr( 0, colon-module_category);
            category = colon+1;
        } else
            module = module_category;
    }
    LOG::Mod_log::Category category_enum = LOG::Mod_log::C_MAIN; // avoid warning
    if( !convert_string_to_category( category.c_str(), category_enum))
        category_enum = LOG::Mod_log::C_MAIN;

    switch( level) {
        case mi::base::MESSAGE_SEVERITY_FATAL:
            LOG::mod_log->fatal  ( module.c_str(), category_enum, details, "%s", message);
            break;
        case mi::base::MESSAGE_SEVERITY_ERROR:
            LOG::mod_log->error  ( module.c_str(), category_enum, details, "%s", message);
            break;
        case mi::base::MESSAGE_SEVERITY_WARNING:
            LOG::mod_log->warning( module.c_str(), category_enum, details, "%s", message);
            break;
        case mi::base::MESSAGE_SEVERITY_INFO:
            LOG::mod_log->info   ( module.c_str(), category_enum, details, "%s", message);
            break;
        case mi::base::MESSAGE_SEVERITY_VERBOSE:
            LOG::mod_log->vstat  ( module.c_str(), category_enum, details, "%s", message);
            break;
        case mi::base::MESSAGE_SEVERITY_DEBUG:
            LOG::mod_log->debug  ( module.c_str(), category_enum, details, "%s", message);
            break;
        default: // treat invalid enum values as MESSAGE_SEVERITY_ERROR
            LOG::mod_log->error  ( module.c_str(), category_enum, details, "%s", message);
            break;
    }
}

// Register the module.
static SYSTEM::Module_registration<Dist_module_impl> s_module( M_DIST, "DIST");

Module_registration_entry* Dist_module::get_instance() {
    return s_module.init_module( s_module.get_name());
}

bool Dist_module_impl::is_valid_distiller_plugin(
    const char* type, const char* name, const char* filename, const mi::base::Plugin* plugin)
{
    if( !type)
        return false;
    // current version
    if( 0 == strcmp( type, MI_DIST_MDL_DISTILLER_PLUGIN_TYPE)) {
        const mi::mdl::Mdl_distiller_plugin* distiller_plugin = 
            static_cast<const mi::mdl::Mdl_distiller_plugin*>( plugin);
        mi::Size api_version = distiller_plugin->get_api_version();
        if ( api_version == MI_MDL_DISTILLER_PLUGIN_API_VERSION)
            return true;
        // Incompatible API version
        std::stringstream message;
        message << "Plugin \"" << filename << "\" has incompatible API version "
                << api_version << ", but version " << MI_MDL_DISTILLER_PLUGIN_API_VERSION
                << " is required.";
        m_logger->message( mi::base::MESSAGE_SEVERITY_ERROR, 
                           "DISTILLER:COMPILER", message.str().c_str());
        return false;
    }
    return false;
}

bool Dist_module_impl::init() {
    mi::base::Lock::Block block( &m_plugins_lock);
    m_logger = mi::base::make_handle< mi::base::ILogger>(new Forwarding_logger);

    m_plug_module.set();

    // Push the one fixed internal target "none" to the list of known targets
    m_distilling_targets.push_back( string( "none"));

    // Call Mdl_distiller_plugin::init() for all registered distiller plugins
    for( size_t i = 0; true; ++i) {
        mi::base::Handle<mi::base::IPlugin_descriptor> descriptor( m_plug_module->get_plugin(i));
        if ( ! descriptor)
            break;

        mi::base::Plugin* plugin = descriptor->get_plugin();
        const char* type = plugin->get_type();
        const char* name = plugin->get_name();
        const char* filename = descriptor->get_plugin_library_path();

        if( is_valid_distiller_plugin( type, name, filename, plugin)) {
            mi::mdl::Mdl_distiller_plugin* distiller_plugin = static_cast<mi::mdl::Mdl_distiller_plugin*>( plugin);
            distiller_plugin->init( m_logger.get());

            // Store plugins for exit()
            mi::Size plugin_index = m_plugins.size();
            m_plugins.push_back( distiller_plugin);

            // Iterate through all targets and store them in the dist module
            mi::Size target_count = distiller_plugin->get_target_count();
            for ( mi::Size k = 0; k < target_count; ++k) {
                const char* target = distiller_plugin->get_target_name( k);
                if ( target) { // protect against nullptr from the plugin                    
                    bool inserted = m_target_to_index_map.insert( 
                        make_pair( string( target), 
                                   make_pair( plugin_index, k))).second;
                    if ( inserted) {
                        m_distilling_targets.push_back( string( target));
                        std::string message = "Plugin \"";
                        message += filename;
                        message += "\" registered distiller target \"";
                        message += target;
                        message += "\"";
                        m_logger->message( mi::base::MESSAGE_SEVERITY_INFO, 
                                           "DISTILLER:COMPILER", message.c_str());
                    } else {
                        std::string message = "Plugin \"";
                        message += filename;
                        message += "\" failed to register already registered distiller target \"";
                        message += target;
                        message += "\"";
                        m_logger->message( mi::base::MESSAGE_SEVERITY_WARNING, 
                                           "DISTILLER:COMPILER", message.c_str());
                    }
                }
            } 
        }
    }
    return true;
}

void Dist_module_impl::exit() {
    // Call Mdl_distiller_plugin::exit() for all registered distiller plugins
    mi::base::Lock::Block block( &m_plugins_lock);
    Plugin_vector::reverse_iterator it     = m_plugins.rbegin();
    Plugin_vector::reverse_iterator it_end = m_plugins.rend();
    for( ; it != it_end; ++it) {
        (*it)->exit();
    }
    m_plugins.clear();
    m_distilling_targets.clear();
    m_target_to_index_map.clear();
    m_logger = 0;

    m_plug_module.reset();
}

mi::Size Dist_module_impl::get_target_count() const
{
    return m_distilling_targets.size();
}

const char* Dist_module_impl::get_target_name( mi::Size index) const
{
    return index < get_target_count() ? m_distilling_targets[index].c_str() : nullptr;
}


mi::Size Dist_module_impl::get_required_module_count(const char *target) const
{
    if (strcmp( "none", target) != 0) {
        Target_to_index_map::const_iterator it = m_target_to_index_map.find( std::string( target));
        if ( it != m_target_to_index_map.end()) {
            mi::Size plugin_index = it->second.first;
            mi::Size target_index = it->second.second;
            mi::mdl::Mdl_distiller_plugin* distiller_plugin = m_plugins[plugin_index];

            return distiller_plugin->get_required_module_count(target_index);
        }
    }
    return 0;
}

const char* Dist_module_impl::get_required_module_name(const char *target, mi::Size index) const
{
    if (strcmp( "none", target) != 0) {
        Target_to_index_map::const_iterator it = m_target_to_index_map.find( std::string( target));
        if ( it != m_target_to_index_map.end()) {
            mi::Size plugin_index = it->second.first;
            mi::Size target_index = it->second.second;
            mi::mdl::Mdl_distiller_plugin* distiller_plugin = m_plugins[plugin_index];

            return distiller_plugin->get_required_module_name(target_index, index);
        }
    }
    return 0;
}

const char* Dist_module_impl::get_required_module_code(const char *target, mi::Size index) const
{
    if (strcmp( "none", target) != 0) {
        Target_to_index_map::const_iterator it = m_target_to_index_map.find( std::string( target));
        if ( it != m_target_to_index_map.end()) {
            mi::Size plugin_index = it->second.first;
            mi::Size target_index = it->second.second;
            mi::mdl::Mdl_distiller_plugin* distiller_plugin = m_plugins[plugin_index];

            return distiller_plugin->get_required_module_code(target_index, index);
        }
    }
    return 0;
}



} // namespace DIST
} // namespace MI
