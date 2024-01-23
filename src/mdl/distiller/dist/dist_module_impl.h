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

/// \file
/// \brief Implementation class of the DIST module

#ifndef MDL_DISTILLER_DIST_DIST_MODULE_IMPL_H
#define MDL_DISTILLER_DIST_DIST_MODULE_IMPL_H

#include "i_dist.h"
#include <mi/mdl/mdl_distiller_plugin.h>

#include <mi/base/lock.h>

#include <utility>
#include <vector>
#include <unordered_map>
#include <base/system/main/access_module.h>

namespace MI {

namespace PLUG { class Plug_module; }

namespace DIST {

/// Implementation class of the DIST module
class Dist_module_impl : public Dist_module
{
public:
    // methods of SYSTEM::IModule
    bool init();

    void exit();

    // Additional methods
    /// Returns the number of available distilling targets.
    virtual mi::Size get_target_count() const;

    /// Returns the name of the distilling target at index position.
    virtual const char* get_target_name( mi::Size index) const;

    /// Main function to distill an MDL material.
    virtual const mi::mdl::IGenerated_code_dag::IMaterial_instance* distill(
        mi::mdl::ICall_name_resolver& call_resolver,
        mi::mdl::IRule_matcher_event* event_handler,
        const mi::mdl::IGenerated_code_dag::IMaterial_instance* material_instance,
        const char* target,
        mi::mdl::Distiller_options* options,
        mi::Sint32* error) const;

    /// Returns the number of required MDL modules for the given
    /// target.
    virtual mi::Size get_required_module_count(const char *target) const;

    /// Returns the name of the required MDL module with the given
    /// index for the given target.
    virtual const char* get_required_module_name(const char *target, mi::Size index) const;

    /// Returns the MDL source code of the required MDL module with
    /// the given index for the given target.
    virtual const char* get_required_module_code(const char *target, mi::Size index) const;

private:
    /// Check for valid distiller plugin
    ///
    /// \param type        Type of the plugin, must be MI_DIST_MDL_DISTILLER_PLUGIN_TYPE.
    /// \param name        Name of the plugin, for diagnostics.
    /// \param filename    Filename of the DSO, for diagnostics.
    /// \return            \c true if the plugin type is known and supported, \c false otherwise.
    ///                    Logs an error for unsupported distiller plugin types.
    bool is_valid_distiller_plugin( 
        const char* type, const char* name, const char* filename, const mi::base::Plugin* plugin);

    /// Access to the PLUG module
    SYSTEM::Access_module<PLUG::Plug_module> m_plug_module;

    /// Logger interface
    mi::base::Handle< mi::base::ILogger> m_logger;

    /// Lock for #m_plugins.
    mutable mi::base::Lock m_plugins_lock;

    /// The registered distiller plugins. Needs #m_plugins_lock.
    typedef std::vector< mi::mdl::Mdl_distiller_plugin* > Plugin_vector;
    Plugin_vector m_plugins;

    /// The sequence of all registered distilling targets
    std::vector< std::string> m_distilling_targets;

    /// The map of distilling targest to the respective plugin and the index
    /// in that plugin to call
    typedef std::unordered_map< std::string, std::pair<mi::Size, mi::Size> > Target_to_index_map;
    Target_to_index_map m_target_to_index_map;
};

} // namespace DIST
} // namespace MI

#endif // MDL_DISTILLER_DIST_DIST_MODULE_IMPL_H
