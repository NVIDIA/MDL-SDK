/******************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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

/// \file mi/mdl/mdl_distiller_plugin.h
/// \brief MDL distiller plugin base class

#ifndef MDL_DISTILLER_PLUGIN_H
#define MDL_DISTILLER_PLUGIN_H

#include <mi/base/interface_declare.h>
#include <mi/base/plugin.h>
#include <mi/base/types.h>
#include <mi/base/ilogger.h>

#include <mi/mdl/mdl_generated_dag.h>
#include <mi/mdl/mdl_distiller_plugin_api.h>
#include <mi/mdl/mdl_distiller_options.h>

namespace mi { namespace neuraylib { class IPlugin_api; } }
namespace mi { namespace mdl { class IRule_matcher_event; } }

namespace mi {
namespace mdl {

/// Type of MDL distiller plugins
#define MI_DIST_MDL_DISTILLER_PLUGIN_TYPE "mdl_distiller v2"

/// Abstract interface for MDL distiller plugins.
class Mdl_distiller_plugin : public mi::base::Plugin {
public:
    /// Returns the name of the plugin.
    ///
    /// \note This method from #mi::base::Plugin is repeated here only for documentation purposes.
    virtual const char* get_name() const = 0;

    /// Return the distiller plugin API version. A plugin is only accepted if it is compiled
    /// against the same API version than the SDK.
    virtual mi::Size get_api_version() const = 0;

    /// Initializes the plugin.
    ///
    /// \return      \c true in case of success, and \c false otherwise.
    virtual bool init( mi::base::ILogger* logger) = 0;

    /// De-initializes the plugin.
    ///
    /// \return      \c true in case of success, and \c false otherwise.
    virtual bool exit() = 0;

    /// Returns the number of available distilling targets.
    virtual mi::Size get_target_count() const = 0;

    /// Returns the name of the distilling target at index position.
    virtual const char* get_target_name( mi::Size index) const = 0;

    /// Main function to distill an MDL material.
    ///
    /// Uses a DAG material instance as input, applies selected rule sets and returns the result
    /// as a new DAG material instance.
    /// The mdl module \::nvidia::distilling_support is loaded before calling this function.
    ///
    /// \param api               The MDL distiller plugin API to manipulate the DAG,
    ///                          and also the API against which the mdltlc generates code
    /// \param event_handler     If non-NULL, an event handler interface used to report events
    ///                          during processing.
    /// \param material_instance The material instance to "distill".
    /// \param target_index      The index of the distilling target model
    /// \param options           The Distiller options
    /// \param p_error           An optional pointer to an #mi::Sint32 to which an error code will
    ///                          be written. The error codes have the following meaning:
    ///                          -  0: Success.
    ///                          - -1: Reserved for API layer.
    ///                          - -2: Reserved for API layer.
    ///                          - -3: Unspecified failure.
    /// \return                  The distilled material instance, or \c NULL in case of failure.
    virtual const mi::mdl::IGenerated_code_dag::IMaterial_instance* distill(
        mi::mdl::IDistiller_plugin_api& api,
        mi::mdl::IRule_matcher_event* event_handler,
        const mi::mdl::IGenerated_code_dag::IMaterial_instance* material_instance,
        mi::Size target_index,
        Distiller_options* options,
        mi::Sint32* p_error) const = 0;

    /// Returns the number of modules that are required by result
    /// materials of this Distiller plugin.
    ///
    /// \param target_index   The index of the distilling target model.
    ///
    /// \return               The number of required MDL modules for the target model.
    virtual mi::Size get_required_module_count(mi::Size target_index) const = 0;

    /// Returns the MDL source code of the required module at the given
    /// index. This can be used to load the module, also using the
    /// corresponding module name (see get_required_module_name()).
    ///
    /// \param target_index   The index of the distilling target model.
    /// \param module_index   The index of the module for the target model.
    ///
    /// \return               The code of the required module.
    virtual const char *get_required_module_code(mi::Size target_index, mi::Size module_index) const = 0;

    /// Returns the fully qualified name of the required module at the
    /// given index. This can be used to load the module, using the
    /// corresponding module code (see get_required_module_code()).
    ///
    /// \param target_index   The index of the distilling target model.
    /// \param module_index   The index of the module for the target model.
    ///
    /// \return               The fully qualified name of the required module.
    virtual const char *get_required_module_name(mi::Size target_index, mi::Size module_index) const = 0;
};

} // namespace mdl

} // namespace mi

#endif // MDL_DISTILLER_PLUGIN_H
