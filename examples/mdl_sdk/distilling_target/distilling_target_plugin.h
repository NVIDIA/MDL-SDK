/******************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_DISTILLER_DISTILLING_TARGET_PLUGIN_H
#define MDL_DISTILLER_DISTILLING_TARGET_PLUGIN_H

#include "mdl_assert.h"

#include <mi/base/ilogger.h>
#include <mi/mdl/mdl_generated_dag.h>
#include <mi/mdl/mdl_distiller_plugin_api.h>
#include <mi/mdl/mdl_distiller_plugin.h>

#define MI_DISTILLER_PLUGIN_NAME "distilling_target_plugin"

namespace MI {
namespace DIST {

class Distilling_target_plugin : mi::mdl::Mdl_distiller_plugin {
public:
    Distilling_target_plugin() { }

    virtual ~Distilling_target_plugin() { }

    // methods of mi::base::Plugin

    const char* get_name() const { return MI_DISTILLER_PLUGIN_NAME; }

    const char* get_type() const { return MI_DIST_MDL_DISTILLER_PLUGIN_TYPE; }

    mi::Sint32  get_version() const { return 1; }

    const char* get_compiler() const { return "unknown"; }

    void release() { delete this; }

    // methods of MI::DIST::Mdl_distiller_plugin

    mi::Size get_api_version() const { return MI_MDL_DISTILLER_PLUGIN_API_VERSION; }

    bool init( mi::base::ILogger* logger);

    bool exit();

    /// Returns the number of available distilling targets.
    mi::Size get_target_count() const;

    /// Returns the name of the distilling target at index position.
    const char* get_target_name( mi::Size index) const;

    /// Main function to distill an MDL material.
    ///
    /// Uses a DAG material instance as input, applies selected rule sets and returns the result
    /// as a new DAG material instance.
    /// The mdl module ::nvidia::distilling_support is loaded before calling this function.
    ///
    /// \param engine            The MDL distiller plugin API to manipulate the DAG,
    ///                          and also the API against which the mdltlc generates code
    /// \param event_handler     If non-NULL, an event handler interface used to report events
    ///                          during processing.
    /// \param material_instance The material instance to "distill".
    /// \param target_index      The index of the distilling target model
    //  \param options           Options with some parameters controlling details in the rules
    /// \param error             An optional pointer to an #mi::Sint32 to which an error code will
    ///                          be written. The error codes have the following meaning:
    ///                          -  0: Success.
    ///                          - -1: Reserved for API layer.
    ///                          - -2: Reserved for API layer.
    ///                          - -3: Unspecified failure.
    /// \return                  The distilled material instance, or \c NULL in case of failure.
    const mi::mdl::IMaterial_instance* distill(
        mi::mdl::IDistiller_plugin_api&    api,
        mi::mdl::IRule_matcher_event*      event_handler,
        const mi::mdl::IMaterial_instance* material_instance,
        mi::Size                           target_index,
        mi::mdl::Distiller_options*        options,
        mi::Sint32*                        error) const;

    mi::Size get_required_module_count(mi::Size target_index) const;

    const char *get_required_module_code(mi::Size target_index, mi::Size index) const;

    const char *get_required_module_name(mi::Size target_index, mi::Size index) const;
};

} // namespace DIST

} // namespace MI

#endif // MDL_DISTILLER_DISTILLING_TARGET_PLUGIN_H
