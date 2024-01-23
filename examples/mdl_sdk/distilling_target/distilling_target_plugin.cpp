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

#include "pch.h"

#include <string>
#include <iostream>

#include <mi/base/ilogger.h>

#include "distilling_target_plugin.h"

// This header is generated from distilling_target_plugin_rules.mdltl
// and contains the declaration of the matcher classes that are used
// in method 'distill()' below.
#include "distilling_target_plugin_rules.h"

namespace MI {
namespace DIST {

// Supported distilling targets in this plugin
// Note: these need to match the switch( target_index) statement below
static const char* s_targets[] = {
    "mini_glossy"
};

/// Returns the dimension of an array.
template<typename T, size_t n>
inline size_t dimension_of(T (&c)[n]) { return n; }


// Global logger
mi::base::Handle<mi::base::ILogger> g_logger;

// Support function to log messages. Skips if no logger is available
void log( mi::base::Message_severity severity, const char* message)
{
    if( g_logger.is_valid_interface())
        g_logger->message( severity, "DISTILLER:COMPILER", message);
}

bool Distilling_target_plugin::init( mi::base::ILogger* logger) {
    g_logger = mi::base::make_handle_dup(logger);

    std::string message = "Plugin \"";
    message += MI_DISTILLER_PLUGIN_NAME;
    message += "\" initialized";
    log( mi::base::MESSAGE_SEVERITY_INFO, message.c_str());

    return true;
}

bool Distilling_target_plugin::exit() {
    g_logger = 0;
    return true;
}

/// Returns the number of available distilling targets.
mi::Size Distilling_target_plugin::get_target_count() const {
    return dimension_of( s_targets);
}

/// Returns the name of the distilling target at index position.
const char* Distilling_target_plugin::get_target_name( mi::Size index) const {
    return (index < dimension_of( s_targets)) ? s_targets[index] : nullptr;
}

mi::Size Distilling_target_plugin::get_required_module_count(mi::Size target_index) const {
    return 0;
}

const char *Distilling_target_plugin::get_required_module_code(mi::Size target_index, mi::Size index) const {
    return nullptr;
}

const char *Distilling_target_plugin::get_required_module_name(mi::Size target_index, mi::Size index) const {
    return nullptr;
}

/// Main function to distill an MDL material.
const mi::mdl::IGenerated_code_dag::IMaterial_instance* Distilling_target_plugin::distill(
    mi::mdl::IDistiller_plugin_api&                         api,
    mi::mdl::IRule_matcher_event*                           event_handler,
    const mi::mdl::IGenerated_code_dag::IMaterial_instance* material_instance,
    mi::Size                                                target_index,
    mi::mdl::Distiller_options*                             options,
    mi::Sint32*                                             p_error) const
{
    // Switch from error pointer to reference to simplify later code for the case of p_error == 0.
    mi::Sint32 dummy;
    mi::Sint32 &error = p_error != NULL ? *p_error : dummy;
    error = 0;

    // Check the preconditions on the input parameters
    if ( (!material_instance) || (!options) || (target_index >= get_target_count())) {
        error = -3;
        return nullptr;
    }

    mi::base::Handle<mi::mdl::IGenerated_code_dag::IMaterial_instance const> res;

#define CHECK_RESULT  if(error != 0) { return NULL; }

    // Note: the case numbers must match the cardinal order of targets in s_targets above
    switch ( target_index) {
    case 0:  // "mini_glossy"
    {
        log(mi::base::MESSAGE_SEVERITY_INFO, "Distilling to target 'mini_glossy'.");

        res = mi::base::make_handle_dup(material_instance);

        Make_simple_rules make_simple;
        res = api.apply_rules( res.get(), make_simple, event_handler, options, error);
        CHECK_RESULT;

        break;
    }

    } // end switch

#undef CHECK_RESULT

    if (res.is_valid_interface()) {
        res->retain();
        return res.get();
    }

    error = -3;
    return nullptr;
}


// Factory to create an instance of CUstom_distilling_target.
extern "C"
MI_DLL_EXPORT
//mi::base::Plugin* mi_plugin_factory(
void* mi_plugin_factory(
    mi::Sint32 index,         // Index of the plugin. We only allow index 0.
    void* context)            // Context given to the library, distiller plugins can ignore it.
{
#if 1
    if( index > 0)
        return 0;
    return new Distilling_target_plugin();
#else
    return 0;
#endif // 0
}

} // namespace DIST
} // namespace MI
