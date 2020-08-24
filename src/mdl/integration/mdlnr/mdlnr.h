/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_INTERGRATION_MDLNR_H
#define MDL_INTERGRATION_MDLNR_H 1

#include <mdl/integration/mdlnr/i_mdlnr.h>

#include <mi/base/handle.h>

#include <vector>
#include <base/system/main/access_module.h>

namespace mi { namespace base { class IAllocator; class IPlugin_descriptor; } }

namespace MI {

namespace PLUG { class Plug_module; }

namespace MDLC {

/// The implementation of the MDL compiler module.
class Mdlc_module_impl : public Mdlc_module
{

public:

    /// Constructor.
    Mdlc_module_impl();

    bool init();

    void exit();

    mi::mdl::IMDL *get_mdl() const;

    void serialize_module(SERIAL::Serializer *serializer, const mi::mdl::IModule *module);

    const mi::mdl::IModule *deserialize_module(SERIAL::Deserializer *deserializer);

    void serialize_code_dag(
        SERIAL::Serializer *serializer, const mi::mdl::IGenerated_code_dag *code);

    const mi::mdl::IGenerated_code_dag *deserialize_code_dag(
        SERIAL::Deserializer *deserializer);

    void serialize_lambda_function(
        SERIAL::Serializer *serializer, const mi::mdl::ILambda_function *lambda);

    mi::mdl::ILambda_function *deserialize_lambda_function(
        SERIAL::Deserializer *deserializer);

    void client_build_version(const char* build, const char* bridge_protocol) const;

    mi::mdl::ICode_cache *get_code_cache() const;

    bool utf8_match(char const *file_mask, char const *file_name) const;

    void set_implicit_cast_enabled(bool value);

    bool get_implicit_cast_enabled() const;

    void set_expose_names_of_let_expressions(bool value);

    bool get_expose_names_of_let_expressions() const;

    MDL::Mdl_module_wait_queue* get_module_wait_queue() const;

private:

    /// Helper function to detect valid MDL core plugin type names.
    ///
    /// Returns \c true if the plugin type is known and supported.
    /// Prints a suitable warning message for known supported but outdated plugins and an
    /// error message for
    ///
    /// \param type        Type of the plugin, should be MI_MDL_CORE_PLUGIN_TYPE.
    /// \param name        Name of the plugin, for diagnostics.
    /// \param filename    Filename of the DSO, for diagnostics.
    /// \return            \c true if the plugin type is known and supported, \c false otherwise.
    ///                    Logs a warning for supported but outdated MDL core plugin types, and an
    ///                    error for unsupported MDL core plugin types.
    bool is_valid_mdl_core_plugin(
       const char* type, const char* name, const char* filename);
    
    /// Pointer to the MDL interface.
    mi::mdl::IMDL *m_mdl;

    /// The allocator that wraps new/delete (overloaded in MI::MEM) as IAllocator.
    mi::base::Handle<mi::base::IAllocator> m_allocator;

    /// The code cache used for JIT-generated source code.
    mi::mdl::ICode_cache *m_code_cache;

    /// Flag that indicates whether the integration should insert casts when needed (and possible).
    bool m_implicit_cast_enabled;

    /// Flag that indicates whether the integration should insert casts when needed (and possible).
    bool m_expose_names_of_let_expressions;

    /// The module wait queue.
    MDL::Mdl_module_wait_queue *m_module_wait_queue;

    /// Access to the PLUG module
    SYSTEM::Access_module<PLUG::Plug_module> m_plug_module;

    /// Lock for #m_plugins.
    mutable mi::base::Lock m_plugins_lock;

    typedef std::vector<mi::base::Handle<mi::base::IPlugin_descriptor> > Plugin_vector;

    /// The registered image plugins. Needs #m_plugins_lock.
    Plugin_vector m_plugins;
};

} // namespace MDLC

} // namespace MI

#endif // MDL_INTERGRATION_MDLNR_H
