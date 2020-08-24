/***************************************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \file
/// \brief

#ifndef API_API_NEURAY_MDL_BACKEND_H
#define API_API_NEURAY_MDL_BACKEND_H

#include <string>
#include <vector>
#include <map>

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/neuraylib/ibuffer.h>
#include <mi/neuraylib/imdl_backend_api.h>
#include <mi/mdl/mdl_code_generators.h>
#include <render/mdl/backends/backends_backends.h>
#include <render/mdl/backends/backends_link_unit.h>

namespace mi {
namespace neuraylib { class ICompiled_material; class Function_call; }
namespace mdl { class IMDL; class IType_struct; class IType; }
}

namespace MI {

namespace DB { class Transaction; }

namespace NEURAY {


/// Implementation of #mi::neuraylib::IMdl_backend for LLVM-IR based backends.
class Mdl_llvm_backend : public mi::base::Interface_implement<mi::neuraylib::IMdl_backend>
{
public:

    /// Constructor.
    ///
    /// \param kind            The backend kind.
    /// \param compiler        The MDL compiler.
    /// \param jit             The JIT code generator.
    /// \param code_cache      If non-NULL, the code cache.
    /// \param string_ids      If True, string arguments are mapped to string identifiers.
    Mdl_llvm_backend(
        mi::neuraylib::IMdl_backend_api::Mdl_backend_kind kind,
        mi::mdl::IMDL* compiler,
        mi::mdl::ICode_generator_jit* jit,
        mi::mdl::ICode_cache *code_cache,
        bool string_ids);

    // API methods

    mi::Sint32 set_option( const char* name, const char* value) final;

    mi::Sint32 set_option_binary( const char* name, const char* data, mi::Size size) final;

    const mi::neuraylib::ITarget_code* translate_environment(
        mi::neuraylib::ITransaction* transaction,
        const mi::neuraylib::IFunction_call* call,
        const char* fname,
        mi::neuraylib::IMdl_execution_context* context) final;

    const mi::neuraylib::ITarget_code* translate_material_expression(
        mi::neuraylib::ITransaction* transaction,
        const mi::neuraylib::ICompiled_material* material,
        const char* path,
        const char* fname,
        mi::neuraylib::IMdl_execution_context* context) final;

    const mi::Uint8* get_device_library( mi::Size &size) const final;

    const mi::neuraylib::ITarget_code* translate_material_df(
        mi::neuraylib::ITransaction* transaction,
        const mi::neuraylib::ICompiled_material* material,
        const char* path,
        const char* base_fname,
        mi::neuraylib::IMdl_execution_context* context) final;

    const mi::neuraylib::ITarget_code* translate_material(
        mi::neuraylib::ITransaction* transaction,
        const mi::neuraylib::ICompiled_material* material,
        mi::neuraylib::Target_function_description* function_descriptions,
        mi::Size description_count,
        mi::neuraylib::IMdl_execution_context* context) final;

    mi::neuraylib::ILink_unit* create_link_unit(
        mi::neuraylib::ITransaction* transaction,
        mi::neuraylib::IMdl_execution_context* context) final;

    const mi::neuraylib::ITarget_code* translate_link_unit(
        mi::neuraylib::ILink_unit const* lu,
        mi::neuraylib::IMdl_execution_context* context) final;

    const mi::neuraylib::ITarget_code* deserialize_target_code(
        const mi::neuraylib::IBuffer* buffer,
        mi::neuraylib::IMdl_execution_context* context) const final;

    const mi::neuraylib::ITarget_code* deserialize_target_code(
        const mi::Uint8* buffer_data,
        mi::Size buffer_size,
        mi::neuraylib::IMdl_execution_context* context) const final;

private:
    /// Get the internal backend.
    BACKENDS::Mdl_llvm_backend &get_backend() { return m_backend; };

private:
    BACKENDS::Mdl_llvm_backend m_backend;
};


/// Implementation of #mi::neuraylib::ILink_unit interface.
class Link_unit : public mi::base::Interface_implement<mi::neuraylib::ILink_unit>
{
public:

    /// Constructor from an LLVM backend.
    ///
    /// \param be           The backend.
    /// \param transaction  The current transaction.
    /// \param context      Context used to pass compilation options to the code generator.
    Link_unit(
        BACKENDS::Mdl_llvm_backend            &be,
        mi::neuraylib::ITransaction           *transaction,
        mi::neuraylib::IMdl_execution_context *context);


    // API methods

    /// Add an MDL environment function call as a function to this link unit.
    /// (see #mi::neuraylib::ILink_unit::add_environment for details)
    mi::Sint32 add_environment(
        mi::neuraylib::IFunction_call const     *call,
        char const                              *fname,
        mi::neuraylib::IMdl_execution_context   *context) final;

    /// Add an expression that is part of an MDL material instance as a function to this
    /// (see #mi::neuraylib::ILink_unit::add_material_expression for details)
    mi::Sint32 add_material_expression(
        mi::neuraylib::ICompiled_material const *material,
        char const                              *path,
        char const                              *fname,
        mi::neuraylib::IMdl_execution_context   *context) final;

    /// Add an MDL distribution function to this link unit.
    /// (see #mi::neuraylib::ILink_unit::add_material_df for details)
    mi::Sint32 add_material_df(
        mi::neuraylib::ICompiled_material const *material,
        char const                              *path,
        char const                              *base_fname,
        mi::neuraylib::IMdl_execution_context   *context) final;

    /// Add (multiple) MDL distribution functions and expressions of a material to this link unit.
    /// (see #mi::neuraylib::ILink_unit::add_material for details)
    mi::Sint32 add_material(
        mi::neuraylib::ICompiled_material const    *material,
        mi::neuraylib::Target_function_description *function_descriptions,
        mi::Size                                    description_count,
        mi::neuraylib::IMdl_execution_context      *context) final;

    BACKENDS::Link_unit const *get_link_unit() const { return &m_link_unit; }

private:
    BACKENDS::Link_unit m_link_unit;
};


} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_MDL_BACKEND_H

