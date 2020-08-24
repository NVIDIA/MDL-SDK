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

#include "pch.h"

#include <cstring>
#include <string>

#include <mi/base/types.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_symbols.h>
#include <mi/mdl/mdl_types.h>
#include <base/data/db/i_db_access.h>
#include <io/scene/mdl_elements/mdl_elements_detail.h> // DETAIL::Type_binder
#include <io/scene/mdl_elements/i_mdl_elements_compiled_material.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_call.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_definition.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>
#include <render/mdl/runtime/i_mdlrt_resource_handler.h>
#include <render/mdl/backends/backends_target_code.h>
#include <api/api/neuray/neuray_compiled_material_impl.h>
#include <api/api/neuray/neuray_function_call_impl.h>
#include <api/api/neuray/neuray_transaction_impl.h>
#include <api/api/neuray/neuray_mdl_execution_context_impl.h>

#include "neuray_mdl_backend_impl.h"

namespace MI {

namespace NEURAY {

static DB::Transaction *unwrap(mi::neuraylib::ITransaction *transaction)
{
    Transaction_impl *transaction_impl =
        static_cast<Transaction_impl *>(transaction);
    return transaction_impl->get_db_transaction();
}

static MDL::Mdl_function_call const *unwrap(mi::neuraylib::IFunction_call const *function_call)
{
    Function_call_impl const *function_call_impl =
        static_cast<Function_call_impl const *>(function_call);
     return function_call_impl->get_db_element();
}

static MDL::Mdl_compiled_material const *unwrap(
    mi::neuraylib::ICompiled_material const *compiled_material)
{
    Compiled_material_impl const *compiled_material_impl =
        static_cast<Compiled_material_impl const *>(compiled_material);
     return compiled_material_impl->get_db_element();
}

static BACKENDS::Link_unit const *unwrap(mi::neuraylib::ILink_unit const *lu)
{
    Link_unit const *lu_impl =
        static_cast<Link_unit const *>(lu);
    return lu_impl->get_link_unit();
}

static MDL::Execution_context *unwrap_and_clear(mi::neuraylib::IMdl_execution_context *context)
{
    Mdl_execution_context_impl *context_impl =
        static_cast<Mdl_execution_context_impl *>(context);
    if (context_impl) {
        MDL::Execution_context& wrapped_context = context_impl->get_context();
        wrapped_context.clear_messages();
        return &wrapped_context;
    }

    return nullptr;
}

// Constructor from an LLVM backend.
Link_unit::Link_unit(
    BACKENDS::Mdl_llvm_backend  &be,
    mi::neuraylib::ITransaction *transaction,
    mi::neuraylib::IMdl_execution_context* context)
: m_link_unit(be, unwrap(transaction), unwrap_and_clear(context))
{
}


// Add an MDL environment function call as a function to this link unit.
mi::Sint32 Link_unit::add_environment(
    mi::neuraylib::IFunction_call const *call,
    char const                          *fname,
    mi::neuraylib::IMdl_execution_context* context)
{
    return m_link_unit.add_environment(unwrap(call), fname, unwrap_and_clear(context));
}

// Add an expression that is part of an MDL material instance as a function to this link unit.
mi::Sint32 Link_unit::add_material_expression(
    mi::neuraylib::ICompiled_material const *material,
    char const                              *path,
    char const                              *fname,
    mi::neuraylib::IMdl_execution_context   *context)
{
    return m_link_unit.add_material_expression(unwrap(material), path, fname, unwrap_and_clear(context));
}

// Add an MDL distribution function to this link unit.
mi::Sint32 Link_unit::add_material_df(
    mi::neuraylib::ICompiled_material const *material,
    char const                              *path,
    char const                              *base_fname,
    mi::neuraylib::IMdl_execution_context   *context)
{
    return m_link_unit.add_material_df(
        unwrap(material), path, base_fname, unwrap_and_clear(context));
}

mi::Sint32 Link_unit::add_material(
    mi::neuraylib::ICompiled_material const    *material,
    mi::neuraylib::Target_function_description *function_descriptions,
    mi::Size                                    description_count,
    mi::neuraylib::IMdl_execution_context      *context)
{
    return m_link_unit.add_material(
        unwrap(material),
        function_descriptions,
        static_cast<size_t>(description_count),
        unwrap_and_clear(context));
}

Mdl_llvm_backend::Mdl_llvm_backend(
    mi::neuraylib::IMdl_backend_api::Mdl_backend_kind kind,
    mi::mdl::IMDL                *compiler,
    mi::mdl::ICode_generator_jit *jit,
    mi::mdl::ICode_cache         *code_cache,
    bool                         string_ids)
: m_backend(kind, compiler, jit, code_cache, string_ids)
{
}

mi::Sint32 Mdl_llvm_backend::set_option(char const *name, char const *value)
{
    return m_backend.set_option(name, value);
}

mi::Sint32 Mdl_llvm_backend::set_option_binary(char const *name, const char* data, mi::Size size)
{
    return m_backend.set_option_binary(name, data, size);
}

mi::neuraylib::ITarget_code const *Mdl_llvm_backend::translate_environment(
    mi::neuraylib::ITransaction           *transaction,
    mi::neuraylib::IFunction_call const   *function_call,
    char const                            *fname,
    mi::neuraylib::IMdl_execution_context *context)
{
    MDL::Execution_context* context_impl = unwrap_and_clear(context);

    if (transaction == nullptr || function_call == nullptr) {
        add_error_message(context_impl, "Invalid parameters (NULL pointer).", -1);
        return nullptr;
    }

    DB::Transaction *db_transaction = unwrap(transaction);
    MDL::Mdl_function_call const *db_function_call = unwrap(function_call);

    return m_backend.translate_environment(
        db_transaction,
        db_function_call,
        fname,
        context_impl);
}

mi::neuraylib::ITarget_code const *Mdl_llvm_backend::translate_material_expression(
    mi::neuraylib::ITransaction             *transaction,
    mi::neuraylib::ICompiled_material const *compiled_material,
    char const                              *path,
    char const                              *fname,
    mi::neuraylib::IMdl_execution_context   *context)
{

    MDL::Execution_context* context_impl = unwrap_and_clear(context);

    if (transaction == nullptr || compiled_material == nullptr) {
        add_error_message(context_impl, "Invalid parameters (NULL pointer).", -1);
        return nullptr;
    }

    return m_backend.translate_material_expression(
        unwrap(transaction), unwrap(compiled_material), path, fname, context_impl);
}

const mi::neuraylib::ITarget_code* Mdl_llvm_backend::translate_material_df(
    mi::neuraylib::ITransaction* transaction,
    const mi::neuraylib::ICompiled_material* compiled_material,
    const char* path,
    const char* base_fname,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context* context_impl = unwrap_and_clear(context);

    if (!transaction || !compiled_material || !path) {
        add_error_message(context_impl, "Invalid parameters (NULL pointer).", -1);
        return nullptr;
    }

    return m_backend.translate_material_df(
        unwrap(transaction), unwrap(compiled_material), path, base_fname, context_impl);
}

const mi::neuraylib::ITarget_code* Mdl_llvm_backend::translate_material(
    mi::neuraylib::ITransaction* transaction,
    const mi::neuraylib::ICompiled_material* material,
    mi::neuraylib::Target_function_description* function_descriptions,
    mi::Size description_count,
    mi::neuraylib::IMdl_execution_context* context)
{
    MDL::Execution_context* context_impl = unwrap_and_clear(context);

    // reuse link unit based implementation
    BACKENDS::Link_unit link_unit(m_backend, unwrap(transaction), context_impl);

    if (link_unit.add_material(
            unwrap(material),
            function_descriptions,
            static_cast<size_t>(description_count),
            context_impl) != 0)
    {
        return nullptr;
    }

    return m_backend.translate_link_unit(&link_unit, context_impl);
}

mi::Uint8 const *Mdl_llvm_backend::get_device_library(Size &size) const
{
    return m_backend.get_device_library(size);
}

mi::neuraylib::ILink_unit* Mdl_llvm_backend::create_link_unit(
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_execution_context* context)
{
    if (context)
        context->clear_messages();

    return new Link_unit(m_backend, transaction, context);
}

mi::neuraylib::ITarget_code const* Mdl_llvm_backend::translate_link_unit(
    mi::neuraylib::ILink_unit const* lu,
    mi::neuraylib::IMdl_execution_context* context)
{
    return m_backend.translate_link_unit(unwrap(lu), unwrap_and_clear(context));
}


const mi::neuraylib::ITarget_code* Mdl_llvm_backend::deserialize_target_code(
    const mi::neuraylib::IBuffer* buffer,
    mi::neuraylib::IMdl_execution_context* context) const
{
    if (context)
        context->clear_messages();

    mi::base::Handle<BACKENDS::Target_code> info(new BACKENDS::Target_code());
    mi::base::Handle<mi::mdl::ICode_generator> code_gen(m_backend.get_jit_be());

    if (!info->deserialize(code_gen.get(), buffer, context))
        return nullptr;

    // make sure the info is of the same back-end kind
    if (info->get_backend_kind() != m_backend.get_kind()) {
        context->add_message(mi::neuraylib::IMessage::MSG_COMILER_BACKEND,
            mi::base::details::MESSAGE_SEVERITY_ERROR, -1,
            "Deserialization failed. The deserialized object was created by different kind of "
            "back-end.");
        return nullptr;
    }

    info->retain();
    return info.get();
}

namespace {

    /// Wraps a memory block identified by a pointer and a length as mi::neuraylib::IBuffer.
    /// Does not copy the data.
    class Buffer_wrapper
        : public mi::base::Interface_implement<mi::neuraylib::IBuffer>,
        public boost::noncopyable
    {
    public:
        Buffer_wrapper(const mi::Uint8* data, mi::Size data_size)
            : m_data(data), m_data_size(data_size) { }
        const mi::Uint8* get_data() const { return m_data; }
        mi::Size get_data_size() const { return m_data_size; }
    private:
        const mi::Uint8* m_data;
        const mi::Size m_data_size;
    };

} // anonymous namespace

const mi::neuraylib::ITarget_code* Mdl_llvm_backend::deserialize_target_code(
    const mi::Uint8* buffer_data,
    mi::Size buffer_size,
    mi::neuraylib::IMdl_execution_context* context) const
{
    Buffer_wrapper buffer(buffer_data, buffer_size);
    return deserialize_target_code(&buffer, context);
}


} // namespace NEURAY

} // namespace MI

