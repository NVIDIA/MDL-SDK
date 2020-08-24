/***************************************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief Header for the IMdl_module_transformer implementation.
 **/

#ifndef API_API_NEURAY_MDL_MODULE_TRANSFORMER_IMPL_H
#define API_API_NEURAY_MDL_MODULE_TRANSFORMER_IMPL_H

#include <mi/base/interface_implement.h>
#include <mi/neuraylib/imdl_module_transformer.h>

#include <memory>
#include <string>

#include <mi/base/handle.h>
#include <boost/core/noncopyable.hpp>

namespace mi {
namespace mdl { class IModule; }
namespace neuraylib { class ITransaction; }
}

namespace MI {

namespace MDL { class Mdl_module_transformer; }

namespace NEURAY {

class Mdl_module_transformer_impl
  : public mi::base::Interface_implement<mi::neuraylib::IMdl_module_transformer>,
    public boost::noncopyable
{
public:
    Mdl_module_transformer_impl(
        mi::neuraylib::ITransaction* transaction,
        const char* module_name,
        const mi::mdl::IModule* mdl_module);

    ~Mdl_module_transformer_impl();

    // public API methods

    mi::Sint32 upgrade_mdl_version(
        mi::neuraylib::Mdl_version version, mi::neuraylib::IMdl_execution_context* context) final;

    mi::Sint32 use_absolute_import_declarations(
        const char* include_filter,
        const char* exclude_filter,
        mi::neuraylib::IMdl_execution_context* context) final;

    mi::Sint32 use_relative_import_declarations(
        const char* include_filter,
        const char* exclude_filter,
        mi::neuraylib::IMdl_execution_context* context) final;

    mi::Sint32 use_absolute_resource_file_paths(
        const char* include_filter,
        const char* exclude_filter,
        mi::neuraylib::IMdl_execution_context* context) final;

    mi::Sint32 use_relative_resource_file_paths(
        const char* include_filter,
        const char* exclude_filter,
        mi::neuraylib::IMdl_execution_context* context) final;

    mi::Sint32 inline_imported_modules(
        const char* include_filter,
        const char* exclude_filter,
        bool omit_anno_origin,
        mi::neuraylib::IMdl_execution_context* context) final;

    mi::Sint32 export_module(
        const char* filename, mi::neuraylib::IMdl_execution_context* context) final;

    mi::Sint32 export_module_to_string(
        mi::IString* exported_module, mi::neuraylib::IMdl_execution_context* context) final;

private:
    mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;

    std::string m_db_module_name;

    std::unique_ptr<MDL::Mdl_module_transformer> m_impl;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_MDL_MODULE_TRANSFORMER_IMPL_H
