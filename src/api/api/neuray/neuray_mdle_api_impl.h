/***************************************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Header for the IMdle_api implementation.
 **/

#ifndef API_API_NEURAY_MDLE_API_IMPL_H
#define API_API_NEURAY_MDLE_API_IMPL_H

#include <mi/base/interface_implement.h>
#include <mi/neuraylib/imdle_api.h>
#include <mi/mdl/mdl_encapsulator.h>
#include <base/system/main/access_module.h>

#include "neuray_mdl_resource_callback.h"

#include <unordered_map>
#include <unordered_set>

namespace mi {

    namespace mdl {
        class IEntity_resolver;
        class IMDL;
        class IMDL_resource_reader;
    }
    
    namespace neuraylib {
        class INeuray;
    }
}

namespace MI {

namespace DB { class Transaction; }
namespace MDL { class Execution_context; }
namespace MDLC { class Mdlc_module; }

namespace NEURAY {

class Mdle_resource_mapper : public Resource_callback
                           , public mi::mdl::IEncapsulate_tool_resource_collector
{
    typedef Resource_callback Base;
public:

    explicit Mdle_resource_mapper(
        mi::mdl::IMDL* mdl,
        DB::Transaction* transaction,
        MI::MDL::Execution_context* context);

    /// Retrieve the "resource name" of an MDL resource value.
    ///
    /// \param  v                            a resource value
    /// \param support_strict_relative_path  if true, the resource name allows strict relative path
    ///
    /// \return The MDL name of this resource value.
    char const *get_resource_name(
        const mi::mdl::IValue_resource* v,
        bool support_strict_relative_path) override;

    /// Number of resources files to encapsulate.
    size_t get_resource_count() const override;

    /// Get the resource path that should be used in the MDLE main module.
    char const *get_mlde_resource_path(size_t index) const override;

    /// Get a stream reader interface that gives access to the requested resource data.
    mi::mdl::IMDL_resource_reader *get_resource_reader(size_t index) const override;

    // Get a stream reader interface that gives access to the requested addition data file.
    mi::mdl::IMDL_resource_reader *get_additional_data_reader(char const* path) override;

private:

    // represents a file that is added to the MDLE container.
    struct Resource_desc
    {
        std::string mdle_file_name;
        std::string mdle_file_name_mask;
        std::string resolved_file_name;
        mi::neuraylib::IBuffer* in_memory_buffer;
    };

    // Avoid file name collisions inside the MDLE.
    std::string generate_mdle_name(const std::string& base_name);

    mi::base::Handle<mi::mdl::IMDL> m_mdl;
    mi::base::Handle<mi::mdl::IEntity_resolver> m_resolver;
    MI::MDL::Execution_context* m_context;
    
    // Map from resolved name (result of Base::get_resource_name) to the resource name in the MDLE.
    std::unordered_map<std::string, std::string> m_resource_names_resolved2mdle;

    // All resource files that will be added to the MDLE.
    std::vector<Resource_desc> m_resources;

    // Keep track of resource names in the MDLE, avoid collision by using 'generate_mdle_name(...)'.
    std::unordered_set<std::string> m_reserved_mdle_names;
};

class Mdle_api_impl final
  : public mi::base::Interface_implement<mi::neuraylib::IMdle_api>
{
public:
    /// Constructor of Mdle_api_impl.
    ///
    /// \param neuray      The neuray instance which contains this Mdle_api_impl
    Mdle_api_impl(mi::neuraylib::INeuray* neuray);

    /// Destructor of Mdle_api_impl.
    virtual ~Mdle_api_impl();

public:
    // public API methods

    mi::Sint32 export_mdle(
        mi::neuraylib::ITransaction* transaction,
        const char* file_name,
        const mi::IStructure* mdle_data,
        mi::neuraylib::IMdl_execution_context* context) const final;

    mi::Sint32 validate_mdle(
        const char* file_name,
        mi::neuraylib::IMdl_execution_context* context) const final;

    mi::neuraylib::IReader* get_user_file(
        const char* mdle_file_name,
        const char* user_file_name,
        mi::neuraylib::IMdl_execution_context* context) const final;

    mi::Sint32 compare_mdle(
        const char* mdle_file_name_a,
        const char* mdle_file_name_b,
        mi::neuraylib::IMdl_execution_context* context) const final;

    mi::Sint32 get_hash(
        const char* mdle_file_name,
        mi::base::Uuid& hash,
        mi::neuraylib::IMdl_execution_context* context) const final;

    // internal methods

    /// Starts this API component.
    ///
    /// The implementation of INeuray::start() calls the #start() method of each API component.
    /// This method performs the API component's specific part of the library start.
    ///
    /// \return            0, in case of success, -1 in case of failure.
    mi::Sint32 start();

    /// Shuts down this API component.
    ///
    /// The implementation of INeuray::shutdown() calls the #shutdown() method of each API
    /// component. This method performs the API component's specific part of the library shutdown.
    ///
    /// \return           0, in case of success, -1 in case of failure
    mi::Sint32 shutdown();

private:
    // non copyable
    Mdle_api_impl(Mdle_api_impl const &) = delete;
    Mdle_api_impl &operator=(Mdle_api_impl const &) = delete;

private:
    mi::neuraylib::INeuray *m_neuray;

    SYSTEM::Access_module<MDLC::Mdlc_module> m_mdlc_module;
};

} // namespace NEURAY
} // namespace MI

#endif // API_API_NEURAY_MDLE_API_IMPL_H
