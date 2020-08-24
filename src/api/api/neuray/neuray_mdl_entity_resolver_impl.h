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
 ** \brief Header for the IMdl_entity_resolver implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_MDL_ENTITY_RESOLVER_IMPL_H
#define API_API_NEURAY_NEURAY_MDL_ENTITY_RESOLVER_IMPL_H

#include <mi/neuraylib/imdl_entity_resolver.h>

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/mdl/mdl_entity_resolver.h>

#include <mdl/compiler/compilercore/compilercore_messages.h>

namespace mi { namespace mdl { class IMDL; } }

namespace MI {

namespace NEURAY {

class Mdl_entity_resolver_impl
  : public mi::base::Interface_implement<mi::neuraylib::IMdl_entity_resolver>
{
public:

    Mdl_entity_resolver_impl( mi::mdl::IMDL* mdl, mi::mdl::IEntity_resolver* resolver);

    // public API methods

    mi::neuraylib::IMdl_resolved_module* resolve_module(
        const char* module_name,
        const char* owner_file_path,
        const char* owner_name,
        mi::Sint32 pos_line,
        mi::Sint32 pos_column,
        mi::neuraylib::IMdl_execution_context* context) final;

    mi::neuraylib::IMdl_resolved_resource* resolve_resource(
        const char* file_path,
        const char* owner_file_path,
        const char* owner_name,
        mi::Sint32 pos_line,
        mi::Sint32 pos_column,
        mi::neuraylib::IMdl_execution_context* context) final;

private:
    mi::base::Handle<mi::mdl::IMDL> m_mdl;
    mi::base::Handle<mi::mdl::IEntity_resolver> m_resolver;
};

class Mdl_resolved_module_impl
  : public mi::base::Interface_implement<mi::neuraylib::IMdl_resolved_module>
{
public:
    Mdl_resolved_module_impl( mi::mdl::IMDL* mdl, mi::mdl::IMDL_import_result* import_result);

    // public API methods

    const char* get_module_name() const final;

    const char* get_filename() const final;

    mi::neuraylib::IReader* create_reader() const final;

private:
    mi::base::Handle<mi::mdl::IMDL> m_mdl;
    mi::base::Handle<mi::mdl::IMDL_import_result> m_import_result;
};

class Mdl_resolved_resource_impl
  : public mi::base::Interface_implement<mi::neuraylib::IMdl_resolved_resource>
{
public:
    Mdl_resolved_resource_impl( mi::mdl::IMDL_resource_set* resource_set);

    // public API methods

    mi::neuraylib::Uvtile_mode get_uvtile_mode() const final;

    const char* get_mdl_file_path_mask() const final;

    const char* get_filename_mask() const final;

    mi::Size get_count() const final;

    const char* get_mdl_file_path( mi::Size i) const final;

    const char* get_filename( mi::Size i) const final;

    mi::neuraylib::IReader* create_reader( mi::Size i) const final;

    mi::base::Uuid get_resource_hash( mi::Size i) const final;

    bool get_uvtile_uv( mi::Size i, mi::Sint32& u, mi::Sint32& v) const final;

private:
    mi::base::Handle<mi::mdl::IMDL_resource_set> m_resource_set;
};

class Core_entity_resolver_impl
  : public mi::base::Interface_implement<mi::mdl::IEntity_resolver>
{
public:
    Core_entity_resolver_impl( mi::mdl::IMDL* mdl, mi::neuraylib::IMdl_entity_resolver* resolver);

    // MDL core API methods

    mi::mdl::IMDL_import_result* resolve_module(
        const char* module_name,
        const char* owner_file_path,
        const char* owner_name,
        const mi::mdl::Position* pos) final;

    mi::mdl::IMDL_resource_set* resolve_resource_file_name(
        const char* file_path,
        const char* owner_file_path,
        const char* owner_name,
        const mi::mdl::Position* pos) final;

    const mi::mdl::Messages& access_messages() const final;

private:
    mi::base::Handle<mi::mdl::IMDL> m_mdl;
    mi::base::Handle<mi::neuraylib::IMdl_entity_resolver> m_resolver;
    mi::mdl::Messages_impl m_messages;
};

class Core_mdl_import_result_impl
  : public mi::base::Interface_implement<mi::mdl::IMDL_import_result>
{
public:
    Core_mdl_import_result_impl( mi::neuraylib::IMdl_resolved_module* resolved_module);

    // MDL core API methods

    const char* get_absolute_name() const final;

    const char* get_file_name() const final;

    mi::mdl::IInput_stream* open( mi::mdl::IThread_context* context) const final;

private:
    mi::base::Handle<mi::neuraylib::IMdl_resolved_module> m_resolved_module;
};

class Core_mdl_resource_set_impl
  : public mi::base::Interface_implement<mi::mdl::IMDL_resource_set>
{
public:
    Core_mdl_resource_set_impl( mi::neuraylib::IMdl_resolved_resource* resolved_resource);

    const char* get_mdl_url_mask() const final;

    const char* get_filename_mask() const final;

    size_t get_count() const final;

    const char* get_mdl_url( size_t i) const final;

    const char* get_filename( size_t i) const final;

    bool get_udim_mapping( size_t i, int &u, int &v) const final;

    mi::mdl::IMDL_resource_reader* open_reader(size_t i) const final;

    mi::mdl::UDIM_mode get_udim_mode() const final;

    bool get_resource_hash( size_t i, unsigned char hash[16]) const final;

private:
    mi::base::Handle<mi::neuraylib::IMdl_resolved_resource> m_resolved_resource;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_MDL_ENTITY_RESOLVER_IMPL_H
