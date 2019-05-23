/***************************************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "mdl_mdl_entity_resolver_impl.h"

#include <mi/mdl/mdl_entity_resolver.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>

namespace MI {

namespace MDL {

// Constructor.
Mdl_resource_set_impl::Mdl_resource_set_impl(mi::mdl::IMDL_resource_set *set)
: m_resource_set(mi::base::make_handle_dup(set))
{
}

// Get resource count.
mi::Size Mdl_resource_set_impl::get_count() const
{
    return m_resource_set->get_count();
}

// Get i'th MDL url.
const char *Mdl_resource_set_impl::get_mdl_url(mi::Size i) const
{
    return m_resource_set->get_mdl_url(i);
}

// Get i'th filename.
const char *Mdl_resource_set_impl::get_filename(mi::Size i) const
{
    return m_resource_set->get_filename(i);
}

// Get i'th udim mapping.
bool Mdl_resource_set_impl::get_udim_mapping(mi::Size i, mi::Sint32 &u, mi::Sint32 &v) const
{
    return m_resource_set->get_udim_mapping(i, u, v);
}

// Get i'th reader.
mi::neuraylib::IReader *Mdl_resource_set_impl::open_reader(mi::Size i) const
{
    mi::base::Handle<mi::mdl::IMDL_resource_reader> reader(
        m_resource_set->open_reader(i));
    if (!reader)
        return nullptr;
    return MDL::get_reader(reader.get());
}

// Constructor.
Mdl_entity_resolver_impl::Mdl_entity_resolver_impl(mi::mdl::IEntity_resolver *resolver)
    : m_entity_resolver(mi::base::make_handle_dup(resolver))
{
}

// Resolve resource file name.
mi::neuraylib::IMdl_resource_set *Mdl_entity_resolver_impl::resolve_resource_file_name(
    const char     *file_path,
    const char     *owner_file_path,
    const char     *owner_name)
{
    mi::base::Handle<mi::mdl::IMDL_resource_set> result(
        m_entity_resolver->resolve_resource_file_name(
            file_path,
            owner_file_path,
            owner_name,
            /*pos=*/0));
    if (!result)
        return nullptr;
    return new Mdl_resource_set_impl(result.get());
}

// Get resource reader.
mi::neuraylib::IReader *Mdl_entity_resolver_impl::open_resource(
    const char     *file_path,
    const char     *owner_file_path,
    const char     *owner_name)
{
    mi::base::Handle<mi::mdl::IMDL_resource_reader> reader(
        m_entity_resolver->open_resource(
            file_path,
            owner_file_path,
            owner_name,
            /*pos=*/0));
    if (!reader)
        return nullptr;
    return MDL::get_reader(reader.get());
}

// Resolve a module.
const char *Mdl_entity_resolver_impl::resolve_module_name(
    const char *name)
{
    mi::base::Handle<mi::mdl::IMDL_import_result> res(m_entity_resolver->resolve_module(
        name,
        /*owner_file_path=*/NULL,
        /*owner_name=*/NULL,
        /*pos=*/NULL));

    if (res.is_valid_interface()) {
        return res->get_absolute_name();
    }
    return NULL;
}

} // namespace MDL

} // namespace MI
