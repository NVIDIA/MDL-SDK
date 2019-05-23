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
/// \file
/// \brief Header for the IMdl_entity_resolver implementation.

#ifndef API_API_MDL_MDL_ENTITY_RESOLVER_IMPL_H
#define API_API_MDL_MDL_ENTITY_RESOLVER_IMPL_H

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <base/system/main/neuray_cc_conf.h>
#include <mi/neuraylib/imdl_entity_resolver.h>

namespace mi
{
    namespace mdl { class IEntity_resolver; class IMDL_resource_set; }
}

namespace MI {

namespace MDL {

/// Implementation of #mi::neuraylib::IMdl_resource_set.
class Mdl_resource_set_impl : public mi::base::Interface_implement<mi::neuraylib::IMdl_resource_set>
{
public:

    /// Constructor.
    ///
    /// \param resolver The core api resource set.
    Mdl_resource_set_impl(mi::mdl::IMDL_resource_set *set);

    // API methods
    mi::Size get_count() const NEURAY_FINAL;

    const char *get_mdl_url(mi::Size i) const NEURAY_FINAL;
    
    const char *get_filename(mi::Size i) const NEURAY_FINAL;
    
    bool get_udim_mapping(mi::Size i, mi::Sint32 &u, mi::Sint32 &v) const NEURAY_FINAL;
    
    mi::neuraylib::IReader *open_reader(mi::Size i) const NEURAY_FINAL;

private:
    mi::base::Handle<mi::mdl::IMDL_resource_set> m_resource_set;
};

/// Implementation of #mi::neuraylib::IMdl_entity_resolver.
class Mdl_entity_resolver_impl :
    public mi::base::Interface_implement<mi::neuraylib::IMdl_entity_resolver>
{
public:
    /// Constructor.
    ///
    /// \param resolver The core api resolver.
    Mdl_entity_resolver_impl(mi::mdl::IEntity_resolver *resolver);

    // API methods
    mi::neuraylib::IMdl_resource_set *resolve_resource_file_name(
        const char     *file_path,
        const char     *owner_file_path,
        const char     *owner_name) NEURAY_FINAL;

    mi::neuraylib::IReader *open_resource(
        const char     *file_path,
        const char     *owner_file_path,
        const char     *owner_name) NEURAY_FINAL;

    const char *resolve_module_name(
        const char *name) NEURAY_FINAL;

private:
    mi::base::Handle<mi::mdl::IEntity_resolver> m_entity_resolver;
};

} // namespace MDL

} // namespace MI

#endif // API_API_MDL_MDL_ENTITY_RESOLVER_IMPL_H
