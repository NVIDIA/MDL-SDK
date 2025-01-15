/***************************************************************************************************
 * Copyright (c) 2017-2025, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Header for the IMdl_distiller_api implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_MDL_DISTILLER_API_IMPL_H
#define API_API_NEURAY_NEURAY_MDL_DISTILLER_API_IMPL_H

#include <mi/neuraylib/imdl_distiller_api.h>

#include <string>
#include <boost/core/noncopyable.hpp>
#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <base/system/main/access_module.h>

namespace mi { namespace neuraylib { class INeuray; class ITarget_code; } }

namespace MI {

namespace DB { class Transaction; }
namespace DIST  { class Dist_module; }
namespace BAKER { class Baker_module; class IBaker_code; }

namespace NEURAY {

class Mdl_distiller_api_impl final
  : public mi::base::Interface_implement<mi::neuraylib::IMdl_distiller_api>,
    public boost::noncopyable
{
public:
    /// Constructor of Mdl_distiller_api_impl
    ///
    /// \param neuray      The neuray instance which contains this Mdl_distiller_api_impl
    Mdl_distiller_api_impl( mi::neuraylib::INeuray* neuray);

    /// Destructor of Library_authentication_impl
    ~Mdl_distiller_api_impl();

    // public API methods

    mi::Size get_target_count() const final;

    const char* get_target_name( mi::Size index) const final;

    mi::neuraylib::ICompiled_material* distill_material(
        const mi::neuraylib::ICompiled_material* material,
        const char* target,
        const mi::IMap* distiller_options,
        mi::Sint32* errors) const final;

    const mi::neuraylib::IBaker* create_baker(
        const mi::neuraylib::ICompiled_material* material,
        const char* path,
        mi::neuraylib::Baker_resource resource,
        Uint32 gpu_device_id) const final;

    mi::Size get_required_module_count( const char* target) const;

    const char* get_required_module_name( const char* target, mi::Size index) const;

    const char* get_required_module_code( const char* target, mi::Size index) const;

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
    mi::neuraylib::INeuray* m_neuray;
    SYSTEM::Access_module<DIST::Dist_module>   m_dist_module;
    SYSTEM::Access_module<BAKER::Baker_module> m_baker_module;
};

class Baker_impl
  : public mi::base::Interface_implement<mi::neuraylib::IBaker>,
    public boost::noncopyable
{
public:
    /// Constructor.
    Baker_impl(
        DB::Transaction* transaction,
        const BAKER::IBaker_code* baker_code,
        const char* pixel_type,
        bool is_uniform);

    ~Baker_impl();

    // public API methods

    const char* get_pixel_type() const;

    bool is_uniform() const;

    mi::Sint32 bake_texture( mi::neuraylib::ICanvas* texture, mi::Uint32 samples) const;

    mi::Sint32 bake_texture( 
        mi::neuraylib::ICanvas* texture, 
        mi::Float32 min_u, mi::Float32 max_u, mi::Float32 min_v, mi::Float32 max_v,
        mi::Uint32 samples) const;

    mi::Sint32 bake_constant( mi::IData* constant, mi::Uint32 samples) const;

    // internal methods

private:
    SYSTEM::Access_module<BAKER::Baker_module> m_baker_module;
    DB::Transaction* m_transaction;
    mi::base::Handle<const BAKER::IBaker_code> m_baker_code;
    std::string m_pixel_type;
    bool m_is_uniform;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_MDL_DISTILLER_API_IMPL_H
