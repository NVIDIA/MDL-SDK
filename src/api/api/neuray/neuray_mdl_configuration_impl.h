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
 ** \brief Implementation of IMdl_configuration
 **
 ** Implements the IMdl_configuration interface
 **/

#ifndef API_API_NEURAY_MDL_CONFIGURATION_IMPL_H
#define API_API_NEURAY_MDL_CONFIGURATION_IMPL_H

#include <mi/base/interface_implement.h>
#include <mi/neuraylib/imdl_configuration.h>
#include <mi/neuraylib/ineuray.h>

#include <boost/core/noncopyable.hpp>
#include <base/system/main/access_module.h>

namespace MI {

namespace MDLC { class Mdlc_module; }

namespace NEURAY {

class Class_factory;

/// This API component is shared between Neuray_impl and Cluster_impl. It must not use
/// Access_module<DATA::Data_module> or DATA::mod_data directly.
class Mdl_configuration_impl
  : public mi::base::Interface_implement<mi::neuraylib::IMdl_configuration>,
    public boost::noncopyable
{
public:
    /// Constructor of Mdl_configuration_impl
    ///
    /// \param neuray_impl           The neuray instance which contains this Mdl_configuration_impl.
    Mdl_configuration_impl(mi::neuraylib::INeuray* neuray);

    /// Destructor of Mdl_configuration_impl
    ~Mdl_configuration_impl();

    // public API methods

    mi::Sint32 set_implicit_cast_enabled( bool value);

    bool get_implicit_cast_enabled() const;

    // internal methods

    /// Starts this API component.
    /// 
    /// The implementation of INeuray::start() calls the #start() method of each API component.
    /// This method performs the API component's specific part of the library start.
    ///
    /// \return 0, in case of success, -1 in case of failure.
    mi::Sint32 start();

    /// Shuts down this API component.
    ///
    /// The implementation of INeuray::shutdown() calls the #shutdown() method of each API
    /// component. This method performs the API component's specific part of the library shutdown.
    ///
    /// \return 0, in case of success, -1 in case of failure
    mi::Sint32 shutdown();

private:
    mi::neuraylib::INeuray* m_neuray;                       // neuray interface
    
    SYSTEM::Access_module<MDLC::Mdlc_module> m_mdlc_module; // mdlc module
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_MDL_CONFIGURATION_IMPL_H
