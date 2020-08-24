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
#include <mi/base/handle.h>
#include <mi/neuraylib/imdl_configuration.h>
#include <mi/neuraylib/ineuray.h>

#include <vector>
#include <string>
#include <boost/core/noncopyable.hpp>
#include <base/system/main/access_module.h>

namespace MI {

namespace MDLC { class Mdlc_module; }
namespace PATH { class Path_module; }

namespace NEURAY {

class Mdl_configuration_impl
  : public mi::base::Interface_implement<mi::neuraylib::IMdl_configuration>,
    public boost::noncopyable
{
public:
    /// Constructor of Mdl_configuration_impl
    ///
    /// \param neuray_impl           The neuray instance which contains this Mdl_configuration_impl.
    Mdl_configuration_impl( mi::neuraylib::INeuray* neuray);

    /// Destructor of Mdl_configuration_impl
    ~Mdl_configuration_impl();

    // public API methods

    void set_logger( mi::base::ILogger* logger) final;

    mi::base::ILogger* get_logger() final;


    mi::Sint32 add_mdl_path( const char* path) final;

    mi::Sint32 remove_mdl_path( const char* path) final;

    void clear_mdl_paths() final;

    mi::Size get_mdl_paths_length() const final;

    const mi::IString* get_mdl_path( mi::Size index) const final;

    mi::Size get_mdl_system_paths_length() const final;

    const char* get_mdl_system_path( mi::Size index) const final;

    mi::Size get_mdl_user_paths_length() const final;

    const char* get_mdl_user_path( mi::Size index) const final;


    mi::Sint32 add_resource_path( const char* path) final;

    mi::Sint32 remove_resource_path( const char* path) final;

    void clear_resource_paths() final;

    mi::Size get_resource_paths_length() const final;

    const mi::IString* get_resource_path( mi::Size index) const final;


    mi::Sint32 set_implicit_cast_enabled( bool value) final;

    bool get_implicit_cast_enabled() const final;

    mi::Sint32 set_expose_names_of_let_expressions( bool value) final;

    bool get_expose_names_of_let_expressions() const final;

    mi::Sint32 set_simple_glossy_bsdf_legacy_enabled( bool value) final;

    bool get_simple_glossy_bsdf_legacy_enabled() const final;


    mi::neuraylib::IMdl_entity_resolver* get_entity_resolver() const final;

    void set_entity_resolver( mi::neuraylib::IMdl_entity_resolver* resolver) final;

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

    /// Resets this API component.
    ///
    /// Releases and re-acquires the MDLC module.
    void reset();

    /// Returns the default MDL system path.
    std::string get_default_mdl_system_path() const;

    /// Returns the default MDL user path.
    std::string get_default_mdl_user_path() const;

private:
    mi::neuraylib::INeuray* m_neuray;                       // neuray interface

    SYSTEM::Access_module<PATH::Path_module> m_path_module; // path module
    SYSTEM::Access_module<MDLC::Mdlc_module> m_mdlc_module; // mdlc module

    bool m_implicit_cast_enabled;
    bool m_expose_names_of_let_expressions;
    bool m_simple_glossy_bsdf_legacy_enabled;
    mi::base::Handle<mi::neuraylib::IMdl_entity_resolver> m_entity_resolver;
    std::vector<std::string> m_mdl_system_paths;
    std::vector<std::string> m_mdl_user_paths;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_MDL_CONFIGURATION_IMPL_H
