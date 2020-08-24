/***************************************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Implementation of the neuray library.
 **
 ** Implements the API to be used by integrators to integrate the neuray library.
 **/

#ifndef API_API_MDL_NEURAY_IMPL_H
#define API_API_MDL_NEURAY_IMPL_H

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/base/lock.h>
#include <mi/neuraylib/ineuray.h>

#include <map>
#include <base/system/main/access_module.h>
#include <boost/core/noncopyable.hpp>

namespace mi { 
    namespace base { class ILogger; }
    namespace neuraylib { class IVersion; }
}

namespace MI {

namespace DB { class Database; }

namespace NEURAY {

class Class_factory;
class Database_impl;
class Debug_configuration_impl;
class Factory_impl;
class Image_api_impl;
class Mdl_archive_api_impl;
class Mdl_backend_api_impl;
class Mdl_compatibility_api_impl;
class Mdl_configuration_impl;
class Mdl_discovery_api_impl;
class Mdl_evaluator_api_impl;
class Mdl_factory_impl;
class Mdl_i18n_configuration_impl;
class Mdl_impexp_api_impl;
class Mdle_api_impl;
class Plugin_configuration_impl;
}

namespace MDL {

class Logger;
class Mdl_compiler_impl;

class Neuray_impl
  : public mi::base::Interface_implement<mi::neuraylib::INeuray>,
    public boost::noncopyable
{
public:
    /// Constructor.
    Neuray_impl();

    /// Destructor.
    ~Neuray_impl();

    // public API methods

    mi::Uint32 get_interface_version() const;

    const char* get_version() const;

    mi::Sint32 start( bool blocking);

    mi::Sint32 shutdown( bool blocking);

    Status get_status() const;

    mi::base::IInterface* get_api_component( const mi::base::Uuid& uuid) const;

    using mi::neuraylib::INeuray::get_api_component;

    mi::Sint32 register_api_component(
        const mi::base::Uuid& uuid, mi::base::IInterface* api_component);

    using mi::neuraylib::INeuray::register_api_component;

    mi::Sint32 unregister_api_component( const mi::base::Uuid& uuid);

    using mi::neuraylib::INeuray::unregister_api_component;

    //  internal methods

    void set_logger( mi::base::ILogger* logger);

    mi::base::ILogger* get_logger();

    /// Returns the class factory.
    ///
    /// \note This method does \em not increase the reference count of the return value.
    NEURAY::Class_factory* get_class_factory();

    /// Counts the number of instances of this class.
    ///
    /// Used by mi_factory() to avoid multiple instances of this class.
    static mi::base::Atom32 s_instance_count;

private:
      /// Logs the startup message (library path and version information).
    void log_startup_message();

    /// The version number.
    mi::base::Handle<mi::neuraylib::IVersion> m_version_impl;

    /// The class factory. Initialize before m_database_impl.
    NEURAY::Class_factory* m_class_factory;

    /// Pointers to the API components
    NEURAY::Factory_impl* m_factory_impl;
    NEURAY::Database_impl* m_database_impl;
    NEURAY::Debug_configuration_impl* m_debug_configuration_impl;
    NEURAY::Image_api_impl* m_image_api_impl;
    Mdl_compiler_impl* m_mdl_compiler_impl;
    NEURAY::Mdl_archive_api_impl* m_mdl_archive_api_impl;
    NEURAY::Mdl_backend_api_impl* m_mdl_backend_api_impl;
    NEURAY::Mdl_compatibility_api_impl* m_mdl_compatibility_api_impl;
    NEURAY::Mdl_configuration_impl* m_mdl_configuration_impl;
    NEURAY::Mdl_discovery_api_impl* m_mdl_discovery_api_impl;
    NEURAY::Mdl_evaluator_api_impl* m_mdl_evaluator_api_impl;
    NEURAY::Mdl_factory_impl* m_mdl_factory_impl;
    NEURAY::Mdl_i18n_configuration_impl* m_mdl_i18n_configuration_impl;
    NEURAY::Mdl_impexp_api_impl* m_mdl_impexp_api_impl;
    NEURAY::Mdle_api_impl* m_mdle_api_impl;
    NEURAY::Plugin_configuration_impl* m_plugin_configuration_impl;
    /// Status of the instance, see #get_status().
    Status m_status;

    /// The type of the map that stores the registered API components
    typedef std::map<mi::base::Uuid, mi::base::IInterface*> Api_components_map;
    /// The map that stores the registered API components
    Api_components_map m_api_components;
    /// The lock for the map that stores the registered API components
    mutable mi::base::Lock m_api_components_lock;

    /// The logger.
    Logger* m_logger;

    /// The database.
    DB::Database* m_database;
};

} // namespace MDL

} // namespace MI

#endif // API_API_MDL_NEURAY_IMPL_H

