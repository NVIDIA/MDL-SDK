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

#include "pch.h"

#include "mdl_neuray_impl.h"

#include <mi/neuraylib/version.h>

#include <boost/core/ignore_unused.hpp>
#include <base/system/main/access_module.h>
#include <base/system/version/i_version.h>
#include <base/lib/log/i_log_logger.h>
#include <base/data/db/i_db_database.h>
#include <base/data/dblight/i_dblight.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <io/image/image/i_image.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>

// API components
#include "neuray_database_impl.h"
#include "neuray_debug_configuration_impl.h"
#include "neuray_factory_impl.h"
#include "neuray_image_api_impl.h"
#include "neuray_mdl_archive_api_impl.h"
#include "neuray_mdl_backend_api_impl.h"
#include "neuray_mdl_compatibility_api_impl.h"
#include "neuray_mdl_configuration_impl.h"
#include "neuray_mdl_discovery_api_impl.h"
#include "neuray_mdl_evaluator_api_impl.h"
#include "neuray_mdl_factory_impl.h"
#include "neuray_mdl_i18n_configuration_impl.h"
#include "neuray_mdl_impexp_api_impl.h"
#include "neuray_mdle_api_impl.h"
#include "neuray_plugin_configuration_impl.h"

#include "neuray_version_impl.h"
#include "mdl_mdl_compiler_impl.h"

#include "neuray_class_factory.h"
#include "neuray_class_registration.h"
#include "neuray_scope_impl.h"
#include "mdl_logger.h"

#ifdef MI_PLATFORM_WINDOWS
#include <mi/base/miwindows.h>
#else
#include <dlfcn.h>
#endif

namespace MI {

namespace MDL {

void pull_in_required_modules();

mi::base::Atom32 Neuray_impl::s_instance_count;

Neuray_impl::Neuray_impl()
  : m_status( PRE_STARTING), m_database( 0)
{
    pull_in_required_modules();

    m_version_impl = new NEURAY::Version_impl();

    m_logger = new Logger();

    NEURAY::s_class_factory = m_class_factory = new NEURAY::Class_factory();
    NEURAY::Class_registration::register_classes_part1( m_class_factory);

    // Be careful with the ordering of API components
    NEURAY::s_factory = m_factory_impl = new NEURAY::Factory_impl( m_class_factory);
    m_mdl_compiler_impl = new Mdl_compiler_impl( this);
    log_startup_message();
    m_debug_configuration_impl = new NEURAY::Debug_configuration_impl();
    m_database_impl = new NEURAY::Database_impl( m_status);
    m_image_api_impl = new NEURAY::Image_api_impl( this);
    m_mdl_factory_impl = new NEURAY::Mdl_factory_impl(this, m_class_factory);
    m_mdl_i18n_configuration_impl = new NEURAY::Mdl_i18n_configuration_impl(this);
    m_mdl_impexp_api_impl = new NEURAY::Mdl_impexp_api_impl(this);
    m_mdl_discovery_api_impl = new NEURAY::Mdl_discovery_api_impl(this);
    m_mdl_evaluator_api_impl = new NEURAY::Mdl_evaluator_api_impl( this);
    m_mdl_archive_api_impl = new NEURAY::Mdl_archive_api_impl( this);
    m_mdl_backend_api_impl = new NEURAY::Mdl_backend_api_impl(this);
    m_mdl_compatibility_api_impl = new NEURAY::Mdl_compatibility_api_impl( this);
    m_mdl_configuration_impl = new NEURAY::Mdl_configuration_impl( this);
    m_mdle_api_impl = new NEURAY::Mdle_api_impl( this);
    m_plugin_configuration_impl = new NEURAY::Plugin_configuration_impl(this);

    // Register API components that are always available,
    // other API components are registered in start()
    register_api_component<mi::neuraylib::IDebug_configuration>( m_debug_configuration_impl);
    register_api_component<mi::neuraylib::IFactory>( m_factory_impl);
    register_api_component<mi::neuraylib::IMdl_compiler>( m_mdl_compiler_impl);
    register_api_component<mi::neuraylib::IVersion>( m_version_impl.get());
    register_api_component<mi::neuraylib::IMdl_configuration>( m_mdl_configuration_impl);
    register_api_component<mi::neuraylib::IMdl_i18n_configuration>( m_mdl_i18n_configuration_impl);
    register_api_component<mi::neuraylib::IPlugin_configuration>(m_plugin_configuration_impl);
}

Neuray_impl::~Neuray_impl()
{
    if (m_status != PRE_STARTING && m_status != SHUTDOWN)
    {
        mi::Sint32 result = 0;
        boost::ignore_unused(result);

        result = shutdown(true);
        ASSERT(M_NEURAY_API, result == 0);
    }

    // Unregister API components that are always available,
    // other API components are unregistered in shutdown()
    unregister_api_component<mi::neuraylib::IPlugin_configuration>();
    unregister_api_component<mi::neuraylib::IMdl_i18n_configuration>();
    unregister_api_component<mi::neuraylib::IMdl_configuration>();
    unregister_api_component<mi::neuraylib::IVersion>();
    unregister_api_component<mi::neuraylib::IMdl_compiler>();
    unregister_api_component<mi::neuraylib::IFactory>();
    unregister_api_component<mi::neuraylib::IDebug_configuration>();

    // Unit tests with a failing check usually hit this assertion because in such a case the
    // library is not properly shut down.
    ASSERT( M_NEURAY_API, m_api_components.empty());

    NEURAY::s_class_factory->unregister_structure_decls();
    NEURAY::s_class_factory->unregister_enum_decls();

#define CHECK_RESULT ASSERT( M_NEURAY_API, ref_count == 0 || m_status == FAILURE);

    // Be careful with the ordering
    mi::Uint32 ref_count = 0;
    boost::ignore_unused( ref_count);
    ref_count = m_mdl_archive_api_impl->release();          CHECK_RESULT;
    ref_count = m_mdl_backend_api_impl->release();          CHECK_RESULT;
    ref_count = m_mdl_compatibility_api_impl->release();    CHECK_RESULT;
    ref_count = m_mdl_configuration_impl->release();        CHECK_RESULT;
    ref_count = m_mdl_discovery_api_impl->release();        CHECK_RESULT;
    ref_count = m_mdl_evaluator_api_impl->release();        CHECK_RESULT;
    ref_count = m_mdl_i18n_configuration_impl->release();   CHECK_RESULT;
    ref_count = m_mdl_impexp_api_impl->release();           CHECK_RESULT;
    ref_count = m_mdl_factory_impl->release();              CHECK_RESULT;
    ref_count = m_mdle_api_impl->release();                 CHECK_RESULT;
    ref_count = m_image_api_impl->release();                CHECK_RESULT;
    ref_count = m_debug_configuration_impl->release();      CHECK_RESULT;
    ref_count = m_database_impl->release();                 CHECK_RESULT;
    ref_count = m_mdl_compiler_impl->release();             CHECK_RESULT;
    ref_count = m_factory_impl->release();                  CHECK_RESULT;
    ref_count = m_plugin_configuration_impl->release();     CHECK_RESULT;

    NEURAY::s_factory = 0;

#undef CHECK_RESULT

    delete m_class_factory;
    NEURAY::s_class_factory = 0;

#ifdef ENABLE_ASSERT
    mi::Size alive_modules = SYSTEM::Module_registration_entry::count_alive_modules();
    if( alive_modules > 0 && m_status != FAILURE)
        SYSTEM::Module_registration_entry::dump_alive_modules();
    ASSERT( M_NEURAY_API, alive_modules == 0 || m_status == FAILURE);
#endif // ENABLE_ASSERT

    delete m_logger;

    --s_instance_count;
}

mi::Uint32 Neuray_impl::get_interface_version() const
{
    return MI_NEURAYLIB_API_VERSION;
}

const char* Neuray_impl::get_version() const
{
    return m_version_impl->get_string();
}

mi::Sint32 Neuray_impl::start( bool blocking)
{
    if( PRE_STARTING != m_status && SHUTDOWN != m_status)
    	return -1;

    m_logger->emit_delayed_log_messages();

    m_status = STARTING;
    mi::Sint32 result = 0;

    NEURAY::Class_registration::register_classes_part2( m_class_factory);

    m_database = DBLIGHT::factory();

#define CHECK_RESULT if( result) { m_status = FAILURE; return result; }

    // Be careful with the ordering
    result = m_database_impl->start( m_database);   CHECK_RESULT;
    result = m_image_api_impl->start();             CHECK_RESULT;
    result = m_mdl_compiler_impl->start();          CHECK_RESULT;
    result = m_mdl_archive_api_impl->start();       CHECK_RESULT;
    result = m_mdl_backend_api_impl->start();       CHECK_RESULT;
    result = m_mdl_compatibility_api_impl->start(); CHECK_RESULT;
    result = m_mdl_configuration_impl->start();     CHECK_RESULT;
    result = m_mdl_discovery_api_impl->start();     CHECK_RESULT;
    result = m_mdl_evaluator_api_impl->start();     CHECK_RESULT;
    result = m_mdl_i18n_configuration_impl->start();CHECK_RESULT;
    result = m_mdl_impexp_api_impl->start();        CHECK_RESULT;
    result = m_mdle_api_impl->start();              CHECK_RESULT;
    result = m_plugin_configuration_impl->start();  CHECK_RESULT;
#undef CHECK_RESULT

    register_api_component<mi::neuraylib::IDatabase>( m_database_impl);
    register_api_component<mi::neuraylib::IImage_api>( m_image_api_impl);
    register_api_component<mi::neuraylib::IMdl_factory>( m_mdl_factory_impl);
    register_api_component<mi::neuraylib::IMdl_archive_api>( m_mdl_archive_api_impl);
    register_api_component<mi::neuraylib::IMdl_backend_api>(m_mdl_backend_api_impl);
    register_api_component<mi::neuraylib::IMdl_compatibility_api>( m_mdl_compatibility_api_impl);
    register_api_component<mi::neuraylib::IMdl_discovery_api>(m_mdl_discovery_api_impl);
    register_api_component<mi::neuraylib::IMdl_evaluator_api>(m_mdl_evaluator_api_impl);
    register_api_component<mi::neuraylib::IMdl_impexp_api>(m_mdl_impexp_api_impl);
    register_api_component<mi::neuraylib::IMdle_api>(m_mdle_api_impl);
    
    NEURAY::Class_registration::register_structure_declarations( m_class_factory);

    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
    mi::base::Handle<IMAGE::IMdl_container_callback> callback( MDL::create_mdl_container_callback());
    image_module->set_mdl_container_callback( callback.get());

    m_status = STARTED;

    return result;
}

mi::Sint32 Neuray_impl::shutdown( bool blocking)
{
    if( STARTED != m_status)
        return -1;

    m_status = SHUTTINGDOWN;

    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);
    image_module->set_mdl_container_callback( 0);

    NEURAY::Class_registration::unregister_structure_declarations( m_class_factory);

    unregister_api_component<mi::neuraylib::IMdle_api>();
    unregister_api_component<mi::neuraylib::IMdl_evaluator_api>();
    unregister_api_component<mi::neuraylib::IMdl_archive_api>();
    unregister_api_component<mi::neuraylib::IMdl_backend_api>();
    unregister_api_component<mi::neuraylib::IMdl_compatibility_api>();
    unregister_api_component<mi::neuraylib::IMdl_discovery_api>();
    unregister_api_component<mi::neuraylib::IMdl_factory>();
    unregister_api_component<mi::neuraylib::IMdl_impexp_api>();
    unregister_api_component<mi::neuraylib::IImage_api>();
    unregister_api_component<mi::neuraylib::IDatabase>();

    m_database->close();

#define CHECK_RESULT  if( result ) { m_status = FAILURE; return result; }

    // Be careful with the ordering
    mi::Sint32 result = 0;
    result = m_mdl_archive_api_impl->shutdown();        CHECK_RESULT;
    result = m_mdl_backend_api_impl->shutdown();        CHECK_RESULT;
    result = m_mdl_compatibility_api_impl->shutdown();  CHECK_RESULT;
    result = m_mdl_configuration_impl->shutdown();      CHECK_RESULT;
    result = m_mdl_discovery_api_impl->shutdown();      CHECK_RESULT;
    result = m_mdl_evaluator_api_impl->shutdown();      CHECK_RESULT;
    result = m_mdl_i18n_configuration_impl->shutdown(); CHECK_RESULT;
    result = m_mdl_impexp_api_impl->shutdown();         CHECK_RESULT;
    result = m_mdle_api_impl->shutdown();               CHECK_RESULT;
    result = m_mdl_compiler_impl->shutdown();           CHECK_RESULT;
    result = m_image_api_impl->shutdown();              CHECK_RESULT;
    result = m_database_impl->shutdown();               CHECK_RESULT;
    result = m_plugin_configuration_impl->shutdown();   CHECK_RESULT;
#undef CHECK_RESULT

    // Reset MDL configuration to prepare for another start
    m_mdl_configuration_impl->reset();

    m_status = SHUTDOWN;

    return result;
}

mi::neuraylib::INeuray::Status Neuray_impl::get_status() const
{
    return m_status;
}

mi::base::IInterface* Neuray_impl::get_api_component( const mi::base::Uuid& uuid) const
{
    mi::base::Lock::Block block( &m_api_components_lock);
    Api_components_map::const_iterator it = m_api_components.find( uuid);
    if( it == m_api_components.end())
        return 0;

    mi::base::IInterface* api_component = it->second;
    api_component->retain();
    return api_component;
}

mi::Sint32 Neuray_impl::register_api_component(
    const mi::base::Uuid& uuid, mi::base::IInterface* api_component)
{
    mi::base::Lock::Block block( &m_api_components_lock);
    if( !api_component)
        return -1;
    Api_components_map::const_iterator it = m_api_components.find( uuid);
    if( it != m_api_components.end())
        return -2;

    m_api_components[uuid] = api_component;
    api_component->retain();
    return 0;
}

mi::Sint32 Neuray_impl::unregister_api_component( const mi::base::Uuid& uuid)
{
    mi::base::Lock::Block block( &m_api_components_lock);
    Api_components_map::const_iterator it = m_api_components.find( uuid);
    if( it == m_api_components.end())
        return -1;

    mi::base::IInterface* api_component = it->second;
    api_component->release();
    m_api_components.erase( uuid);
    return 0;
}

void Neuray_impl::set_logger( mi::base::ILogger* logger)
{
    m_logger->set_logger( logger);
}

mi::base::ILogger* Neuray_impl::get_logger()
{
    return m_logger->get_logger();
}

NEURAY::Class_factory* Neuray_impl::get_class_factory()
{
    return m_class_factory;
}

void Neuray_impl::log_startup_message()
{
    m_logger->delay_log_messages( true);

#ifdef MI_PLATFORM_WINDOWS
    HMODULE handle = GetModuleHandle( "libmdl_sdk");
    TCHAR filename[MAX_PATH];
    if( handle && GetModuleFileName( handle, filename, MAX_PATH))
        LOG::mod_log->info( M_NEURAY_API, LOG::Mod_log::C_DATABASE, "Loaded \"%s\"", filename);
#else
    void* symbol = dlsym( RTLD_DEFAULT, "mi_factory");
    if( symbol) {
        Dl_info dl_info;
        dladdr( symbol, &dl_info);
        if( dl_info.dli_fname)
            LOG::mod_log->info(
                M_NEURAY_API, LOG::Mod_log::C_MISC, "Loaded \"%s\"", dl_info.dli_fname);
    }
#endif
    LOG::mod_log->info(
        M_NEURAY_API, LOG::Mod_log::C_MISC, "%s", m_version_impl->get_string());

    m_logger->delay_log_messages( false);
}

} // namespace MDL

} // namespace MI

