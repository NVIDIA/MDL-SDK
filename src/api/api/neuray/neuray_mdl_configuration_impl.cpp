/***************************************************************************************************
 * Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include "neuray_mdl_configuration_impl.h"

#include "neuray_class_factory.h"
#include "neuray_mdl_entity_resolver_impl.h"
#include "neuray_string_impl.h"

#include <mi/mdl/mdl_entity_resolver.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/neuraylib/ilogging_configuration.h>

#include <base/lib/config/config.h>
#include <base/lib/log/i_log_assert.h>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/path/i_path.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/util/registry/i_config_registry.h>
#include <base/util/string_utils/i_string_utils.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>

namespace MI {

namespace NEURAY {

namespace {

// Get an environment variable as UTF8 string
std::string getenv_utf8(const char *env_var, const wchar_t *w_env_var)
{
#ifdef MI_PLATFORM_WINDOWS
    const wchar_t *s = _wgetenv( w_env_var);
    if( s == nullptr)
        return std::string();
    return STRING::wchar_to_utf8( s);
#else
    const char *s = getenv( env_var);
    if( s == nullptr)
        return std::string();
    return std::string( s);
#endif
}

}

Mdl_configuration_impl::Mdl_configuration_impl( mi::neuraylib::INeuray* neuray)
  : m_neuray( neuray),
    m_config_module( /*deferred=*/false),
    m_path_module( /*deferred=*/false),
    m_mdlc_module( /*deferred=*/true)
{
    const std::string& separator = HAL::Ospath::get_path_set_separator();

    // set MDL system path
    std::string env_mdl_system_path = getenv_utf8( "MDL_SYSTEM_PATH", L"MDL_SYSTEM_PATH");
    if( !env_mdl_system_path.empty())
        STRING::split( env_mdl_system_path, separator, m_mdl_system_paths);
    else
        m_mdl_system_paths.push_back( get_default_mdl_system_path());

    // set MDL user path
    std::string env_mdl_user_path = getenv_utf8( "MDL_USER_PATH", L"MDL_USER_PATH");
    if( !env_mdl_user_path.empty())
        STRING::split( env_mdl_user_path, separator, m_mdl_user_paths);
    else
        m_mdl_user_paths.push_back( get_default_mdl_user_path());
}

Mdl_configuration_impl::~Mdl_configuration_impl()
{
    m_config_module.reset();
    m_path_module.reset();
    m_mdlc_module.reset();
}

mi::Sint32 Mdl_configuration_impl::add_mdl_path( const char* path)
{
    return !path ? -1 : m_path_module->add_path( PATH::MDL, path);
}

mi::Sint32 Mdl_configuration_impl::remove_mdl_path( const char* path)
{
    return !path ? -1 : m_path_module->remove_path( PATH::MDL, path);
}

void Mdl_configuration_impl::clear_mdl_paths()
{
    m_path_module->clear_search_path( PATH::MDL);
}

mi::Size Mdl_configuration_impl::get_mdl_paths_length() const
{
    return m_path_module->get_path_count( PATH::MDL);
}

const mi::IString* Mdl_configuration_impl::get_mdl_path( mi::Size index) const
{
    const std::string& result = m_path_module->get_path( PATH::MDL, index);
    if( result.empty())
        return nullptr;
    return new String_impl(result.c_str());
}

mi::Size Mdl_configuration_impl::get_mdl_system_paths_length() const
{
    return m_mdl_system_paths.size();
}

const char* Mdl_configuration_impl::get_mdl_system_path( mi::Size index) const
{
    return index < m_mdl_system_paths.size() ? m_mdl_system_paths[index].c_str() : nullptr;
}

mi::Size Mdl_configuration_impl::get_mdl_user_paths_length() const
{
    return m_mdl_user_paths.size();
}

const char* Mdl_configuration_impl::get_mdl_user_path( mi::Size index) const
{
    return index < m_mdl_user_paths.size() ? m_mdl_user_paths[index].c_str() : nullptr;
}

mi::Sint32 Mdl_configuration_impl::add_resource_path( const char* path)
{
    return !path ? -1 : m_path_module->add_path( PATH::RESOURCE, path);
}

mi::Sint32 Mdl_configuration_impl::remove_resource_path( const char* path)
{
    return !path ? -1 : m_path_module->remove_path( PATH::RESOURCE, path);
}

void Mdl_configuration_impl::clear_resource_paths()
{
    m_path_module->clear_search_path( PATH::RESOURCE);
}

mi::Size Mdl_configuration_impl::get_resource_paths_length() const
{
    return m_path_module->get_path_count( PATH::RESOURCE);
}

const mi::IString* Mdl_configuration_impl::get_resource_path( mi::Size index) const
{
    const std::string& result = m_path_module->get_path( PATH::RESOURCE, index);
    if( result.empty())
        return nullptr;
    return new String_impl(result.c_str());
}

mi::Sint32 Mdl_configuration_impl::set_implicit_cast_enabled( bool value)
{
    mi::neuraylib::INeuray::Status status = m_neuray->get_status();
    if(    (status != mi::neuraylib::INeuray::PRE_STARTING)
        && (status != mi::neuraylib::INeuray::SHUTDOWN))
        return -1;

    m_implicit_cast_enabled = value;
    return 0;
}

bool Mdl_configuration_impl::get_implicit_cast_enabled() const
{
    return m_implicit_cast_enabled;
}

mi::Sint32 Mdl_configuration_impl::set_expose_names_of_let_expressions( bool value)
{
    mi::neuraylib::INeuray::Status status = m_neuray->get_status();
    if(    (status != mi::neuraylib::INeuray::PRE_STARTING)
        && (status != mi::neuraylib::INeuray::SHUTDOWN))
        return -1;

    m_expose_names_of_let_expressions = value;
    return 0;
}

bool Mdl_configuration_impl::get_expose_names_of_let_expressions() const
{
    return m_expose_names_of_let_expressions;
}

mi::Sint32 Mdl_configuration_impl::set_simple_glossy_bsdf_legacy_enabled( bool value)
{
    return -1; // cannot be changed for mdl_sdk
}

bool Mdl_configuration_impl::get_simple_glossy_bsdf_legacy_enabled() const
{
    return m_simple_glossy_bsdf_legacy_enabled;
}

mi::Sint32 Mdl_configuration_impl::set_material_ior_frequency(
    mi::neuraylib::IType::Modifier frequency_qualifier)
{
    mi::neuraylib::INeuray::Status status = m_neuray->get_status();
    if(    (status != mi::neuraylib::INeuray::PRE_STARTING)
        && (status != mi::neuraylib::INeuray::SHUTDOWN))
        return -1;

    switch( frequency_qualifier) {
        case mi::neuraylib::IType::MK_UNIFORM:
            m_config_module->override( "mdl_material_ior_is_varying=0");
            return 0;
        case mi::neuraylib::IType::MK_VARYING:
            m_config_module->override( "mdl_material_ior_is_varying=1");
            return 0;
        case mi::neuraylib::IType::MK_NONE:
        case mi::neuraylib::IType::MK_FORCE_32_BIT:
            break;
    }

    return -2;
}

mi::neuraylib::IType::Modifier Mdl_configuration_impl::get_material_ior_frequency() const
{
    const CONFIG::Config_registry& registry = m_config_module->get_configuration();
    bool varying = true;
    registry.get_value( "mdl_material_ior_is_varying", varying);
    return varying ? mi::neuraylib::IType::MK_VARYING : mi::neuraylib::IType::MK_UNIFORM;
}

mi::neuraylib::IMdl_entity_resolver* Mdl_configuration_impl::get_entity_resolver() const
{
    mi::neuraylib::INeuray::Status status = m_neuray->get_status();
    if(    (status == mi::neuraylib::INeuray::PRE_STARTING)
        || (status == mi::neuraylib::INeuray::SHUTDOWN))
        return nullptr;

    mi::base::Handle<mi::mdl::IMDL> mdl( m_mdlc_module->get_mdl());

    mi::base::Handle<mi::mdl::IEntity_resolver> mdl_resolver(
        mdl->create_entity_resolver( /*module_cache*/ nullptr));
    return new Mdl_entity_resolver_impl( mdl.get(), mdl_resolver.get());
}

void Mdl_configuration_impl::set_entity_resolver( mi::neuraylib::IMdl_entity_resolver* resolver)
{
    m_entity_resolver = mi::base::make_handle_dup( resolver);

    mi::neuraylib::INeuray::Status status = m_neuray->get_status();
    if(    (status == mi::neuraylib::INeuray::PRE_STARTING)
        || (status == mi::neuraylib::INeuray::SHUTDOWN))
        return;

    mi::base::Handle<mi::mdl::IMDL> mdl( m_mdlc_module->get_mdl());
    if( !resolver) {
        mdl->set_external_entity_resolver( nullptr);
        return;
    }

    mi::base::Handle<mi::mdl::IEntity_resolver> mdl_resolver(
        new Core_entity_resolver_impl( mdl.get(), resolver));
    mdl->set_external_entity_resolver( mdl_resolver.get());
}

void Mdl_configuration_impl::deprecated_set_logger( mi::base::ILogger* logger)
{
    mi::base::Handle<mi::neuraylib::ILogging_configuration> logging_configuration(
        m_neuray->get_api_component<mi::neuraylib::ILogging_configuration>());
    logging_configuration->set_receiving_logger( logger);
}

mi::base::ILogger* Mdl_configuration_impl::deprecated_get_logger()
{
    mi::base::Handle<mi::neuraylib::ILogging_configuration> logging_configuration(
        m_neuray->get_api_component<mi::neuraylib::ILogging_configuration>());
    return logging_configuration->get_forwarding_logger();
}

mi::Sint32 Mdl_configuration_impl::start()
{
    m_mdlc_module.set();

    // configure implicit casts
    m_mdlc_module->set_implicit_cast_enabled(m_implicit_cast_enabled);

    // configure exposure of let-expression names
    m_mdlc_module->set_expose_names_of_let_expressions(m_expose_names_of_let_expressions);

    // configure simple-glossy legacy behavior
    mi::base::Handle<mi::mdl::IMDL> mdl(m_mdlc_module->get_mdl());

    // install external entity resolver, if given
    if (m_entity_resolver) {
        mi::base::Handle<mi::mdl::IEntity_resolver> mdl_resolver(
            new Core_entity_resolver_impl(mdl.get(), m_entity_resolver.get()));
        mdl->set_external_entity_resolver(mdl_resolver.get());
    }

    return 0;
}

mi::Sint32 Mdl_configuration_impl::shutdown()
{
    // drop any reference to a user resolver
    m_entity_resolver.reset();
    if (m_mdlc_module.get_status() == MODULE_STATUS_INITIALIZED) {
        mi::base::Handle<mi::mdl::IMDL> mdl(m_mdlc_module->get_mdl());
        if (mdl)
            mdl->set_external_entity_resolver(nullptr);
    }
    return 0;
}

void Mdl_configuration_impl::reset()
{
    m_path_module.reset();
    m_path_module.set();

    m_mdlc_module.reset();
    m_mdlc_module.set();
}

std::string Mdl_configuration_impl::get_default_mdl_system_path() const
{
#if defined(MI_PLATFORM_WINDOWS)
    return HAL::Ospath::get_windows_folder_programdata() + "\\NVIDIA Corporation\\mdl";
#elif defined(MI_PLATFORM_LINUX)
    return "/opt/nvidia/mdl";
#elif defined(MI_PLATFORM_MACOSX)
    return "/Library/Application Support/NVIDIA Corporation/mdl";
#else
#warning Unknown platform
    return "";
#endif
}

std::string Mdl_configuration_impl::get_default_mdl_user_path() const
{
#if defined(MI_PLATFORM_WINDOWS)
    return HAL::Ospath::get_windows_folder_documents() + "\\mdl";
#elif defined(MI_PLATFORM_LINUX)
    const char* home = getenv( "HOME");
    return home ? std::string( home) + "/Documents/mdl"  : "";
#elif defined(MI_PLATFORM_MACOSX)
    const char* home = getenv( "HOME");
    return home ? std::string( home) + "/Documents/mdl" : "";
#else
#warning Unknown platform
    return "";
#endif
}

} // namespace NEURAY

} // namespace MI

