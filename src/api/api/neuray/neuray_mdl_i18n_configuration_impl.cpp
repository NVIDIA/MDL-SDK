/***************************************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Implementation of IMdl_i18n_configuration
 **
 ** Implements the IMdl_i18n_configuration interface
 **/

#include "pch.h"

#include "neuray_mdl_i18n_configuration_impl.h"
#include <mdl/integration/i18n/i_i18n.h>
#include <base/lib/log/i_log_assert.h>
#include <string>
using std::string;

namespace MI {

namespace NEURAY {

const string Mdl_i18n_configuration_impl::system_keyword("{SYSTEM}");

Mdl_i18n_configuration_impl::Mdl_i18n_configuration_impl(mi::neuraylib::INeuray* neuray_impl)
    : m_neuray_impl(neuray_impl)
    , m_i18n_module(false/*deferred*/) // Need to set before neuray starts
    //, m_i18n_module(true/*deferred*/)
{
    set_locale(system_keyword.c_str());
}

Mdl_i18n_configuration_impl::~Mdl_i18n_configuration_impl()
{
    // Tear down the module.
    m_i18n_module.reset();
}

mi::Sint32 Mdl_i18n_configuration_impl::set_locale( const char* locale)
{
    using namespace mi::neuraylib;
    const INeuray::Status status = m_neuray_impl->get_status();
    const bool valid_call = (status == INeuray::PRE_STARTING || status == INeuray::SHUTDOWN);

    if (!valid_call)
    {
        /// This function can only be called before neuray has been started.
        /// Or after neuray has been shutdown.
        return -1;
    }
    string translator_locale; //Locale to set in the translator
    if (!locale)
    {
        m_locale = "";
    }
    else
    {
        if (system_keyword == locale)
        {
            m_locale = system_keyword;
            // Get value of the system locale
            const char * system_locale = get_system_locale();
            if (system_locale)
            {
                translator_locale = system_locale;
            }
        }
        else
        {
            m_locale = locale;
            translator_locale = m_locale;            
        }
    }
    ASSERT(M_I18N, m_i18n_module.is_module_initialized());
    m_i18n_module->set_locale(translator_locale);
    return 0;
}

const char* Mdl_i18n_configuration_impl::get_locale() const
{
    return m_locale.empty() ? nullptr : m_locale.c_str();
}

const char* Mdl_i18n_configuration_impl::get_system_locale() const
{
    // NOTE TODO: setlocal() fails to return the value of LC_ALL env var, not sure why
    // Get the env variable using getenv() instead for the time being
    const char * locale_set = nullptr;
    //locale_set = setlocale(LC_ALL, NULL);
    //if (locale_set == NULL)
    //{
    //    return false;
    //}

    locale_set = getenv("LC_ALL");
    if (!locale_set)
    {
        return nullptr;
    }
    if (string(locale_set) == "C")
    {
        return nullptr;
    }
    return locale_set;
}

const char* Mdl_i18n_configuration_impl::get_system_keyword() const
{
    return system_keyword.c_str();
}

mi::Sint32 Mdl_i18n_configuration_impl::start()
{
    // Setup the module.
    //m_i18n_module.set();
    //set_locale(system_keyword.c_str());
    return 0;
}

mi::Sint32 Mdl_i18n_configuration_impl::shutdown()
{
    return m_i18n_module->cleanup_database();
}

} // namespace NEURAY

} // namespace MI
