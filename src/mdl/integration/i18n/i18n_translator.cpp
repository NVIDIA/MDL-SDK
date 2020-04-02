/******************************************************************************
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
 *****************************************************************************/
#include "pch.h"

#include <set>

#include "i18n_db.h"
#include "i_i18n.h"
#include "i18n_translator.h"
#include <base/system/main/access_module.h>
#include <base/system/main/module_registration.h>
#include <base/lib/log/i_log_assert.h>

using MI::MDL::I18N::Mdl_translator_impl;
using MI::MDL::I18N::Mdl_translator_module;
using std::string;

static MI::SYSTEM::Module_registration<Mdl_translator_impl> s_module(
    MI::SYSTEM::M_I18N, Mdl_translator_module::get_name());

MI::SYSTEM::Module_registration_entry* Mdl_translator_module::get_instance()
{
    return s_module.init_module(s_module.get_name());
}

Mdl_translator_impl::Mdl_translator_impl()
    : m_database(NULL)
{
}

Mdl_translator_impl::~Mdl_translator_impl()
{
}

bool Mdl_translator_impl::init()
{
    m_database = new Database;
    return true;
}

void Mdl_translator_impl::exit()
{
    delete m_database;
}

bool Mdl_translator_impl::need_translation(const std::string & name) const
{
#ifdef _DEBUG
    static std::set<std::string> ignored; // For debug purpose
    static std::set<std::string> found; // For debug purpose
#endif
    // WARNING: To keep in sync with:
    //      prod\bin\i18n\xliff.cpp
    //      mdl\integration\i18n\i_i18n.h
    //      mdl\integration\i18n\i18n_translator.h
    static std::set<std::string> translation =
    {
          "::anno::display_name(string)"
        , "::anno::in_group(string)"
        , "::anno::in_group(string,string)"
        , "::anno::in_group(string,string,string)"
        , "::anno::key_words(string[N])"
        , "::anno::copyright_notice(string)"
        , "::anno::description(string)"
        , "::anno::author(string)"
        , "::anno::contributor(string)"
        , "::anno::unused(string)"
        , "::anno::deprecated(string)"
    };

    bool need(false);
    if (translation.find(name) != translation.end())
    {
        need = true;
    }

#ifdef _DEBUG
    // Help to place breakpoints to test specific annotations
    if (need && found.find(name) == found.end())
    {
        found.insert(name);
    }
    if (!need && ignored.find(name) == ignored.end())
    {
        ignored.insert(name);
    }
#endif

    return need;
}

mi::Sint32 Mdl_translator_impl::translate(Mdl_translator_module::Translation_unit & sentence) const
{
    string locale;
    if (!is_localization_set(locale))
    {
        return -2;
    }
    sentence.set_locale(locale);
    ASSERT(M_I18N, m_database != NULL);
    return m_database->translate(sentence);
}

mi::Sint32 Mdl_translator_impl::set_locale(const string & locale)
{
    m_locale = locale;
    return 0;
}

bool Mdl_translator_impl::is_localization_set(std::string & locale) const
{
    bool localization_set(!m_locale.empty());
    if (localization_set)
    {
        locale = m_locale;
    }
    return localization_set;
}

mi::Sint32 Mdl_translator_impl::cleanup_database()
{
    ASSERT(M_I18N, m_database != NULL);
    return m_database->cleanup();
}

void Mdl_translator_module::Translation_unit::set_module_name(const char * module_name)
{
    if (module_name)
    {
        m_module_name = module_name;
    }
    else
    {
        m_module_name = "";
    }
}

const char * Mdl_translator_module::Translation_unit::get_module_name() const
{
    return m_module_name.empty() ? NULL : m_module_name.c_str();
}

void Mdl_translator_module::Translation_unit::set_context(const std::string & context)
{
    m_context = context;
}

const std::string & Mdl_translator_module::Translation_unit::get_context() const
{
    return m_context;
}

void Mdl_translator_module::Translation_unit::set_source(const std::string & source)
{
    m_source = source;
}

const std::string & Mdl_translator_module::Translation_unit::get_source() const
{
    return m_source;
}

void Mdl_translator_module::Translation_unit::set_target(const std::string & target)
{
    m_target = target;
}

const std::string & Mdl_translator_module::Translation_unit::get_target() const
{
    return m_target;
}

void Mdl_translator_module::Translation_unit::set_locale(const std::string & locale)
{
    m_locale = locale;
}

const std::string & Mdl_translator_module::Translation_unit::get_locale() const
{
    return m_locale;
}
