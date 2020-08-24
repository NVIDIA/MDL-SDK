/***************************************************************************************************
 * Copyright (c) 2009-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Attribute system's module

#include "pch.h"

#include "attr_module.h"

#include <base/data/serial/serial.h>
#include <base/system/main/module_registration.h>
#include <base/util/string_utils/i_string_utils.h>
#include <string>

namespace MI {
namespace ATTR {

using std::string;

// Module registration.
static SYSTEM::Module_registration<Attr_module_impl> s_module(SYSTEM::M_ATTR, "ATTR");
// Definition.
Attr_module_impl* Attr_module_impl::s_attr_module = 0;


//==================================================================================================

// Module creation.
SYSTEM::Module_registration_entry* Attr_module::get_instance()
{
    return s_module.init_module(s_module.get_name());
}


void Attr_module::retrieve_reserved_attr_default(
    const std::string& name,
    STLEXT::Any& any) const
{
    Attribute_id id = Attribute::id_lookup(name.c_str());

    const Attribute_spec* spec = this->get_reserved_attr_spec(id);
    if (spec)
        any = spec->get_default();
}


//==================================================================================================

// Give names to the reserved flag attributes (they are defined above BASE).
void Attr_module_impl::set_reserved_attr(
    Attribute_id id,
    const char* name,
    const Type_code tc,
    const DB::Journal_type flags,
    bool inheritable,
    const STLEXT::Any& def)
{
    // fill new registry accordingly
    m_registry.add_entry(id, name? name : string(), tc, null_index, def, inheritable, flags);
}

// Add additional, deprecated name to an attribute.
void Attr_module_impl::set_deprecated_attr_name(
    Attribute_id id,
    const char* name)
{
    // update new registry accordingly
    m_registry.add_deprecated_name(name? name : string(), id);
}

// Return deprecated name, or 0 if there is none.
const char* Attr_module_impl::get_deprecated_attr_name(
    Attribute_id id)
{
    // check all reserved Attributes
    const Attribute_spec* const spec = m_registry.get_attribute(id);
    if (!spec || spec->get_deprecated_name().empty())
        return 0;

    return spec->get_deprecated_name().c_str();
}

// Register journal-flags for all user attributes.
void Attr_module_impl::set_user_attr_journal_flags(
    const DB::Journal_type flags)
{
    m_user_attr_journal_flags = flags;
}

// Return previously set names of the reserved flag attributes.
const char* Attr_module_impl::get_reserved_attr(
    Attribute_id id,
    Type_code* tc/*=0*/,
    DB::Journal_type* jf/*=0*/,
    bool* inh/*=0*/) const
{
    const char* result = 0;
    const Attribute_spec* const spec = m_registry.get_attribute(id);
    if (spec) {
        if (tc)  *tc  = spec->get_typecode();
        if (jf)  *jf  = spec->get_journal_flags();
        if (inh) *inh = spec->is_inheritable();
        result = spec->get_name().c_str();
    }
    else {
        if (tc)  *tc  = TYPE_UNDEF;	// don't know
        if (jf)  *jf  = m_user_attr_journal_flags;
        if (inh) *inh = true;
    }

    return result;
}

// Retrieve the Attribute_spec for the given id.
const Attribute_spec* Attr_module_impl::get_reserved_attr_spec(
    Attribute_id id) const
{
    return m_registry.get_attribute(id);
}

// Look up a type code given a type name.
ATTR::Type_code Attr_module_impl::get_type_code(
    const std::string& type_name) const
{
    Map_name_to_code::const_iterator iter =
        m_map_name_to_code.find(type_name);

    if (iter == m_map_name_to_code.end()) // not found
    {
        // try lower case
        string lower_case = STRING::to_lower(type_name);

        iter = m_map_name_to_code.find(lower_case);

        if (iter == m_map_name_to_code.end()) // not found
            return ATTR::TYPE_UNDEF;
    }

    return iter->second;
}

// Build the type name to type code map.
void Attr_module_impl::build_name_to_code_map()
{
    const int num_types(ATTR::TYPE_NUM);
    for (int i = 0; i < num_types; ++i)
    {
        const ATTR::Type_code type_code = ATTR::Type_code(i);
        string type_name = STRING::to_lower(ATTR::Type::type_name(type_code));
        m_map_name_to_code[type_name] = type_code;
    }

    // Two special cases (see attr_type.cpp) :
    // - TYPE_RGBA_FP is an alias for TYPE_COLOR.
    // - TYPE_MATRIX4X4 is an alias for TYPE_MATRIX.
    m_map_name_to_code["rgba_fp"] = ATTR::TYPE_RGBA_FP;
    m_map_name_to_code["matrix4x4"] = ATTR::TYPE_MATRIX4X4;
}

const Attr_module_impl::Custom_attr_filters& Attr_module_impl::get_custom_attr_filters() const
{
    return m_custom_attr_filters;
}

static bool build_regex(const std::string& regex_str, std::wregex& out_regex)
{
    try {
        std::wstring m_custom_attr_filter_wstr = STRING::utf8_to_wchar(
            regex_str.c_str());
        out_regex.assign(m_custom_attr_filter_wstr, std::wregex::extended);
    }
    catch (const std::regex_error&) {
        return false;
    }
    return true;
}

bool Attr_module_impl::add_custom_attr_filter(const std::string& filter)
{
    auto p = std::find(m_custom_attr_filters.begin(), m_custom_attr_filters.end(), filter);
    if (p != m_custom_attr_filters.end()) {
        return false;
    }
    const std::string filter_expr = (m_custom_attr_filters.size() == 0 ? "(" : " | (")  + filter + ")";
    if (!build_regex(
        m_custom_attr_filter + filter_expr, m_custom_attr_regex)) {
        return false;
    }
    m_custom_attr_filters.push_back(filter);
    m_custom_attr_filter += filter_expr;
    return true;
}

bool Attr_module_impl::remove_custom_attr_filter(const std::string& filter)
{
    auto p = std::find(m_custom_attr_filters.begin(), m_custom_attr_filters.end(), filter);
    if (p != m_custom_attr_filters.end()) {
        m_custom_attr_filters.erase(p);

        // update regex
        m_custom_attr_filter.clear();
        m_custom_attr_regex = std::wregex();

        if (!m_custom_attr_filters.empty()) {
            m_custom_attr_filter += "(" + m_custom_attr_filters[0] + ")";
            for (size_t i = 1; i < m_custom_attr_filters.size(); ++i) {
                m_custom_attr_filter += "| (" + m_custom_attr_filters[i] + ")";
            }
            build_regex(m_custom_attr_filter, m_custom_attr_regex);
        }
        return true;
    }
    return false;
}

void Attr_module_impl::clear_custom_attr_filters()
{
    m_custom_attr_filters.clear();
    m_custom_attr_filter.clear();
    m_custom_attr_regex = std::wregex();
}

const std::wregex& Attr_module_impl::get_custom_attr_filter() const
{
    return m_custom_attr_regex;
}

// Fully initialize the module such that its members can register themselves.
void Attr_module_impl::register_for_serialization(
    SERIAL::Deserialization_manager* dm)
{
    if (dm) {
        dm->register_class(Attribute_set::id, Attribute_set::factory);
        dm->register_class(Attribute	::id, Attribute	   ::factory);
        dm->register_class(Type		::id, Type	   ::factory);
    }
}

// Setup module.
bool Attr_module_impl::init()
{
    build_name_to_code_map();

    s_attr_module = this;
    return true;
}

// Finalize module.
void Attr_module_impl::exit()
{
    s_attr_module = 0;
}

}
}
