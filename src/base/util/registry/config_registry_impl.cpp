/***************************************************************************************************
 * Copyright (c) 2009-2019, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The configuration registry implementation.

#include "pch.h"

#include "config_registry_impl.h"

#include <base/util/string_utils/i_string_lexicographic_cast.h>
#include <cctype>
#include <ostream>
#include <vector>

namespace MI {
namespace CONFIG {

Config_registry_impl::~Config_registry_impl()
{}


// Retrieving a value.
// \param name name of the value
// \return its value or the empty type iff not registered
STLEXT::Any Config_registry_impl::get_value(
    const std::string& name) const
{
    std::map<std::string, STLEXT::Any>::const_iterator it = m_values.find(name);

    if (it != m_values.end())
	return it->second;
    else
	return STLEXT::Any();
}


// Registering a value.
// \param name name of the value
// \param value the value
bool Config_registry_impl::add_value(
    const std::string& name,
    const STLEXT::Any& value)
{
    std::pair<std::map<std::string, STLEXT::Any>::iterator, bool> result =
	m_values.insert(std::make_pair(name, value));
/*
    // this would add dependency on base/lib/log!
    if (!result.second)
	LOG::mod_log->debug(M_CONFIG, LOG::Mod_log::C_DISK,
	    "insertion of value %s into config registry failed", name.c_str());
 */
    return result.second;
}


// Registering a value into a list of values of the same type.
// \param name name of the value
// \param value the value
// \return success of registration
bool Config_registry_impl::add_value_multiple(
    const std::string& name,
    const STLEXT::Any& value)
{
    typedef std::vector<STLEXT::Any> Many;

    if (value.empty()) {
	// we do not support containers of void type
	return false;
    }

    bool result = false;

    std::map<std::string, STLEXT::Any>::iterator it = m_values.find(name);
    if (it != m_values.end()) {
	Many* val = STLEXT::any_cast<Many>(&it->second);
	if (!val)
	    // we do have a problem!
	    result = false;
	else {
	    val->push_back(value);
	    result = true;
	}
    }
    else {
	Many val;
	val.push_back(value);
	result = add_value(name, STLEXT::Any(val));
    }
    return result;
}


// Update, ie overwrite the existing value with the given one.
// \param name name of the value
// \param value the value
void Config_registry_impl::overwrite_value(
    const std::string& name,
    const STLEXT::Any& value)
{
    std::map<std::string, STLEXT::Any>::iterator it = m_values.find(name);
    if (it != m_values.end())
	m_values.erase(it);
    add_value(name, value);
}


void stream_any(std::ostream& os, const STLEXT::Any& any)
{
    if (any.type() == typeid(int)) {
	const int* v = STLEXT::any_cast<int>(&any);
	if (v)
	    os << *v;
    }
    else if (any.type() == typeid(float)) {
	const float* v = STLEXT::any_cast<float>(&any);
	if (v)
	    os << *v;
    }
    else if (any.type() == typeid(bool)) {
	const bool* v = STLEXT::any_cast<bool>(&any);
	if (v)
	    os << std::boolalpha << *v;
    }
    else if (any.type() == typeid(std::string)) {
	const std::string* v = STLEXT::any_cast<std::string>(&any);
	if (v)
	    os << v->c_str();
    }
    else if (any.type() == typeid(std::vector<STLEXT::Any>)) {
	os << '[';
	const std::vector<STLEXT::Any>* v =
	    STLEXT::any_cast<std::vector<STLEXT::Any> >(&any);
	if (v) {
	    for (size_t i=0; i<v->size(); ++i) {
		if (i)
		    os << ", ";
		stream_any(os, (*v)[i]);
	    }
	}
	os << ']';
    }
}

/// Writing out the current values.
std::ostream& operator<<(std::ostream& os, const Config_registry& configuration)
{
    const Config_registry_impl& config = static_cast<const Config_registry_impl&>(configuration);
    std::map<std::string, STLEXT::Any>::const_iterator it, end=config.end();
    for (it=config.begin(); it != end; ++it) {
	os << "Value \"" << it->first.c_str() << "\" ";
	stream_any(os, it->second);
	os << std::endl;
    }
    return os;
}

namespace {
// Convert the given string \p input to a string where all characters are lowercase.
void to_lower(
    std::string& input)
{
    // using the global ::tolower() as the STLport implementation is breaking on MAC OSX.
    std::transform(input.begin(), input.end(), input.begin(), ::tolower);
}
}

bool as_bool(const std::string& str)
{
    // try interpreting the string as a number
    const STLEXT::Likely<double> num(STRING::lexicographic_cast_s<double>(str));
    if (num.get_status()) // valid conversion
        return static_cast<bool>(static_cast<double>(num) != 0);

    // it's not a number, check strings
    std::string lower(str);
    to_lower(lower);
    return lower == "true" || lower == "on" || lower == "yes";
}

}
}
