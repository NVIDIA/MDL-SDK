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
/// \brief The configuration registry interface.

#ifndef BASE_UTIL_CONFIG_I_CONFIG_REGISTRY_H
#define BASE_UTIL_CONFIG_I_CONFIG_REGISTRY_H

#include <base/util/string_utils/i_string_lexicographic_cast.h>
#include <base/system/stlext/i_stlext_any.h>

#include <iosfwd>
#include <map>
#include <string>

namespace MI {
namespace SITE { class Rs_hostconfig; }
namespace CONFIG {

/// The configuration interface. This interface allows the retrieval of all registered
/// entries. The idea is that it might be passed to the modules which have to retrieve the
/// entries they know about, to retrieve values set in config files, on the cmdline, etc.
/// First, it gets all values from the cmdline set. Second, all entries from the config or
/// initrc file get added, iff they weren't already set via cmdline.
class Config_registry
{
public:
    virtual ~Config_registry() {}

    /// Registering a value.
    /// \param name name of the value
    /// \param value the value
    /// \return success of registration
    virtual bool add_value(
	const std::string& name,
	const STLEXT::Any& value) = 0;
    /// Registering a value into a list of values of the same type.
    /// \param name name of the value
    /// \param value the value
    /// \return success of registration
    virtual bool add_value_multiple(
	const std::string& name,
	const STLEXT::Any& value) = 0;
    /// Retrieving a value.
    /// \param name name of the value
    /// \return its value or the empty type iff not registered
    virtual STLEXT::Any get_value(
	const std::string& name) const = 0;
    /// Retrieving a value and checking whether it was stored as string- or float-typed value.
    /// Certain values are added as string-typed values, eg when given via
    ///   --config SOFT_shader_language=2
    /// the cmdline parser has no idea of what type that given value is. Values from the initrc
    /// which are not strings are stored as float-typed values.
    ///
    /// Now we have turned this member into a safe way to check both for the correct typed value
    /// and the string-typed value and the float-typed value.
    /// Currently disabled! In addition to that, check for the name prefixed by the module.
    template <typename T> bool get_value(
	const std::string& name,
	T& val) const
    {
	if (this->get_typed_value(name, val))
	    return true;

	// now try to match with other "similiar" types
	STLEXT::Any any = get_value(name);
	if (!any.empty()) {
	    if (any.type() == typeid(std::string)) {
		std::string value = *STLEXT::any_cast<std::string>(&any);
		STLEXT::Likely<T> v = STRING::lexicographic_cast_s<T, std::string>(value);
		if (v.get_status()) {
		    val = v;
		    return true;
		}
	    }
	    // try the float type as the last resort - see the comments above!
	    else if (any.type() == typeid(float)) {
		float* value = STLEXT::any_cast<float>(&any);
		if (value) {
		    val = static_cast<T>(*value);
		    return true;
		}
	    }
	}
	return false;
    }

    /// Retrieving a value. This is the typesafe version of \c get_value(). Ie it does
    /// not try to match with other types.
    template <typename T> bool get_typed_value(
	const std::string& name,
	T& val) const
    {
	STLEXT::Any any = get_value(name);
	if (!any.empty()) {
	    if (any.type() == typeid(T)) {
		val = *STLEXT::any_cast<T>(&any);
		return true;
	    }
	}
	return false;
    }

    /// \name Iteration support
    /// Iteration support.
    //@{
    typedef std::map<std::string, STLEXT::Any>::const_iterator Const_iter;
    /// Retrieve the start of the registry.
    virtual Const_iter begin() const = 0;
    /// Retrieve the end of the registry.
    virtual Const_iter end() const = 0;
    //@}

    /// Update, ie overwrite the existing value with the given one.
    /// \param name name of the value
    /// \param value the value
    virtual void overwrite_value(
	const std::string& name,
	const STLEXT::Any& value) = 0;
};

/// Writing out the current values.
std::ostream& operator<<(std::ostream& os, const Config_registry& configuration);


// Member template specializations have to be defined outside of the class.
template <>
inline bool Config_registry::get_value(
    const std::string& name,
    bool& val) const
{
    STLEXT::Any any = get_value(name);
    if (!any.empty()) {
	if (any.type() == typeid(bool)) {
	    val = *STLEXT::any_cast<bool>(&any);
	    return true;
	}
	else if (any.type() == typeid(std::string)) {
	    std::string value = *STLEXT::any_cast<std::string>(&any);
	    STLEXT::Likely<bool> v = STRING::lexicographic_cast_s<bool, std::string>(value);
	    if (v.get_status()) {
		val = v;
		return true;
	    }
	}
	// try the float type as the last resort - T should be either int or bool!
	else {
	    float* value = STLEXT::any_cast<float>(&any);
	    if (value) {
		// this is to avoid the "forcing value to bool" warning on VC++
		val = !!*value;
		return true;
	    }
	}
    }
    return false;
}

template <>
inline bool Config_registry::get_value(
    const std::string& name,
    std::string& val) const
{
    STLEXT::Any any = get_value(name);
    if (!any.empty()) {
	if (any.type() == typeid(std::string)) {
	    val = *STLEXT::any_cast<std::string>(&any);
	    return true;
	}
    }
    return false;
}


/// A helper utility for setting a variable. Its implementation relies simply on
/// the \c Config_registry::get_value() functionality.
/// \return true, when the \p variable was set, false else
template <typename T>
inline bool update_value(
    const CONFIG::Config_registry& registry,
    const std::string& name,
    T& variable)
{
    return registry.get_value<T>(name, variable);
}

/// A helper utility for setting a variable. Its implementation relies simply on
/// the \c Config_registry::get_typed_value() functionality.
/// \return true, when the \p variable was set, false else
template <typename T>
inline bool update_typed_value(
    const CONFIG::Config_registry& registry,
    const std::string& name,
    T& variable)
{
    return registry.get_typed_value<T>(name, variable);
}


/// A helper function for interpreting a string as a boolean value.
/// "on", "true", "1", "yes" get interpreted as true.
bool as_bool(const std::string& str);

}
}
#endif
