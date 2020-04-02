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
/// \brief The configuration registry implementation.

#ifndef BASE_UTIL_CONFIG_CONFIG_REGISTRY_IMPL_H
#define BASE_UTIL_CONFIG_CONFIG_REGISTRY_IMPL_H

#include "i_config_registry.h"

#include <base/system/stlext/i_stlext_any.h>
#include <map>
#include <string>

namespace MI {
namespace CONFIG {

/// The configuration registry impl.
class Config_registry_impl : public Config_registry
{
  public:
    /// Destructor.
    ~Config_registry_impl();

    /// Registering a value.
    /// \param name name of the value
    /// \param value the value
    /// \return success of registration
    virtual bool add_value(
	const std::string& name,
	const STLEXT::Any& value);
    /// Registering a value into a list of values of the same type.
    /// \param name name of the value
    /// \param value the value
    /// \return success of registration
    virtual bool add_value_multiple(
	const std::string& name,
	const STLEXT::Any& value);
    /// Update, ie overwrite the existing value with the given one.
    /// \param name name of the value
    /// \param value the value
    virtual void overwrite_value(
	const std::string& name,
	const STLEXT::Any& value);

    /// Retrieving a value.
    /// \param name name of the value
    /// \return its value or the empty type iff not registered
    STLEXT::Any get_value(
	const std::string& name) const;

    /// Retrieve the start of the registry.
    virtual Const_iter begin() const { return m_values.begin(); }
    /// Retrieve the end of the registry.
    virtual Const_iter end() const { return m_values.end(); }

  private:
    std::map<std::string, STLEXT::Any> m_values;	///< the stored registered values
};

}
}

#endif
