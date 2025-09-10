/***************************************************************************************************
 * Copyright (c) 2009-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The implementation of both the Attribute_spec and Attribute_registry.

#include "pch.h"

#include "i_attr_registry.h"

#include <base/lib/log/i_log_logger.h>

#define INVALID_SPEC \
    Attribute_spec(null_index, "", ATTR::TYPE_UNDEF)
#define DUMMY_ID_SPEC( id ) \
    Attribute_spec(id, "", ATTR::TYPE_UNDEF)

namespace MI {
namespace ATTR {

using namespace LOG;
using namespace std;

//==================================================================================================

//--------------------------------------------------------------------------------------------------

// Constructor.
Attribute_spec::Attribute_spec(
    Uint id,
    const string& name,
    Type_code typecode,
    Uint array_size,
    const std::any& value,
    bool inheritable,
    DB::Journal_type journal_flags)
  : m_id(id),
    m_name(name),
    m_type(typecode),
    m_array_size(array_size),
    m_default(value),
    m_inheritable(inheritable),
    m_journalflags(journal_flags)
{}


//==================================================================================================

//--------------------------------------------------------------------------------------------------

// Constructor.
Attribute_registry::Attribute_registry()
: m_counter(0)
{}


//--------------------------------------------------------------------------------------------------

void Attribute_registry::reserve_ids(
    Uint count)
{
    m_counter += count;
}


//--------------------------------------------------------------------------------------------------

// Register an \c Attribute. The main functionality behind is the assignment of a correct
// new id to the new entry.
Uint Attribute_registry::add_entry(
    const string& name,
    Type_code typecode,
    Uint array_size,
    const std::any& value,
    bool inheritable,
    DB::Journal_type journal_flags)
{
    Uint id = get_new_id(name);
    return add_entry(id, name, typecode, array_size, value, inheritable, journal_flags);
}


//--------------------------------------------------------------------------------------------------

// Retrieve the \c Attribute_spec of the given \p name.
const Attribute_spec* Attribute_registry::get_attribute(
    const std::string& name) const
{
    // find id
    ankerl::unordered_dense::map<string, Uint>::const_iterator id_it = m_name_mapping.find(name);
    if (id_it == m_name_mapping.end())
        // this is not an error - all user-defined attributes will go through this
        return nullptr;

    set<Attribute_spec>::const_iterator it = m_registry.find( DUMMY_ID_SPEC(id_it->second) );
    if (it == m_registry.end()) {
        mod_log->warning(M_ATTR, Mod_log::C_DATABASE, 2,
            "Failed to find the registered Attribute %s", name.c_str());
        return nullptr;
    }
    return &(*it);
}


//--------------------------------------------------------------------------------------------------

// Retrieve the \c Attribute_spec of the given \p id.
const Attribute_spec* Attribute_registry::get_attribute(
    Uint id) const
{
    set<Attribute_spec>::const_iterator it = m_registry.find( DUMMY_ID_SPEC(id) );
    if (it == m_registry.end())
        // this is not an error - all user-defined attributes will go through this
        return 0;
    return &(*it);
}


//--------------------------------------------------------------------------------------------------

// Find a new unique id for a new registry entry.
Uint Attribute_registry::get_new_id(
    const std::string& name)
{
    return ++m_counter;
}


//--------------------------------------------------------------------------------------------------

Uint Attribute_registry::add_entry(
    Uint id,
    const std::string& name,
    Type_code typecode,
    Uint array_size,
    const std::any& value,
    bool inheritable,
    DB::Journal_type journal_flags)
{
    pair<set<Attribute_spec>::iterator, bool> result = m_registry.insert(
        Attribute_spec(
            id,
            name,
            typecode,
            array_size,
            value,
            inheritable,
            journal_flags));
    ASSERT(M_ATTR, result.second);
    if (!result.second) {
        mod_log->warning(M_ATTR, Mod_log::C_DATABASE, 1,
            "Failed to register the Attribute %s", name.c_str());
        return null_index;
    }
    add_name_mapping(name, id);
    return id;
}


//--------------------------------------------------------------------------------------------------

bool Attribute_registry::add_name_mapping(
    const std::string& name,
    Uint id)
{
    bool result = m_name_mapping.insert(std::pair<std::string,Uint>(name, id)).second;
    if (!result)
        mod_log->warning(M_ATTR, Mod_log::C_DATABASE, 1,
            "The attribute name %s does already exist.", name.c_str());
    return result;
}


//--------------------------------------------------------------------------------------------------

// Retrieve the id for a given name.
Uint Attribute_registry::get_id(
    const std::string& name)
{
    ankerl::unordered_dense::map<std::string, Uint>::const_iterator it = m_name_mapping.find(name);
    if (it != m_name_mapping.end())
        return it->second;
    else
        return null_index;
}


}
}
