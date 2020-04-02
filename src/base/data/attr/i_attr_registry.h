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
/// \brief The definition of both the Attribute_spec and Attribute_registry

#ifndef BASE_DATA_ATTR_I_ATTR_REGISTRY_H
#define BASE_DATA_ATTR_I_ATTR_REGISTRY_H

#include "i_attr_types.h"

#include <base/data/db/i_db_journal_type.h>
#include <base/system/stlext/i_stlext_any.h>
#include <base/system/stlext/i_stlext_atomic_counter.h>

#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <utility>

namespace MI {
namespace ATTR {

class Attribute_registry;
class Attribute_spec;
bool operator<(const Attribute_spec&, const Attribute_spec&);
// for friend decl only
class Attr_module_impl;

/// Specification for an \c Attribute. This characterizes the corresponding Attribute. All
/// configurable data like listsize, values, ... is left out.
/// Except the two for reserved attributes required members m_inheritable and m_journalflags.
class Attribute_spec
{
  public:
    /// Data retrieval
    //@{
    Uint get_id() const					{ return m_id; }
    const std::string& get_name() const		{ return m_name; }
    Type_code get_typecode() const			{ return m_type; }
    Uint get_array_size() const				{ return m_array_size; }
    STLEXT::Any get_default() const			{ return m_default; }
    // These two values are here for reserved attributes only, since there former storage had this
    // information attached to it.
    bool is_inheritable() const				{ return m_inheritable; }
    DB::Journal_type get_journal_flags() const		{ return m_journalflags; }
    // Support for deprecated names - for backward compatability.
    const std::string& get_deprecated_name() const	{ return m_deprecated_name; }
    //@}

  private:
    /// Constructor. Only the \c Attribute_registry can create such an object.
    Attribute_spec(
        Uint id,
        const std::string& name,
        Type_code typecode,
        Uint array_size=null_index,
        const STLEXT::Any& value=STLEXT::Any(),
        bool inheritable=true,
        DB::Journal_type journal_flags=DB::JOURNAL_NONE);

    Uint m_id;						///< id of attribute
    std::string m_name;				///< name of attribute
    std::string m_deprecated_name;			///< deprecated name
    Type_code m_type;					///< required type of sttribute
    Uint m_array_size;					///< the arraysize
    STLEXT::Any m_default;				///< its default
    // These two values are here only for reserved attributes, since there former storage had this
    // information attached to it.
    bool m_inheritable;					///< may have GLOBAL flag set
    DB::Journal_type m_journalflags;			///< what to do if value changes

    friend class Attribute_registry;
    friend bool operator<(const Attribute_spec&, const Attribute_spec&);
};

/// The comparison operator. This is actually required for handling inside a std::set insertion.
/// Based on the id.
/// \return l.m_id < r.m_id
inline bool operator<(const Attribute_spec& l, const Attribute_spec& r) { return l.m_id < r.m_id; }


/// The \c Attribute registry. This registry eases the extension of and the access to the
/// "built-in" attributes. They are registered during the setup phase.
class Attribute_registry
{
  public:
    /// Constructor.
    Attribute_registry();
    /// Since ATTR cannot know how many fixed predefined Attribute_ids are existing, the client has
    /// to set this number explicitly.
    void reserve_ids(
        Uint count);

    /// Register an \c Attribute.
    /// \return the id of the \c Attribute or null_index else
    Uint add_entry(
        const std::string& name,
        Type_code typecode,
        Uint array_size=null_index,
        const STLEXT::Any& value=STLEXT::Any(),
        bool inheritable=true,
        DB::Journal_type journal_flags=DB::JOURNAL_NONE);
    /// Add an deprecated name for the given id. Currently only one deprecated name is allowed.
    /// \return success
    bool add_deprecated_name(
        const std::string& dep_name,
        Uint id);

    /// Retrieve the \c Attribute_spec of the given \p name.
    /// \return found Attribute_spec or 0 else
    const Attribute_spec* get_attribute(
        const std::string& name) const;
    /// Retrieve the \c Attribute_spec of the given \p id.
    /// \return found Attribute_spec or 0 else
    const Attribute_spec* get_attribute(
        Uint id) const;

    /// Retrieve the id for a given name.
    /// \return the id registered under the given \p name or null_index else
    Uint get_id(
        const std::string& name);

  private:
    std::set<Attribute_spec> m_registry;		///< the actual collection
    std::map<std::string, Uint> m_name_mapping;	///< mapping name to id
    STLEXT::Atomic_counter m_counter;

    /// Find a new unique id for a new registry entry.
    /// \return the id of the \c Attribute or null_index else
    Uint get_new_id(
        const std::string& name);

    /// Register an \c Attribute.
    /// \return the id of the \c Attribute or null_index else
    Uint add_entry(
        Uint id,
        const std::string& name,
        Type_code typecode,
        Uint array_size=null_index,
        const STLEXT::Any& value=STLEXT::Any(),
        bool inheritable=true,
        DB::Journal_type journal_flags=DB::JOURNAL_NONE);

    /// Add an user-defined Attribute_id-name mapping. This should be called by
    /// \c Attribute::id_create() only.
    /// \sa Attribute::id_create
    /// \return success
    bool add_name_mapping(
        const std::string& name,
        Uint id);

    friend class Attr_module_impl;
};

}
}
#endif
