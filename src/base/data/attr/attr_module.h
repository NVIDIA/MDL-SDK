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

#ifndef BASE_DATA_ATTR_ATTR_MODULE_H
#define BASE_DATA_ATTR_ATTR_MODULE_H

#include "attr.h"
#include <boost/unordered_map.hpp>
#include <string>
#include <vector>

namespace MI {
namespace ATTR {

/// The implementation class of the \c Attr_module.
class Attr_module_impl : public Attr_module
{
  public:
    /// Create the module.
    /// \return success
    bool init();
    /// Shut down this module.
    void exit();

    /// Fully initialize the module such that its members can register themselves.
    /// \param dm a deserialization manager
    void register_for_serialization(
        SERIAL::Deserialization_manager* dm);

    /// Give names to the reserved flag attributes (they are defined above BASE).
    /// \param id give this attribute ID a name
    /// \param name new attr name, literal, not copied
    /// \param tc simple type of attribute
    /// \param flags if attr changes, put these flags
    /// \param inh inheritable, may have GLOBAL flag
    /// \param def default value for this Attributes
    void set_reserved_attr(
        Attribute_id id,
        const char* name,
        const Type_code tc,
        const DB::Journal_type flags,
        bool inh,
        const STLEXT::Any& def=STLEXT::Any());

    /// Some attributes have deprecated names that should also work, and map to
    /// the same IDs. For example, "sample_max" is now "samples".
    /// \param id give this ID an alternate name
    /// \param name new attr name, literal, not copied
    void set_deprecated_attr_name(
        Attribute_id id,
        const char* name);

    /// Return deprecated name, or 0 if there is none.
    /// \param id return alternate name of this ID
    /// \return deprecated name, or 0 if there is none.
    const char* get_deprecated_attr_name(
        Attribute_id id);


    /// Register journal-flags for all user attributes.
    /// For now, all user attributes have the same journal flags
    /// SCENE::JOURNAL_CHANGE_SHADER.
    /// \param flags if attr changes, put journal flags
    void set_user_attr_journal_flags(
        const DB::Journal_type flags);


    /// Return previously set names of the reserved flag attributes.
    /// \param id return name of this attribute ID
    /// \param tc return simple type of attr if nz
    /// \param jf return journal flags of attr if nz
    /// \param inh false if attr is never inheritable
    /// \return previously set names of the reserved flag attributes.
    const char* get_reserved_attr(
        Attribute_id id,
        Type_code* tc=0,
        DB::Journal_type* jf=0,
        bool* inh=0) const;

    /// Retrieve the \c Attribute_spec for the given \p id.
    /// \param id the \c Attribute_id of the reserved \c Attribute
    /// \return the corresponding \c Attribute_spec or 0 else
    const Attribute_spec* get_reserved_attr_spec(
        Attribute_id id) const;

    /// Look up a type code given a type name. This uses a hash map, so it should
    /// be pretty fast. Returns \c TYPE_UNDEF if the look up fails.
    /// \param type_name name of the type
    /// \return \c Type_code, \c TYPE_UNDEF if the look up fails.
    ATTR::Type_code get_type_code(
        const std::string& type_name) const;

    const Custom_attr_filters& get_custom_attr_filters() const;

    bool add_custom_attr_filter(const std::string& filter);

    bool remove_custom_attr_filter(const std::string& filter);

    void clear_custom_attr_filters();

    const std::wregex& get_custom_attr_filter() const;

  private:
    static Attr_module_impl* s_attr_module;	///< internal pointer to the module
    Attribute_registry m_registry;		///< internal registry for "built-in" types
    DB::Journal_type m_user_attr_journal_flags;	///< the default journal-flags for all user attrs

    /// Mapping from type name to type code, used in \c get_type_code.
    typedef boost::unordered_map<std::string, ATTR::Type_code> Map_name_to_code;
    Map_name_to_code m_map_name_to_code;

    /// List of installed attribute filters.
    Custom_attr_filters m_custom_attr_filters;

    /// Regex string assembled from filters above.
    std::string m_custom_attr_filter;
   /// Regex object assembled from filters above.
    std::wregex m_custom_attr_regex;
   
    /// Build the type name to type code map.
    void build_name_to_code_map();

    friend class Attribute;
};

}
}
#endif
