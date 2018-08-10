/******************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_DAG_TYPE_COLLECTOR_H
#define MDL_GENERATOR_DAG_TYPE_COLLECTOR_H

#include "mdl/compiler/compilercore/compilercore_cc_conf.h"
#include "mdl/compiler/compilercore/compilercore_memory_arena.h"
#include "mdl/compiler/compilercore/compilercore_factories.h"

namespace mi {
namespace mdl {

class IType;
class IType_array;
class IType_atomic;
class Type_factory;

/// Helper value type class for collecting indexable types.
class Indexable {
public:
    /// Constructor.
    ///
    /// \param type                   the indexable type
    /// \param is_array_element_type  if true, the indexable type is type[], else type itself
    /// \param is_exported            true if this type is visible from exported interface
    Indexable(
        IType const *type,
        bool        is_array_element_type,
        bool        is_exported)
    : m_type(type)
    , m_is_array_element_type(is_array_element_type)
    , m_is_exported(is_exported)
    {
    }

    /// Return the indexable type.
    IType const *get_type() const { return m_type; }

    /// If returns true, the indexable type is an deferred array.
    bool is_array_element_type() const { return m_is_array_element_type; }

    /// Returns true if this type is visible from the exported interface.
    bool is_exported() const { return m_is_exported; }

private:
    /// The indexable type.
    IType const *m_type;

    /// If true, the indexable type is m_type[], else m_type itself
    bool        m_is_array_element_type;

    /// True, iff this indexable type is visible from exported interface.
    bool        m_is_exported;
};

/// Helper class to collect various aspects of the types found inside a module.
class Type_collector {
    /// The type of sets of types.
    typedef ptr_hash_set<IType const>::Type Type_set;

    /// The type of indexable types.
    typedef vector<Indexable>::Type Indexable_vector;

    /// The type of vectors of types.
    typedef vector<IType const *>::Type Type_vector;

public:
    /// Constructor.
    ///
    /// \param alloc          the allocator
    /// \param type_factory   the type factory of the DAG code
    Type_collector(IAllocator *alloc, Type_factory &type_factory);

    /// Collect exported types supporting operator[].
    ///
    /// \param type            a type that can be accessed through an export
    /// \param is_exported     true if can be access trough the exported interface,
    ///                        false if only visible through inlining
    void collect_indexable_types(
        IType const *type,
        bool        is_exported);

    /// Collect all indexable builtin types.
    void collect_builtin_types();

    /// Add a type to the "defined type" list.
    ///
    /// \param tp  the type to add
    void add_defined_type(IType const *tp);

    /// Check if the given type is already "defined".
    ///
    /// \param type  the type to check
    bool is_defined_type(IType const *type) const;

    /// Get the number of indexables.
    size_t indexable_type_size() const {
        return m_indexable_type_list.size();
    }

    /// Get the number of array types.
    size_t array_constructor_type_size() const {
        return m_array_constructor_types.size();
    }

    /// Get the number of defined types.
    size_t defined_type_size() const {
        return m_defined_type_list.size();
    }

    /// Get the indexable at given index.
    ///
    /// \param i  the index of the indexable to retrieve
    Indexable const &get_indexable(size_t i) const {
        return m_indexable_type_list[i];
    }

    /// Get the array constructor type at given index.
    ///
    /// \param i  the index of the array constructor type to retrieve
    IType const *get_array_constructor_type(size_t i) const
    {
        return m_array_constructor_types[i];
    }

    /// Check if the given type is a known array constructor type.
    ///
    /// \param type  the type to check
    bool is_array_constructor_type(IType const *type) const;

    /// Get the defined type at given index.
    ///
    /// \param i  the index of the defined type to retrieve
    IType const *get_defined_type(size_t i) const {
        return m_defined_type_list[i];
    }

    /// Check if a given array type is known by this collector.
    ///
    /// \param a_type  the array type
    bool known_array_type(IType_array const *a_type) const;

    /// Import a type into the type factory of the type collector.
    ///
    /// \param type  the type to be imported
    IType const *import(IType const *type) {
        return m_type_factory.import(type);
    }

    /// Import a type into the type factory of the type collector.
    ///
    /// \param type  the type to be imported
    template<typename T>
    T const *import(T const *type) {
        return static_cast<T const *>(m_type_factory.import(type));
    }

private:
    /// Collect all indexable vector types of a base type.
    ///
    /// \param base_type  the base type
    void collect_vector_types(IType_atomic const *base_type);

    /// Collect all indexable matrix types of a base type.
    ///
    /// \param base_type  the base type
    void collect_matrix_types(IType_atomic const *base_type);

private:
    /// The set of all already visited indexable types.
    Type_set         m_indexable_types;

    /// The set of all already visited array element types.
    Type_set         m_array_elements;

    /// The set of all already visited array constructor types.
    Type_set         m_array_constrs;

    /// The set of all defined types.
    Type_set         m_defined_types;

    /// The list of all collected indexable types.
    Indexable_vector m_indexable_type_list;

    /// The array constructor types.
    Type_vector      m_array_constructor_types;

    /// All defined types of this module.
    Type_vector      m_defined_type_list;

    /// The type factory of the DAG.
    Type_factory     &m_type_factory;
};

} // mdl
} // mi

#endif
