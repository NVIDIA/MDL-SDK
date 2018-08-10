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

#include "pch.h"

#include "generator_dag_type_collector.h"
#include "mdl/compiler/compilercore/compilercore_tools.h"

namespace mi {
namespace mdl {

// Constructor.
Type_collector::Type_collector(
    IAllocator   *alloc,
    Type_factory &type_factory)
: m_indexable_types(0, Type_set::hasher(), Type_set::key_equal(), alloc)
, m_array_elements(0, Type_set::hasher(), Type_set::key_equal(), alloc)
, m_array_constrs(0, Type_set::hasher(), Type_set::key_equal(), alloc)
, m_defined_types(0, Type_set::hasher(), Type_set::key_equal(), alloc)
, m_indexable_type_list(alloc)
, m_array_constructor_types(alloc)
, m_defined_type_list(alloc)
, m_type_factory(type_factory)
{
}

// Collect exported types supporting operator[].
void Type_collector::collect_indexable_types(
    IType const *type,
    bool        is_exported)
{
    type = type->skip_type_alias();
    switch (type->get_kind()) {
    case IType::TK_ARRAY:
        {
            IType_array const *a_type = cast<IType_array>(type);
            IType const       *e_type = a_type->get_element_type()->skip_type_alias();
            collect_indexable_types(e_type, is_exported);
            if (a_type->is_immediate_sized()) {
                // create one access for EVERY immediate sized array
                if (m_indexable_types.insert(type).second)
                    m_indexable_type_list.push_back(Indexable(a_type, false, is_exported));
            } else {
                // create only ONE access for all deferred arrays of the
                // same element type
                if (m_array_elements.insert(e_type).second)
                    m_indexable_type_list.push_back(Indexable(e_type, true, is_exported));
            }
            // create one array constructor for all kind of arrays
            if (m_array_constrs.insert(e_type).second)
                m_array_constructor_types.push_back(e_type);
        }
        break;
    case IType::TK_STRUCT:
        {
            // check member field types
            IType_struct const *st = cast<IType_struct>(type);
            int count = st->get_field_count();
            for (int i = 0; i < count; ++i) {
                IType const   *field_type;
                ISymbol const *field_symbol;

                st->get_field(i, field_type, field_symbol);
                collect_indexable_types(field_type, is_exported);
            }
        }
        break;
    default:
        break;
    }
}

// Collect all indexable builtin types.
void Type_collector::collect_builtin_types()
{
    IType_bool const *bool_type = m_type_factory.create_bool();
    collect_vector_types(bool_type);

    IType_int const *int_type = m_type_factory.create_int();
    collect_vector_types(int_type);

    IType_float const *float_type = m_type_factory.create_float();
    collect_vector_types(float_type);
    collect_matrix_types(float_type);

    IType_double const *double_type = m_type_factory.create_double();
    collect_vector_types(double_type);
    collect_matrix_types(double_type);
}

// Add a type to the "defined type" list.
void Type_collector::add_defined_type(IType const *tp)
{
    if (m_defined_types.insert(tp).second) {
        m_defined_type_list.push_back(tp);
    }
}

// Check if the given type is already "defined".
bool Type_collector::is_defined_type(IType const *type) const
{
    return m_defined_types.find(type) != m_defined_types.end();
}

// Check if the given type is a known array constructor type.
bool Type_collector::is_array_constructor_type(IType const *type) const
{
    return m_array_constrs.find(type) != m_array_constrs.end();
}

// Check if a given array type is known by this collector.
bool Type_collector::known_array_type(IType_array const *a_type) const
{
    if (a_type->is_immediate_sized()) {
        // there is one access for EVERY immediate sized array
        return m_indexable_types.find(a_type) != m_indexable_types.end();
    } else {
        // there is only ONE access for all deferred arrays of the
        // same element type
        return m_array_elements.find(a_type->get_element_type()) != m_array_elements.end();
    }
}

// Collect all indexable vector types of a base type.
void Type_collector::collect_vector_types(IType_atomic const *base_type)
{
    IType_vector const *vt;

    vt = m_type_factory.create_vector(base_type, 2);
    m_indexable_type_list.push_back(Indexable(vt, false, true));
    vt = m_type_factory.create_vector(base_type, 3);
    m_indexable_type_list.push_back(Indexable(vt, false, true));
    vt = m_type_factory.create_vector(base_type, 4);
    m_indexable_type_list.push_back(Indexable(vt, false, true));
}

// Collect all indexable matrix types of a base type.
void Type_collector::collect_matrix_types(IType_atomic const *base_type)
{
    for (int rows = 2; rows <= 4; ++rows) {
        IType_vector const *vt = m_type_factory.create_vector(base_type, rows);
        IType_matrix const *mt;

        mt = m_type_factory.create_matrix(vt, 2);
        m_indexable_type_list.push_back(Indexable(mt, false, true));
        mt = m_type_factory.create_matrix(vt, 3);
        m_indexable_type_list.push_back(Indexable(mt, false, true));
        mt = m_type_factory.create_matrix(vt, 4);
        m_indexable_type_list.push_back(Indexable(mt, false, true));
    }
}

} // mdl
} // mi

