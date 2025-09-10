/******************************************************************************
 * Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "generator_dag_unit.h"
#include "generator_dag_generated_dag.h"
#include "generator_dag_serializer.h"
#include "generator_dag_tools.h"

namespace mi {
namespace mdl {

// Register a new file name (for debug info) and get its ID.
size_t DAG_unit::register_file_name(
    char const *fname)
{
    m_dbg_fnames.push_back(Arena_strdup(m_arena, fname));
    return m_dbg_fnames.size();
}

// Get the file name of a debug info.
char const *DAG_unit::get_fname(size_t id) const
{
    if (id < m_dbg_fnames.size()) {
        return m_dbg_fnames[id];
    }
    return NULL;
}

// Get the file name of a debug info.
char const *DAG_unit::get_fname(DAG_DbgInfo dbg_info) const
{
    unsigned id = dbg_info.get_file_id();

    if (id == 0u) {
        if (dbg_info == DAG_DbgInfo::generated) {
            return "<generated>";
        }
        if (dbg_info == DAG_DbgInfo::builtin) {
            return "<builtin>";
        }
        MDL_ASSERT(!"unexpected special debug info");
    } else if (id <= m_dbg_fnames.size()) {
        return m_dbg_fnames[id - 1];
    }

    return NULL;
}

// Copy the file name table from another DAG_unit.
bool DAG_unit::copy_fname_table(DAG_unit const &other)
{
    if (has_dbg_info()) {
        size_t n_fnames = get_file_name_count();
        if (n_fnames == 0) {
            // fast copy
            for (size_t id = 0, n = other.get_file_name_count(); id < n; ++id) {
                char const *fname = other.get_fname(id);

                register_file_name(fname);
            }
            return true;
        } else {
            // with checking
            typedef hash_set<char const *, cstring_hash, cstring_equal_to>::Type Fname_set;

            Fname_set fname_set(0, Fname_set::hasher(), Fname_set::key_equal(), get_allocator());

            for (size_t i = 0; i < n_fnames; ++i) {
                fname_set.insert(get_fname(i));
            }

            bool res = true;
            for (size_t id = 0, n = other.get_file_name_count(); id < n; ++id) {
                char const *fname = other.get_fname(id);

                if (fname_set.find(fname) != fname_set.end()) {
                    register_file_name(fname);
                } else {
                    res = false;
                }
            }
            return res;
        }
    }
    // as no file name was imported if debug info was disabled, assume false
    return false;
}

// Get the name for a DAG node if there is any.
ISymbol const *DAG_unit::get_node_name(DAG_node const *n) const
{
    auto it = m_node_name_map.find(n);
    if (it != m_node_name_map.end()) {
        return it->second;
    }
    return nullptr;
}

// Serialize the factories of the unit. Must be called before the DAG nodes are serialized.
void DAG_unit::serialize_factories(DAG_serializer &dag_serializer) const
{
    m_sym_tab.serialize(dag_serializer);
    m_type_factory.serialize(dag_serializer);
    m_value_factory.serialize(dag_serializer);
}

// Serialize the rest of the unit.
void DAG_unit::serialize_attributes(DAG_serializer &dag_serializer) const
{
    // serialize the node names
    size_t n_names = m_node_name_map.size();
    dag_serializer.write_size_t(n_names);
    for (auto const &it : m_node_name_map) {
        Tag_t t = dag_serializer.get_ir_node_tag(it.first);
        dag_serializer.write_encoded_tag(t);

        Tag_t s = dag_serializer.get_symbol_tag(it.second);
        dag_serializer.write_encoded_tag(s);
    }

    size_t n_files = m_dbg_fnames.size();
    dag_serializer.write_size_t(n_files);

    for (size_t i = 0; i < n_files; ++i) {
        dag_serializer.write_cstring(m_dbg_fnames[i]);
    }
}

// Deserialize the factories of the unit. Must be called before the DAG nodes are deserialized.
void DAG_unit::deserialize_factories(DAG_deserializer &dag_deserializer)
{
    m_sym_tab.deserialize(dag_deserializer);
    m_type_factory.deserialize(dag_deserializer);
    m_value_factory.deserialize(dag_deserializer);
}

// Deserialize the rest of the unit.
void DAG_unit::deserialize_attributes(DAG_deserializer &dag_deserializer)
{
    // deserialize the node names
    size_t n_names = dag_deserializer.read_size_t();
    m_node_name_map.clear();
    for (size_t i = 0; i < n_names; ++i) {
        Tag_t t = dag_deserializer.read_encoded_tag();
        DAG_node const *node = dag_deserializer.get_ir_node(t);

        Tag_t s = dag_deserializer.read_encoded_tag();
        ISymbol const *sym = dag_deserializer.get_symbol(s);

        m_node_name_map[node] = sym;
    }

    size_t n_files = dag_deserializer.read_size_t();
    m_dbg_fnames.reserve(n_files);

    for (size_t i = 0; i < n_files; ++i) {
        m_dbg_fnames.push_back(Arena_strdup(m_arena, dag_deserializer.read_cstring()));
    }
}

} // mdl
} // mi
