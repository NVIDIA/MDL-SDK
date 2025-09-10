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

#ifndef MDL_GENERATOR_DAG_UNIT_H
#define MDL_GENERATOR_DAG_UNIT_H 1

#include <mi/mdl/mdl_generated_dag.h>

#include "mdl/compiler/compilercore/compilercore_allocator.h"
#include "mdl/compiler/compilercore/compilercore_symbols.h"
#include "mdl/compiler/compilercore/compilercore_factories.h"
#include "mdl/compiler/compilercore/compilercore_mdl.h"

namespace mi {
namespace mdl {

class DAG_serializer;
class DAG_deserializer;
class Position;

/// A DAG unit holds all additional data regarding a DAG that forms a "unit".
/// So far, this is the DAG debug info only.
class DAG_unit {
public:
    typedef unsigned File_ID;
    typedef unsigned Lineno_type;

    /// The type of maps from DAG nodes to names.
    typedef ptr_hash_map<DAG_node const, ISymbol const *>::Type Node_name_map;

    /// True, if debug info is enabled.
    bool has_dbg_info() const { return m_has_dbg_info; }

    /// Constructor.
    ///
    /// \param compiler           the MDL compiler
    /// \param enable_debug_info  if True, debug info is enabled
    DAG_unit(
        MDL  *compiler,
        bool enable_debug_info)
    : m_arena(compiler->get_allocator())
    , m_sym_tab(m_arena)
    , m_type_factory(m_arena, compiler->get_type_factory(), m_sym_tab)
    , m_value_factory(m_arena, m_type_factory)
    , m_node_name_map(compiler->get_allocator())
    , m_cse_nodes_with_different_names(false)
    , m_dbg_fnames(compiler->get_allocator())
    , m_has_dbg_info(enable_debug_info)
    {}

    /// Get the allocator.
    IAllocator *get_allocator() { return m_arena.get_allocator(); }

    /// Get the memory arena.
    Memory_arena &get_arena() { return m_arena; }

    /// Get the memory arena.
    Memory_arena const &get_arena() const { return m_arena; }

    /// Get the symbol table of this unit.
    Symbol_table &get_symbol_table() { return m_sym_tab; }

    /// Get the symbol table of this unit.
    Symbol_table const &get_symbol_table() const { return m_sym_tab; }

    /// Get the type factory of this unit.
    Type_factory &get_type_factory() { return m_type_factory; }

    /// Get the type factory of this unit.
    Type_factory const &get_type_factory() const { return m_type_factory; }

    /// Get the value factory of this unit.
    Value_factory &get_value_factory() { return m_value_factory; }

    /// Get the value factory of this unit.
    Value_factory const &get_value_factory() const { return m_value_factory; }

    /// Get or create a new shared Symbol for the given name.
    ///
    /// \param name  the shared symbol name to lookup
    ///
    /// \return the symbol for this  name, creates one if not exists
    ISymbol const *get_shared_symbol(char const *name) {
        return m_sym_tab.get_shared_symbol(name);
    }

    /// Import a category from another type factory.
    ///
    /// \param cat  the category to import
    MDL_CHECK_RESULT IStruct_category const *import_category(
        IStruct_category const *cat) {
        return m_type_factory.import_category(cat);
    }

    /// Import a type.
    ///
    /// \param type  the type to import
    MDL_CHECK_RESULT IType const *import(IType const *type) {
        return m_type_factory.import(type);
    }

    /// Import a value.
    ///
    /// \param value  the value to import
    MDL_CHECK_RESULT IValue const *import(IValue const *value) {
        return m_value_factory.import(value);
    }

    /// Get the equivalent type for a given type in our type factory or return NULL if
    /// type is not imported.
    ///
    /// \param type  the type to import
    ///
    /// \note Similar to import(), but does not create new types and does not support
    ///       function and abstract array types.
    IType const *get_equal(IType const *type) const { return m_type_factory.get_equal(type); }

    /// Register a new file name (for debug info) and get its ID.
    ///
    /// \param fname  the file name
    ///
    /// \note The zero value is reserved for "no debug info".
    /// \note This method does NOT check if the file name is already registered, it should
    ///       only be called from a Dag_builder.
    size_t register_file_name(char const *fname);

    /// Get the number of registered file names.
    size_t get_file_name_count() const { return m_dbg_fnames.size(); }

    /// Get the file name of a debug info for a file ID.
    ///
    /// \param id  the file ID
    char const *get_fname(size_t id) const;

    /// Get the file name of a debug info.
    ///
    /// \param dbg_info  the debug info
    char const *get_fname(DAG_DbgInfo dbg_info) const;

    /// Get the line of a debug info.
    ///
    /// \param dbg_info  the debug info
    unsigned get_line(DAG_DbgInfo dbg_info) const { return dbg_info.get_line(); }

    /// Copy the file name table from another DAG_unit.
    ///
    /// \param other  another DAG_unit
    ///
    /// \return true if no file name from other exist so far, false if at least
    ///         one file name already exists
    bool copy_fname_table(DAG_unit const &other);

    /// Check if this unit owns the given DAG node.
    MDL_CHECK_RESULT bool is_owner(DAG_node const *n) const { return m_arena.contains(n); }

    /// Import a symbol.
    ///
    /// \param sym  the symbol to import
    ///
    /// \return the imported symbol
    ISymbol const *import_symbol(ISymbol const *sym) {
        return m_sym_tab.get_symbol(sym->get_name());
    }

    /// Get the node name map.
    ///
    /// \return the node name map
    DAG_unit::Node_name_map const &get_node_name_map() const {
        return m_node_name_map;
    }

    /// Set the name for a DAG node.
    ///
    /// \param n  the DAG node
    /// \param sym  the symbol of the name
    void set_node_name(DAG_node const *n, ISymbol const *sym) {
        m_node_name_map[n] = sym;
    }

    /// Get the name for a DAG node if there is any.
    ///
    /// \param n  the DAG node
    ///
    /// \return  the symbol of the name or nullptr if there is no associated name for the node
    ISymbol const *get_node_name(DAG_node const *n) const;

    /// Get the name for a DAG node if there is any and the name should influence CSE.
    ///
    /// \param n  the DAG node
    ///
    /// \return  the symbol of the name or nullptr if there is no associated name for the node
    ///          or the name should not influence CSE
    ISymbol const *get_cse_node_name(DAG_node const *n) const {
        if (m_cse_nodes_with_different_names) {
            return get_node_name(n);
        }
        return nullptr;
    }

    /// Set whether nodes with different names should be CSE'd.
    ///
    /// \param flag  if true, CSE nodes with different names, otherwise don't
    ///
    /// \return the previous value
    bool set_cse_nodes_with_different_names(bool flag) {
        bool old = m_cse_nodes_with_different_names;
        m_cse_nodes_with_different_names = flag;
        return old;
    }

    /// Serialize the factories of the unit. Must be called before the DAG nodes are serialized.
    ///
    /// \param serializer  the DAG IR serializer
    void serialize_factories(DAG_serializer &serializer) const;

    /// Serialize the rest of the unit.
    ///
    /// \param serializer  the DAG IR serializer
    void serialize_attributes(DAG_serializer &serializer) const;

    /// Deserialize the factories of the unit. Must be called before the DAG nodes are deserialized.
    ///
    /// \param deserializer  the DAG IR deserializer
    void deserialize_factories(DAG_deserializer &deserializer);

    /// Deserialize the rest of the unit.
    ///
    /// \param deserializer  the DAG IR deserializer
    void deserialize_attributes(DAG_deserializer &deserializer);

private:
    /// The memory arena that contains all entities owned by this unit.
    Memory_arena m_arena;

    /// The symbol table of this unit
    Symbol_table m_sym_tab;

    /// The type factory of this unit.
    Type_factory m_type_factory;

    /// The value factory of this unit.
    Value_factory m_value_factory;

    /// The map for node names.
    Node_name_map m_node_name_map;

    /// If true, CSE nodes with different names.
    bool m_cse_nodes_with_different_names;

    typedef vector<char const *>::Type  Cstr_vector;

    /// List of registered file names for debug info.
    Cstr_vector m_dbg_fnames;

    /// True if debug info is enabled.
    bool m_has_dbg_info;
};

} // mdl
} // mi

#endif // MDL_GENERATOR_DAG_UNIT_H
