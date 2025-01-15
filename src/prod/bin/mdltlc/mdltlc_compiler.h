/******************************************************************************
 * Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDLTLC_COMPILER_H
#define MDLTLC_COMPILER_H 1

#include <mi/mdl/mdl_mdl.h>

#include <mdl/compiler/compilercore/compilercore_memory_arena.h>
#include <mdl/compiler/compilercore/compilercore_type_cache.h>
#include <mdl/compiler/compilercore/compilercore_factories.h>
#include <mi/mdl/mdl_mdl.h>

#include "mdltlc_message.h"
#include "mdltlc_symbols.h"
#include "mdltlc_types.h"
#include "mdltlc_compiler_options.h"
#include "mdltlc_compilation_unit.h"

/// Subclass of mi::mdl::IMDL_search_path to enable loading of MDL
/// modules.
class Mdltlc_search_path : public mi::mdl::Allocator_interface_implement<mi::mdl::IMDL_search_path>
{
    typedef mi::mdl::Allocator_interface_implement<IMDL_search_path> Base;
    friend class mi::mdl::Allocator_builder;

  public:
    /// Return the number of configured MDL search paths if set ==
    /// MDL_SEARCH_PATH, 0 otherwise.
    virtual size_t get_search_path_count(Path_set set) const {
        if (set == MDL_SEARCH_PATH)
            return m_options->get_mdl_path_count();
        return 0;
    }

    /// Return the search path at index `i`.
    virtual char const *get_search_path(Path_set set, size_t i) const {
        return m_options->get_mdl_path(i);
    }

  private:

    Mdltlc_search_path(mi::mdl::IAllocator *alloc, Compiler_options const *options)
        : Base(alloc)
        , m_options(options) {}

    Compiler_options const *m_options;
};

/// Interface of the mdltl compilation unit.
class ICompiler : public
    mi::base::Interface_declare<0x61017a2e,0xe877,0x40af,0x87,0x06,0x0c,0x2f,0x23,0xfe,0x03,0x88,
    mi::base::IInterface>
{
public:
};

/// Implementation of an mdltl compilation unit.
class Compiler : public mi::mdl::Allocator_interface_implement<ICompiler>
{
    typedef mi::mdl::Allocator_interface_implement<ICompiler> Base;
    friend class mi::mdl::Allocator_builder;
public:
    // --------------------------- non interface methods ---------------------------

    /// Get a (writable) reference to the current compiler options.
    Compiler_options &get_compiler_options();

    /// Get a read-only reference to the current compiler options.
    Compiler_options const &get_compiler_options() const;

    /// Compile all the configured mdltl files according to the
    /// compilation options in effect.
    ///
    /// \param err_count  This out parameter is incremented by the number
    ///                   of errors encountered.
    void run(unsigned& err_count);

    /// Return a reference to the message list.
    Message_list const &get_messages() const;

private:
    /// Constructor.
    ///
    /// \param alloc        the allocator
    /// \param file_name    the file name of the module
    explicit Compiler(
        mi::mdl::IMDL *imdl);

    /// Add a node (loaded from the standard library) to the internal
    /// map that is used in type checking and code generation.
    ///
    /// \param symbol    symbol (used for lookup) of the node
    /// \param fq_symbol fully qualified version of `symbol` (can be the same)
    /// \param mdl_type  type of the symbol in the MDL type system
    /// \oaram sema      MDL semantics of the symbol
    void add_builtin(Symbol *symbol,
                     Symbol *fq_symbol,
                     mi::mdl::IType const *mdl_type,
                     mi::mdl::IDefinition::Semantics sema);

    /// Load the builting definitions from a builtin MDL module.
    ///
    /// \param name    fully qualified identifier of the builtin module to load.
    void load_builtin_module(const char *name);

    /// Load the builtin definitions from all supported builtin MDL modules.
    void load_builtins();

    /// Create a new compilation unit for the given file name.
    mi::base::Handle<Compilation_unit> create_unit(const char *fname);

    /// Print all messages that have been generated during compilation
    /// to stderr.
    void print_messages();

private:
    // non copyable
    Compiler(Compiler const &) = delete;
    Compiler &operator=(Compiler const &) = delete;

private:
    /// The MDL implementation to use for access to intrinsics.
    mi::mdl::IMDL *m_imdl;

    /// Downcasted and referenced version of the MDL interface above.
    mi::mdl::MDL &m_mdl;

    mi::mdl::Node_types m_node_types;

    /// The type factory of the MDL instance.
    mi::mdl::Type_factory &m_mdl_type_factory;

    /// The symbol table of the MDL instance.
    mi::mdl::Symbol_table &m_mdl_symbol_table;

    /// The allocator to be used for all allocations during the
    /// compiler run.
    mi::mdl::IAllocator *m_allocator;

    /// The memory arena of this module, use to allocate all elements
    /// of this module.
    mi::mdl::Memory_arena m_arena;

    /// Arena builder for allocating objects on the memory arena
    /// m_arena.
    mi::mdl::Arena_builder m_arena_builder;

    /// Compilation options in effect for this compiler instance.
    Compiler_options m_comp_options;

    /// Map from names to builtin BSDF nodes.
    Builtin_type_map m_builtin_type_map;

    /// The symbol table of this module.
    Symbol_table m_symbol_table;

    /// Compiler message collector.
    Message_list m_messages;

    /// Type cache for accessing MDL core types.
    mi::mdl::Type_cache m_mdl_type_cache;
};

#endif
