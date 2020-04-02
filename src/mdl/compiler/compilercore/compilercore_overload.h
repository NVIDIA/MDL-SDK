/******************************************************************************
 * Copyright (c) 2014-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILERCORE_OVERLOAD_H
#define MDL_COMPILERCORE_OVERLOAD_H 1

#include <mi/mdl/mdl_types.h>
#include "compilercore_memory_arena.h"

namespace mi {
namespace mdl {

class Definition;
class Definition_table;
class IExpression;
class Module;
class Type_factory;

///
/// A helper class for overload resolution.
///
class Overload_solver {
    /// A signature entry for function overload resolution.
    struct Signature_entry {
        IType const *const *signature;
        bool const *const  bounds;
        size_t             sig_length;
        Definition const   *def;

        Signature_entry(
            IType const *const *signature,
            bool const *const  bounds,
            size_t             sig_length,
            Definition const *def)
        : signature(signature), bounds(bounds), sig_length(sig_length), def(def) {}
    };

    typedef Arena_list<Signature_entry>::Type Signature_list;

public:
    typedef list<Definition const *>::Type Definition_list;

public:
    /// Constructor.
    ///
    /// \param module  the module inside overload are solved
    explicit Overload_solver(Module &module);

    /// Resolve overloads for a positional call.
    ///
    /// \param def             the overload definition set
    /// \param arg_types       positional argument types
    /// \param num_args        number of positional arguments
    ///
    /// \note Note that the result list is in "reverse" order compared to the declaration
    ///       order of overloads inside an MDL source file.
    /// \note Note also that this version can handle function calls and annotations.
    Definition_list find_positional_overload(
        Definition const *def,
        IType const      *arg_types[],
        size_t           num_args);

private:
    /// Retrieve the used allocator.
    IAllocator *get_allocator() const;

    /// Checks if types are equal.
    static bool equal_types(IType const *a, IType const *b);

    /// Compare two signature entries representing functions for "specific-ness".
    ///
    /// \param a  first function entry
    /// \param b  second function entry
    ///
    /// \return true if a has some is more specific parameters than b
    bool is_more_specific(Signature_entry const &a, Signature_entry const &b);

    /// Given a list and a (call) signature, kill any less specific definition from the list.
    /// If new_sig is less specific then an entry in the list return false
    /// else true.
    ///
    /// \param list     the candidate list
    /// \param new_sig  a new candidate
    ///
    /// \return false : the candidate list contains an already more specific version,
    ///                 drop the new candidate
    ///         true  : add the new candidate
    bool kill_less_specific(Signature_list &list, Signature_entry const &new_sig);

    /// Check if a parameter type is already bound.
    ///
    /// \param abs_type  a deferred array size type
    bool is_bound_type(IType_array const *abs_type) const;

    /// Bind the given deferred sized array type to another array type.
    ///
    /// \param abs_type  a deferred array size type
    /// \param type      the immediate or deferred sized array type that is bound
    void bind_array_type(IType_array const *abs_type, IType_array const *type);

    /// Return the bound type for a deferred type.
    ///
    /// \param type  a type
    ///
    /// \return the bound array type or type itself it if was not bound
    IType const *get_bound_type(IType const *type);

    /// Clear all bindings of deferred sized array types.
    void clear_type_bindings();

    /// Check if it is possible to assign an argument type to the parameter
    /// type of a call.
    ///
    /// \param param_type  the type of a function parameter
    /// \param src_type    the type of the corresponding call argument
    /// \param new_bound   set to true if a new type bound was executed
    bool can_assign_param(
        IType const *param_type,
        IType const *arg_type,
        bool        &new_bound);

    /// Find the Definition of an implicit conversion constructor or an conversion
    /// operator.
    ///
    /// \param from_tp  the source type of the conversion
    /// \param to_tp    the destination type of the conversion
    ///
    /// \return the Definition of the constructor/conversion operator or NULL
    ///         if an implicit conversion is not possible.
    Definition *find_implicit_conversion(
        IType const    *from_tp,
        IType const    *to_tp);

    /// Returns a short signature from a (function) definition.
    ///
    /// \param def  the function (or constructor) definition
    string get_short_signature(Definition const *def) const;

    /// Get the default expression of a parameter of a function, constructor or annotation.
    ///
    /// \param def        the entity definition
    /// \param param_idx  the index of the parameter
    ///
    /// \note In contrast to Definition::get_default_param_initializer() this version
    ///       retrieves the parameter initializer from imported entities
    IExpression const *get_default_param_initializer(
        Definition const *def,
        int              param_idx) const;

private:
    /// The module.
    Module &m_module;

    /// The type factory of the module
    Type_factory &m_tf;

    /// The definition table of the module.
    Definition_table &m_def_tab;

    typedef ptr_hash_map<IType const, IType const *>::Type                       Bind_type_map;
    typedef ptr_hash_map<IType_array_size const, int>::Type                      Bind_size_map;
    typedef ptr_hash_map<IType_array_size const, IType_array_size const *>::Type Bind_symbol_map;

    /// Type bindings for overload resolution.
    Bind_type_map   m_type_bindings;
    Bind_size_map   m_size_bindings;
    Bind_symbol_map m_sym_bindings;
};

}  // mdl
}  // mi

#endif
