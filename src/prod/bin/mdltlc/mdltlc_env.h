/******************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDLTLC_ENV_H
#define MDLTLC_ENV_H 1

#include <mdl/compiler/compilercore/compilercore_memory_arena.h>

#include "mdltlc_pprint.h"
#include "mdltlc_types.h"

class Environment {
public:
    enum Kind {
        ENV_BUILTIN,
        ENV_ATTRIBUTE,
        ENV_LOCAL
    };

    // List of types bound to a name.
    //
    // Type in the following is not const because type variables are
    // bound during type inference.
    typedef mi::mdl::vector<Type *>::Type Type_list;

    // Map from symbols (names) to type lists.
    typedef mi::mdl::ptr_hash_map<Symbol const, Type_list >::Type Type_map;

    /// Constructor for environment.
    ///
    /// \param arena Arena to use for allocating map entries and type
    ///              lists.
    /// \param enclosing Pointer to enclosing environment or nullptr.
    ///
    Environment(mi::mdl::Memory_arena &arena, Kind kind, Environment *enclosing)
      : m_arena(arena)
      , m_kind(kind)
      , m_type_map(arena.get_allocator())
      , m_enclosing(enclosing)
    {}

    /// Create a binding for symbol `name` to type `type`. If there
    /// already were bindings for `name`, add the type to the
    /// binding's type list.
    ///
    /// \return true if this is the first binding for `name` in the
    ///         environment, return false if there already were
    ///         bindings
    bool bind(Symbol const *name, Type *type);

    /// Look up a symbol in the environments and it's enclosing
    /// environments.
    ///
    /// \return Pointer to vector of type pointers for `name` in
    ///         either this environment or the nearest enclosing env
    ///         that has a binding for `name`. nullptr is returned if
    ///         there is no binding for `name`.
    Type_list *find(Symbol const *name, Environment **binding_env = nullptr);

    void pp(pp::Pretty_print &p);

    Kind get_kind() const { return m_kind; }
    Environment * get_enclosing() const { return m_enclosing; }

    /// Return the innermost enclosing attribute environment.
    ///
    /// \return Pointer to innermost enclosing attribute environment,
    ///         or nullptr if there is none. Can be the same as
    ///         `this`, if it is an attribute environment.
    Environment *attribute_environment();

private:
    mi::mdl::Memory_arena &m_arena;
    Kind m_kind;
    Type_map m_type_map;
    Environment *m_enclosing;
};

#endif
