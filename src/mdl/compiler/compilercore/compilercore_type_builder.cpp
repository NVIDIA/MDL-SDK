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
 ******************************************************************************/

#include "pch.h"

#include "compilercore_builder.h"
#include "compilercore_factories.h"
#include "compilercore_symbols.h"
#include "compilercore_tools.h"
#include "compilercore_type_cache.h"

namespace mi {
namespace mdl {

/// Helper class to create build-i types for the compiler owned factory.
class Type_builder {
public:
    /// Build all builtin types.
    void build_types()
    {
        // Build categories first, structs depend on them.
        build_category_material_category();

#define BUILTIN_STRUCT_BEGIN(structname, flags)     build_struct_##structname();
#define BUILTIN_ENUM_BEGIN(enumname, flags)         build_enum_##enumname();

#include "compilercore_known_defs.h"
    }

public:
    /// Constructor.
    Type_builder(
        Type_factory &factory,
        bool         extra_types_are_uniform)
    : mod_extra(extra_types_are_uniform ? mod_uniform : IType::MK_NONE)
    , m_tf(factory)
    , m_sym_tab(*factory.get_symbol_table())
    , m_tc(factory)
    {

    }

private:
    /// Get a predefined Symbol for the given name.
    ISymbol const *get_predef_symbol(char const *name)
    {
        ISymbol const *sym = m_sym_tab.get_symbol(name);

        MDL_ASSERT(sym->get_id() < ISymbol::SYM_SHARED_NAME &&
            "symbols used in predefined types must be also predefined");
        return sym;
    }

// Name of the built-in ::tex::gamma_mode predefined symbol
#define ENUM_tex_gamma_mode "::tex::gamma_mode"

// Name of the built-in intensity_mode predefined symbol
#define ENUM_intensity_mode "intensity_mode"

// declare the begin of a builtin struct type.
#define BUILTIN_STRUCT_BEGIN(structname, flags)     \
    void build_struct_##structname() {              \
        IType_struct::Field fields[] = {

// declare a struct field
#define STRUCT_FIELD(classname, mod, ftype, fieldname, flags) \
    IType_struct::Field(                                      \
        m_tc.decorate_type(m_tc.ftype##_type, mod_##mod),     \
        get_predef_symbol(#fieldname)                         \
    ),

// declare the end of a builtin struct type.
#define BUILTIN_STRUCT_END(structname)                       \
        };                                                   \
                                                             \
        *const_cast<IType_struct const **>(&m_tc.structname##_type) = m_tf.insert_predef_struct( \
            get_predef_symbol(#structname),                  \
            fields,                                          \
            dimension_of(fields));                           \
    }

// declare the begin of a builtin enum type.
#define BUILTIN_ENUM_BEGIN(enumname, flags) \
    void build_enum_##enumname() {          \
        IType_enum::Value values[] = {

// declare a enum value
#define ENUM_VALUE(enumname, name, value, flags)             \
    IType_enum::Value(                                       \
        get_predef_symbol(#name),                            \
        value                                                \
    ),

// declare the end of a builtin struct type.
#define BUILTIN_ENUM_END(enumname)                           \
        };                                                   \
                                                             \
        *const_cast<IType_enum const **>(&m_tc.enumname##_type) = m_tf.insert_predef_enum( \
            get_predef_symbol(ENUM_##enumname),              \
            values,                                          \
            dimension_of(values));                           \
    }

#include "compilercore_known_defs.h"

    void build_category_material_category() {
        *const_cast<IStruct_category const **>(&m_tc.material_category) =
            m_tf.insert_predef_category(get_predef_symbol("material_category"));
    }
private:
    static IType::Modifier const  mod_            = IType::MK_NONE;
    static IType::Modifier const  mod_uniform     = IType::MK_UNIFORM;
    static IType::Modifier const  mod_const       = IType::MK_CONST;

    /// The meaning of the eXtra modifier.
    IType::Modifier const mod_extra;

    /// The factory we are filing up.
    Type_factory &m_tf;

    /// The symbol table of the type factory.
    Symbol_table &m_sym_tab;

    /// A local type cache.
    Type_cache m_tc;
};

// Enter all predefined types into the given type factory.
void enter_predefined_types(
    Type_factory &tf,
    bool         extra_types_are_uniform)
{
    Type_builder builder(tf, extra_types_are_uniform);

    builder.build_types();
}

} // mdl
} // mi
