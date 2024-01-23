/******************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
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

#include <mdl/compiler/compilercore/compilercore_allocator.h>

#include "compiler_glsl_version.h"
#include "compiler_glsl_compilation_unit.h"
#include "compiler_glsl_definitions.h"
#include "compiler_glsl_type_cache.h"
#include "compiler_glsl_builtins.h"

namespace mi {
namespace mdl {
namespace glsl {

/// Helper class to create all builtins
class Builtins {

public:
    /// Constructor.
    ///
    /// \param unit  the compilation unit to fill up
    Builtins(
        Compilation_unit &unit,
        Type_cache       &tc)
    : m_alloc(unit.get_allocator())
    , m_ctx(unit.get_glslang_context())
    , m_symtab(unit.get_symbol_table())
    , m_tc(tc)
    , m_deftab(unit.get_definition_table())
    , m_params(m_alloc)
    , m_fields(m_alloc)

    // helper to reduce generated code
    , VERSION(m_ctx.get_version())
    , PROFILE(m_ctx.get_profile())
    , LANG(m_ctx.get_language())
    {
    }

    /// Insert entities.
    void build(Compilation_unit *unit);

private:

    /// Start a generic type.
    ///
    /// \param type   the type
    Symbol *type_begin(Type *type)
    {
        Symbol     *sym = type->get_sym();
        Definition *def = m_deftab.enter_type_definition(sym, type, /*loc=*/NULL);
        m_deftab.enter_scope(type, def);

        // build the copy constructor
        {
            Type_function::Parameter param(type, Type_function::Parameter::PM_IN);
            Type_function *f_type = m_tc.get_function(type, param);
            m_deftab.enter_function_definition(
                sym, f_type, Def_function::DS_CONV_CONSTRUCTOR, /*loc=*/NULL);
        }
        return sym;
    }

    /// Start a vector type.
    void vector_type_begin(Type_vector *v_type)
    {
        // See: 5.4.2. Vector and Matrix Constructors
        Symbol *sym    = type_begin(v_type);
        Type   *e_type = v_type->get_element_type();
        size_t n_elems = v_type->get_size();

        // build the vector splat constructor:
        // "If there is a single scalar parameter to a vector constructor, it is used to
        // initialize all components of the constructed vector to that scalar's value."
        {
            m_params.clear();

            Type_function::Parameter param(e_type, Type_function::Parameter::PM_IN);
            Type_function *f_type = m_tc.get_function(v_type, param);
            m_deftab.enter_function_definition(
                sym, f_type, Def_function::DS_VECTOR_SPLAT_CONSTRUCTOR, /*loc=*/NULL);
        }

        // build the elemental constructor
        {
            m_params.clear();

            for (size_t i = 0, n = v_type->get_size(); i < n; ++i) {
                m_params.push_back(
                    Type_function::Parameter(e_type, Type_function::Parameter::PM_IN));
            }
            Type_function *f_type = m_tc.get_function(v_type, m_params);
            m_deftab.enter_function_definition(
                sym, f_type, Def_function::DS_ELEM_CONSTRUCTOR, /*loc=*/NULL);
        }

        // construct conversion constructors:
        // "If the basic type (bool, int, float, or double) of a parameter to a constructor
        // does not match the basic type of the object being constructed, the scalar construction
        // rules(above) are used to convert the parameters."
        {
            Scope        *e_scope = m_deftab.get_type_scope(e_type);
            Symbol       *e_sym   = e_type->get_sym();
            Def_function *f_def   =
                as_or_null<Def_function>(e_scope->find_definition_in_scope(e_sym));

            for (; f_def != nullptr; f_def = as_or_null<Def_function>(f_def->get_prev_def())) {
                if (f_def->get_semantics() != Def_function::DS_CONV_CONSTRUCTOR) {
                    continue;
                }

                Type_function *func_type = f_def->get_type();
                Type_scalar   *arg_type  =
                    cast<Type_scalar>(func_type->get_parameter(0)->get_type());

                // arg_type can be converted to the element type, hence create a vector
                // conversion too
                m_params.clear();

                m_params.push_back(
                    Type_function::Parameter(
                        m_tc.get_vector(arg_type, n_elems),
                        Type_function::Parameter::PM_IN));

                Type_function *f_type = m_tc.get_function(v_type, m_params);
                m_deftab.enter_function_definition(
                    sym, f_type, Def_function::DS_CONV_CONSTRUCTOR, /*loc=*/NULL);
            }
        }
    }

    /// Start a matrix type.
    void matrix_type_begin(Type_matrix *type)
    {
        Symbol      *sym    = type_begin(type);
        Type_vector *v_type = type->get_element_type();
        size_t       n_cols = type->get_columns();
        Type        *e_type = v_type->get_element_type();
        size_t       n_rows = v_type->get_size();

        // build the matrix diagonal constructor
        {
            m_params.clear();

            Type_function::Parameter param(e_type, Type_function::Parameter::PM_IN);
            Type_function *f_type = m_tc.get_function(type, param);
            m_deftab.enter_function_definition(
                sym, f_type, Def_function::DS_MATRIX_DIAG_CONSTRUCTOR, /*loc=*/NULL);
        }

        // build the elemental constructor
        {
            m_params.clear();

            for (size_t i = 0; i < n_cols; ++i) {
                m_params.push_back(
                    Type_function::Parameter(v_type, Type_function::Parameter::PM_IN));
            }
            Type_function *f_type = m_tc.get_function(type, m_params);
            m_deftab.enter_function_definition(
                sym, f_type, Def_function::DS_ELEM_CONSTRUCTOR, /*loc=*/NULL);
        }

        // build the matrix elemental constructor
        {
            m_params.clear();

            for (size_t i = 0; i < n_cols * n_rows; ++i) {
                m_params.push_back(
                    Type_function::Parameter(e_type, Type_function::Parameter::PM_IN));
            }
            Type_function *f_type = m_tc.get_function(type, m_params);
            m_deftab.enter_function_definition(
                sym, f_type, Def_function::DS_MATRIX_ELEM_CONSTRUCTOR, /*loc=*/NULL);
        }
    }

    // declare a function for every declared block
    #define BLOCK(name, pred) void block_##name();
    #include "compiler_glsl_runtime.h"

private:
    /// The allocation.
    IAllocator *m_alloc;

    /// The GLSLang version to build for.
    GLSLang_context const &m_ctx;

    /// The symbol table of the unit.
    Symbol_table &m_symtab;

    /// The type cache.
    Type_cache &m_tc;

    /// The definition table.
    Definition_table &m_deftab;

    /// Helper for creating parameters
    vector<Type_function::Parameter>::Type m_params;

    /// Helper for creating struct types
    vector<Type_struct::Field>::Type       m_fields;

    // helper for version
    unsigned         VERSION;
    unsigned         PROFILE;
    unsigned         LANG;
};

// general defines
#define ES                           (PROFILE == GLSL_PROFILE_ES)
#define CORE                         (PROFILE == GLSL_PROFILE_CORE)
#define COMPATIBILITY                (PROFILE == GLSL_PROFILE_COMPATIBILITY)
#define NOPROFILE                    (VERSION == 0)
#define VERTEX_LANG                  (LANG == GLSL_LANG_VERTEX)
#define TESSCONTROL_LANG             (LANG == GLSL_LANG_TESSCONTROL)
#define GEOMETRY_LANG                (LANG == GLSL_LANG_GEOMETRY)
#define FRAGMENT_LANG                (LANG == GLSL_LANG_FRAGMENT)
#define COMPUTE_LANG                 (LANG == GLSL_LANG_COMPUTE)
#define HAS(ext)                     m_ctx.has_extension(ext)
#define HAS_64BIT_INT_TYPES          m_ctx.has_int64_types()
#define HAS_EXPLICIT_SIZED_INT_TYPES m_ctx.has_explicit_sized_int_types()

// Insert entities.
void Builtins::build(Compilation_unit *unit)
{
#define BLOCK(name, pred) if (pred) { block_##name(); }

#include "compiler_glsl_runtime.h"
}

// Finally build all the block builder
#define BLOCK(name, pred) void Builtins::block_##name() {
#define BEND(name)        }

#define ARG0()
#define ARG1(a1)               a1
#define ARG2(a1, a2)           a1 a2
#define ARG3(a1, a2, a3)       a1 a2 a3
#define ARG4(a1, a2, a3, a4)   a1 a2 a3 a4

// an argument
#define ARGUMENT(T, N, MOD) \
    m_params.push_back(Type_function::Parameter(m_tc. T##_type, MOD));

// an IN argument
#define ARG(T, N)    ARGUMENT(T, N, Type_function::Parameter::PM_IN)

// an OUT argument
#define OUTARG(T, N) ARGUMENT(T, N, Type_function::Parameter::PM_OUT)

#define EXTENSION(ext) if (m_ctx.has_extension(ext)) {
#define EEND(ext)                                    }

#define HAS(ext) m_ctx.has_extension(ext)

// a function
#define FUNCTION(ret, name, args) \
    m_params.clear(); \
    args \
    { \
        Type_function *f_type = m_tc.get_function(m_tc. ret##_type, m_params); \
        Symbol        *f_sym  = m_symtab.get_symbol(#name); \
        m_deftab.enter_function_definition(f_sym, f_type, Def_function::DS_RT_##name, NULL); \
    }

// EXPLICIT constructor behavior
#undef EXPLICIT
#define EXPLICIT (void)def

// IMPLICIT constructor behavior
#undef IMPLICIT
#define IMPLICIT (void)def

#define CONSTRUCTOR(kind, name, args, sema, pred) \
    if (pred) { \
        m_params.clear(); \
        args \
        { \
            Type_function *f_type = m_tc.get_function(m_tc. name##_type, m_params); \
            Symbol        *f_sym  = m_symtab.get_symbol(#name); \
            Def_function  *def    = \
                m_deftab.enter_function_definition(f_sym, f_type, Def_function:: sema, NULL); \
            kind; \
        } \
    }

// a struct
#define STRUCT_BEGIN(name) \
    m_fields.clear();

#define STRUCT_END(name) \
    { \
        Symbol      *sym    = m_symtab.get_symbol(#name); \
        Type_struct *s_type = m_tc.get_struct(m_fields, sym); \
        m_deftab.enter_type_definition(sym, s_type, NULL); \
    }

#define FIELD(M, T, N) \
    { \
        Type   *f_type = m_tc. M##T##_type; \
        Symbol *f_sym  = m_symtab.get_symbol(#N); \
        m_fields.push_back(Type_struct::Field(f_type, f_sym)); \
    }

#define UVARIABLE(T, N) \
{ \
    Symbol     *t_sym  = m_symtab.get_symbol(#T); \
    Definition *def    = m_deftab.get_definition(t_sym); \
    Type       *s_type = def->get_type(); \
    Symbol     *v_sym  = m_symtab.get_symbol(#N); \
    Type       *v_type = m_tc.get_alias(s_type, Type::MK_UNIFORM); \
    m_deftab.enter_variable_definition(v_sym, v_type, NULL); \
}

// a type (including the copy constructor)
#define TYPE_BEGIN(T)  type_begin(m_tc. T##_type);
#define TYPE_END       m_deftab.leave_scope();

// a vector type (including the copy and elemental constructors)
#define VECTOR_TYPE_BEGIN(T)  vector_type_begin(m_tc. T##_type);
#define VECTOR_TYPE_END       m_deftab.leave_scope();

// a matrix type (including the copy, elemental, and matrix elemental constructors)
#define MATRIX_TYPE_BEGIN(T)  matrix_type_begin(m_tc. T##_type);
#define MATRIX_TYPE_END       m_deftab.leave_scope();

// a type alias
#define TYPE_ALIAS(A, T)  GLSL_ASSERT(m_tc. A##_type == m_tc. T##_type && "wrong alias type");

#include "compiler_glsl_runtime.h"

// Create GLSL builtin entities and put them into the predefined scope.
void enter_predefined_entities(
    Compilation_unit &unit,
    Type_cache       &tc)
{
    Builtins buitins(unit, tc);

    buitins.build(&unit);
}

}  // glsl
}  // mdl
}  // mi
