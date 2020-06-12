/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <base/system/stlext/i_stlext_no_unused_variable_warning.h>

#include <mi/mdl/mdl_definitions.h>

#include "compilercore_symbols.h"
#include "compilercore_def_table.h"
#include "compilercore_allocator.h"
#include "compilercore_factories.h"
#include "compilercore_mdl.h"
#include "compilercore_type_cache.h"
#include "compilercore_builder.h"
#include "compilercore_tools.h"
#include "compilercore_assert.h"

namespace mi {
namespace mdl {

#define ARG0()                              num_args = 0;
#define ARG1(a1)                            num_args = 1; a1
#define ARG2(a1, a2)                        num_args = 2; a1 a2
#define ARG3(a1, a2, a3)                    num_args = 3; a1 a2 a3
#define ARG4(a1, a2, a3, a4)                num_args = 4; a1 a2 a3 a4
#define ARG5(a1, a2, a3, a4, a5)            num_args = 5; a1 a2 a3 a4 a5
#define ARG6(a1, a2, a3, a4, a5, a6)        num_args = 6; a1 a2 a3 a4 a5 a6
#define ARG7(a1, a2, a3, a4, a5, a6, a7)    num_args = 7; a1 a2 a3 a4 a5 a6 a7
#define ARG8(a1, a2, a3, a4, a5, a6, a7, a8) \
    num_args = 8; a1 a2 a3 a4 a5 a6 a7 a8
#define ARG9(a1, a2, a3, a4, a5, a6, a7, a8, a9) \
    num_args = 9; a1 a2 a3 a4 a5 a6 a7 a8 a9
#define ARG12(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12) \
    num_args = 12; a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12
#define ARG16(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16) \
    num_args = 16; a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16

// create an array type defining a new size
//
// Note, that get_array_size() expects a full qualified symbol as the first
// parameter, but we are creating entity in the predefined scope that has no
// addressable name... To solve this, we use the NON-qualified symbol here.
// This should be sufficient, as it will be hopefully always different
// from all other fully qualified names.
#define ARRDEF(name) \
    N_sym = get_predef_symbol(#name); \
    IType_array_size const *name##_size = m_tc.get_array_size(/*fq*/N_sym, N_sym); \
    ptype = m_tc.create_array(ptype, name##_size);

// create an array type using a size
#define ARRUSE(name) \
    ptype = m_tc.create_array(ptype, name##_size);

// declare a parameter for every argument in the specification
#define ARG(type, name, arr)                \
    ++arg_num;                              \
    ptype = m_tc.type##_type;               \
    arr                                     \
    psym = get_predef_symbol(#name);        \
    m_params[arg_num].p_type = ptype;       \
    m_params[arg_num].p_sym  = psym;        \
    m_inits[arg_num]         = NULL;

// declare an uniform parameter for every argument in the specification
#define UARG(type, name, arr)                \
    ++arg_num;                              \
    ptype = m_tc.decorate_type(m_tc.type##_type, IType::MK_UNIFORM);    \
    arr                                     \
    psym = get_predef_symbol(#name);        \
    m_params[arg_num].p_type = ptype;       \
    m_params[arg_num].p_sym  = psym;        \
    m_inits[arg_num]         = NULL;

// declare a parameter for every argument in the specification with an default argument
#define DEFARG(type, name, arr, expr)       \
    ARG(type, name, arr)                    \
    expr

// declare an uniform  parameter for every argument in the specification with an default argument
#define UDEFARG(type, name, arr, expr)      \
    UARG(type, name, arr)                   \
    expr

// a literal expression
#define EXPR_LITERAL(value)                             \
    has_initializers = true;                            \
    m_inits[arg_num] = build_literal(value);

// a color literal expression
#define EXPR_COLOR_LITERAL(value)                       \
    has_initializers = true;                            \
    m_inits[arg_num] = build_rgb_literal(value);

// a float3 literal expression
#define EXPR_FLOAT3_LITERAL(value)                      \
    has_initializers = true;                            \
    m_inits[arg_num] = build_float3_literal(value);

// a constructor call
#define EXPR_CONSTRUCTOR(type)                          \
    has_initializers = true;                            \
    m_inits[arg_num] = build_constructor(get_predef_symbol(#type), m_tc.type##_type);

// a state function call
#define EXPR_STATE(type, name)                          \
    has_initializers = true;                            \
    m_inits[arg_num] = build_state_func_call(get_predef_symbol(#name), m_tc.type##_type);

// a tex enum value
#define EXPR_TEX_ENUM(name)                             \
    has_initializers = true;                            \
    m_inits[arg_num] = build_tex_enum(m_sym_tab.get_symbol(#name));

// an intensity_mode enum value
#define EXPR_INTENSITY_MODE_ENUM(name)                  \
    has_initializers = true;                            \
    m_inits[arg_num] = build_intensity_mode_enum(m_sym_tab.get_symbol(#name));

// retrieve the version from flags
#define VERSION(flags)          (flags)

// declare a new function/method
#define _FUNCTION(kind, ret, name, args, flags)                 \
    sym = get_predef_symbol(#name);                             \
    arg_num = -1;                                               \
    args                                                        \
    func_type = m_tc.create_function(m_tc.ret##_type,           \
        Type_cache::Function_parameters(m_params, num_args));   \
    def  = m_def_tab.enter_definition(kind, sym, func_type, NULL); \
    if (has_initializers) set_initializers(def, num_args);      \
    def->set_flag(Definition::DEF_IS_PREDEFINED);               \
    def->set_version_flags(VERSION(flags));


/// Version flags of predefined entities.
enum Version_flags {
    SINCE_1_1   = IMDL::MDL_VERSION_1_1,
    REMOVED_1_1 = (IMDL::MDL_VERSION_1_1 << 8),
    SINCE_1_2   = IMDL::MDL_VERSION_1_2,
    REMOVED_1_2 = (IMDL::MDL_VERSION_1_2 << 8),
    SINCE_1_3   = IMDL::MDL_VERSION_1_3,
    REMOVED_1_3 = (IMDL::MDL_VERSION_1_3 << 8),
    SINCE_1_4   = IMDL::MDL_VERSION_1_4,
    REMOVED_1_4 = (IMDL::MDL_VERSION_1_4 << 8),
    SINCE_1_5   = IMDL::MDL_VERSION_1_5,
    REMOVED_1_5 = (IMDL::MDL_VERSION_1_5 << 8),
    SINCE_1_6   = IMDL::MDL_VERSION_1_6,
    REMOVED_1_6 = (IMDL::MDL_VERSION_1_6 << 8),
    SINCE_1_7   = IMDL::MDL_VERSION_1_7,
    REMOVED_1_7 = (IMDL::MDL_VERSION_1_7 << 8),
};

// compilercore_known_defs.h is too big to be compiled in one function, use this class to split
// things up.
class Entity_builder {

public:
    /// Constructor.
    ///
    /// \param module                  the module that gets the builtin entities
    /// \param tc                      the type cache
    /// \param build_predefined_types  if true, fields of predefined types will be entered
    Entity_builder(Module &module, Type_cache &tc, bool build_predefined_types)
    : m_module(module)
    , m_def_tab(module.get_definition_table())
    , m_sym_tab(module.get_symbol_table())
    , m_name_fact(*m_module.get_name_factory())
    , m_value_fact(*m_module.get_value_factory())
    , m_expr_fact(*m_module.get_expression_factory())
    , m_tc(tc)
    , m_state_sym(m_sym_tab.get_predefined_symbol(ISymbol::SYM_CNST_STATE))
    , m_tex_sym(m_sym_tab.get_predefined_symbol(ISymbol::SYM_CNST_TEX))
    , m_mod_version(module.get_version())
    , m_build_predefined_types(build_predefined_types)
    {
    }

private:
    static IType::Modifier const  mod_            = IType::MK_NONE;
    static IType::Modifier const  mod_uniform     = IType::MK_UNIFORM;
    static IType::Modifier const  mod_const       = IType::MK_CONST;

    Module             &m_module;
    Definition_table   &m_def_tab;
    Symbol_table       &m_sym_tab;
    Name_factory       &m_name_fact;
    Value_factory      &m_value_fact;
    Expression_factory &m_expr_fact;

    Type_cache         &m_tc;

    ISymbol const      *m_state_sym;
    ISymbol const      *m_tex_sym;

    /// The version of the current module.
    unsigned m_mod_version;

    /// if true, fields of predefined type are entered
    bool m_build_predefined_types;

    /// temporary space for building function parameters
    IType_factory::Function_parameter m_params[16];  // -V730_NOINIT

    // temporary space for initializers
    IExpression const *m_inits[16];  // -V730_NOINIT

private:
    /// Check if the current entity represented by its version flags is available
    /// in the current module.
    bool available(unsigned flags) {
        return is_available_in_mdl(m_mod_version, flags);
    }

    /// Get a predefined Symbol for the given name.
    ISymbol const *get_predef_symbol(const char *name) {
        ISymbol const *sym = m_sym_tab.get_symbol(name);

        MDL_ASSERT(sym->get_id() < ISymbol::SYM_SHARED_NAME &&
            "symbols used in predefined types must be also predefined");
        return sym;
    }

    /// Set the initializer expressions.
    void set_initializers(Definition *def, size_t num_args) {
        m_module.allocate_initializers(def, num_args);
        for (size_t i = 0; i < num_args; ++i) {
            def->set_default_param_initializer(i, m_inits[i]);
        }
    }

    /// Creates a RGB color literal expression.
    ///
    /// \param value  the literal value
    IExpression const *build_rgb_literal(float value) {
        IValue_float const     *v = m_value_fact.create_float(value);
        IValue_rgb_color const *c = m_value_fact.create_rgb_color(v, v, v);
        return m_expr_fact.create_literal(c);
    }

    /// Creates a float3 literal expression.
    ///
    /// \param value  the literal value
    IExpression const *build_float3_literal(float value) {
        IValue_float const *v = m_value_fact.create_float(value);
        IValue const *values[] = { v, v, v };
        IValue_vector const *vec = m_value_fact.create_vector(m_tc.float3_type, values, 3);
        return m_expr_fact.create_literal(vec);
    }

    /// Creates a bool literal expression.
    ///
    /// \param value  the literal value
    IExpression const *build_literal(bool value) {
        IValue_bool const *v = m_value_fact.create_bool(value);
        return m_expr_fact.create_literal(v);
    }

    /// Creates an int literal expression.
    ///
    /// \param value  the literal value
    IExpression const *build_literal(int value) {
        IValue_int const *v = m_value_fact.create_int(value);
        return m_expr_fact.create_literal(v);
    }

    /// Creates a float literal expression.
    ///
    /// \param value  the literal value
    IExpression const *build_literal(float value) {
        IValue_float const *v = m_value_fact.create_float(value);
        return m_expr_fact.create_literal(v);
    }

    /// Creates a double literal expression.
    ///
    /// \param value  the literal value
    IExpression const *build_literal(double value) {
        IValue_double const *v = m_value_fact.create_double(value);
        return m_expr_fact.create_literal(v);
    }

    /// Creates a string literal expression.
    ///
    /// \param value  the literal value
    IExpression const *build_literal(char const *value) {
        IValue_string const *v = m_value_fact.create_string(value);
        return m_expr_fact.create_literal(v);
    }

    /// Creates a default constructor call expression.
    ///
    /// \param sym   the name of the constructor
    /// \param type  the (return) type of the constructor
    IExpression const *build_constructor(ISymbol const *sym, IType const *type) {
        IQualified_name *qname = m_name_fact.create_qualified_name();
        qname->set_absolute();
        ISimple_name const *sname = m_name_fact.create_simple_name(sym);
        qname->add_component(sname);
        IType_name *tname = m_name_fact.create_type_name(qname);
        IExpression_reference *ref = m_expr_fact.create_reference(tname);
        IExpression_call *res = m_expr_fact.create_call(ref);

        // find the definition
        Scope *scope = m_def_tab.get_type_scope(type);
        MDL_ASSERT(scope);

        Definition *def = scope->find_definition_in_scope(sym);
        MDL_ASSERT(def && def->get_kind() == Definition::DK_CONSTRUCTOR);

        for (; def != NULL; def = def->get_prev_def()) {
            IType_function const *ftype = as<IType_function>(def->get_type());

            int i = ftype->get_parameter_count() - 1;
            for (; i >= 0; --i) {
                if (def->get_default_param_initializer(i) == NULL) {
                    // found an argument that must be given
                    break;
                }
            }
            if (i < 0) {
                // all arguments are optional, found the default constructor
                break;
            }
        }
        MDL_ASSERT(def && "did not find default constructor");
        ref->set_definition(def);
        res->set_type(type);
        return res;
    }

    /// Creates a state function call expression.
    ///
    /// \param sym   the name of the state function
    /// \param type  the return type of the state function
    IExpression const *build_state_func_call(ISymbol const *sym, IType const *type)
    {
        IQualified_name *qname = m_name_fact.create_qualified_name();
        qname->set_absolute();
        qname->add_component(m_name_fact.create_simple_name(m_state_sym));
        qname->add_component(m_name_fact.create_simple_name(sym));
        IType_name *tname = m_name_fact.create_type_name(qname);
        IExpression_reference *ref = m_expr_fact.create_reference(tname);
        IExpression_call *res = m_expr_fact.create_call(ref);

        // find the definition
        Definition const *def = m_module.find_stdlib_symbol("::state", sym);

        if (def != NULL) {
            // inside the state module itself, the def can still be NULL
            qname->set_definition(def);
            ref->set_definition(def);
        }
        res->set_type(type);
        return res;
    }

    /// Creates a reference to an enum exported from ::tex.
    ///
    /// \param sym   the name of enum value
    IExpression const *build_tex_enum(ISymbol const *sym)
    {
        IQualified_name *qname = m_name_fact.create_qualified_name();
        qname->set_absolute();
        qname->add_component(m_name_fact.create_simple_name(m_tex_sym));
        qname->add_component(m_name_fact.create_simple_name(sym));
        IType_name *tname = m_name_fact.create_type_name(qname);
        IExpression_reference *ref = m_expr_fact.create_reference(tname);

        // find the definition
        Definition const *def = m_module.find_stdlib_symbol("::tex", sym);

        if (def != NULL) {
            // inside the tex module itself, the def can still be NULL
            qname->set_definition(def);
            ref->set_definition(def);
        } else {
            if (m_module.is_stdlib() && strcmp(m_module.get_name(), "::state") == 0) {
                // UGLY: inside state, tex is NOT available, hence we cannot have this
                // default there, so we leave it WITHOUT a default. Should be no problem
                // until someone want to add an instantiation of a texture to ::state.
                return NULL;
            }
        }
        MDL_ASSERT(def != NULL || strcmp(m_module.get_name(), "::tex") == 0);
        return ref;
    }

    /// Creates a reference to an intensity_mode.
    ///
    /// \param sym   the name of enum value
    IExpression const *build_intensity_mode_enum(ISymbol const *sym)
    {
        IQualified_name *qname = m_name_fact.create_qualified_name();
        qname->add_component(m_name_fact.create_simple_name(sym));
        IType_name *tname = m_name_fact.create_type_name(qname);
        IExpression_reference *ref = m_expr_fact.create_reference(tname);

        // find the definition
        Definition const *def = m_def_tab.get_definition(sym);

        MDL_ASSERT(def != NULL);
        qname->set_definition(def);
        ref->set_definition(def);
        return ref;
    }

// declare the begin of a builtin type.
#define BUILTIN_TYPE_BEGIN(classname, flags)                \
    void build_type_##classname() {                         \
        bool                 has_initializers = false;      \
        IType const          *this_type, *type, *ptype;     \
        ISymbol const        *sym, *ssym, *psym, *N_sym;    \
        Definition           *def, *odef;                   \
        IType_function const *func_type;                    \
        int                  num_args = 0, arg_num;         \
        int                  num_fields = 0;                \
        unsigned             evalue;                        \
        Scope                *file_scope, *type_scope;      \
        int                  idx = 0;                       \
                                                            \
        MI::STLEXT::no_unused_variable_warning_please(has_initializers);    \
        MI::STLEXT::no_unused_variable_warning_please(type);    \
        MI::STLEXT::no_unused_variable_warning_please(ptype);   \
        MI::STLEXT::no_unused_variable_warning_please(sym);     \
        MI::STLEXT::no_unused_variable_warning_please(ssym);    \
        MI::STLEXT::no_unused_variable_warning_please(psym);    \
        MI::STLEXT::no_unused_variable_warning_please(N_sym);    \
        MI::STLEXT::no_unused_variable_warning_please(def);     \
        MI::STLEXT::no_unused_variable_warning_please(odef);    \
        MI::STLEXT::no_unused_variable_warning_please(func_type); \
        MI::STLEXT::no_unused_variable_warning_please(num_args); \
        MI::STLEXT::no_unused_variable_warning_please(arg_num); \
        MI::STLEXT::no_unused_variable_warning_please(num_fields); \
        MI::STLEXT::no_unused_variable_warning_please(evalue);  \
        MI::STLEXT::no_unused_variable_warning_please(idx); \
        MI::STLEXT::no_unused_variable_warning_please(file_scope); \
                                                            \
        if (!available(flags)) return;                      \
                                                            \
        file_scope = m_def_tab.get_curr_scope();            \
        this_type = m_tc.classname##_type;                  \
        sym  = get_predef_symbol(#classname);               \
        def  = m_def_tab.enter_definition(Definition::DK_TYPE, sym, this_type, NULL); \
        def->set_flag(Definition::DEF_IS_PREDEFINED);       \
        type_scope = m_def_tab.enter_scope(this_type, def); \
        def->set_own_scope(type_scope);                     \
        def->set_version_flags(VERSION(flags));

// EXPLICIT constructor behavior
#undef EXPLICIT
#define EXPLICIT def->set_flag(Definition::DEF_IS_EXPLICIT);

#undef EXPLICIT_WARN
#define EXPLICIT_WARN def->set_flag(Definition::DEF_IS_EXPLICIT_WARN);

// IMPLICIT constructor behavior
#undef IMPLICIT
#define IMPLICIT /* do nothing */

// declare a constructor
#define CONSTRUCTOR(kind, classname, args, sema, flags)                         \
    if (available(flags)) {                           \
        _FUNCTION(Definition::DK_CONSTRUCTOR, classname, classname, args, flags)\
        kind                                                                    \
        def->set_semantic(Definition::sema);                                    \
    } else if (m_module.is_builtins()) {                                        \
        /* ensure the builtins module has ALL constructors */                   \
        _FUNCTION(Definition::DK_CONSTRUCTOR, classname, classname, args, flags)\
        kind                                                                    \
        def->set_semantic(Definition::sema);                                    \
        def->set_flag(Definition::DEF_IGNORE_OVERLOAD);                         \
    }

// declare a (data) field
#define FIELD(classname, mod, ftype, fieldname, flags)                                      \
    type = m_tc.decorate_type(m_tc.ftype##_type, mod_##mod);                                \
    sym  = get_predef_symbol(#fieldname);                                                   \
    if (available(flags)) {                                                                 \
        def  = m_def_tab.enter_definition(Definition::DK_MEMBER, sym, type, NULL);          \
        def->set_field_index(num_fields++);                                                 \
        def->set_version_flags(VERSION(flags));                                             \
    }                                                                                       \
    if (m_build_predefined_types && this_type->get_kind() == IType::TK_STRUCT) {            \
        IType_struct *s_type = const_cast<IType_struct *>(cast<IType_struct>(this_type));   \
        s_type->add_field(type, sym);                                                       \
    }

// handle enum values
#define ENUM_VALUE(classname, name, value, flags)                                   \
    m_def_tab.transition_to_scope(file_scope);                                      \
    sym  = get_predef_symbol(#name);                                                \
    def  = m_def_tab.enter_definition(Definition::DK_ENUM_VALUE, sym, this_type, NULL); \
    {                                                                               \
        IType_enum *e_type = const_cast<IType_enum *>(cast<IType_enum>(this_type)); \
        if (m_build_predefined_types)                                               \
            idx = e_type->add_value(sym, value);                                    \
        IValue const *enum_value = m_value_fact.create_enum(e_type, idx++);         \
        def->set_constant_value(enum_value);                                        \
        def->set_flag(Definition::DEF_IS_PREDEFINED);                               \
    }                                                                               \
    m_def_tab.transition_to_scope(type_scope);

// declare the end of a builtin type.
#define BUILTIN_TYPE_END(classname)                     \
        m_def_tab.leave_scope();                        \
    }
#include "compilercore_known_defs.h"

public:
    void enter_builtins(bool is_stdlib)
    {
#define BUILTIN_TYPE_BEGIN(classname, flags) build_type_##classname();
#include "compilercore_known_defs.h"

        enter_eq_operators();
        enter_binary_plus_minus_operators();
        enter_rel_operators();
        enter_mul_operators();
        enter_mul_assign_operators();
        enter_assign_operators();
        enter_operators();
    }

    /// enter operator== and operator!=
    void enter_eq_operators() {
        static const IExpression::Operator ops[] = {
            IExpression::OK_EQUAL, IExpression::OK_NOT_EQUAL
        };

        IType const *bool_type = m_tc.bool_type;
        for (unsigned int i = 0; i < 2; ++i)
        {
            IType const          *ptype;
            ISymbol const        *sym = NULL, *psym = NULL;
            IType_function const *func_type;
            int                  num_args = 0, arg_num;

            sym = m_sym_tab.get_operator_symbol(ops[i]);

// declare operators
#define EQ_OPERATORS(args, flags)                                       \
            arg_num = -1;                                               \
            args                                                        \
            func_type = m_tc.create_function(bool_type,                 \
                Type_cache::Function_parameters(m_params, num_args));   \
            m_def_tab.enter_operator_definition(ops[i], sym, func_type);

#include "compilercore_known_defs.h"
        }
    }

     /// enter binary operator+ and operator-
    void enter_binary_plus_minus_operators() {
        static const IExpression::Operator ops[] = {
            IExpression::OK_PLUS, IExpression::OK_MINUS
        };

        for (unsigned int i = 0; i < 2; ++i)
        {
            IType const          *ptype;
            ISymbol const        *sym = NULL, *psym = NULL;
            IType_function const *func_type;
            size_t               num_args = 0;
            int                  arg_num;

            sym = m_sym_tab.get_operator_symbol(ops[i]);

// declare operators
#define BINARY_PLUS_MINUS_OPERATORS(ret, args, flags)                   \
            arg_num = -1;                                               \
            args                                                        \
            func_type = m_tc.create_function(m_tc.ret##_type,           \
                Type_cache::Function_parameters(m_params, num_args));   \
            m_def_tab.enter_operator_definition(ops[i], sym, func_type);

#include "compilercore_known_defs.h"
        }
    }

    /// enter operator<, operator<=, operator>, operator>=
    void enter_rel_operators()
    {
        static const IExpression::Operator ops[] = {
            IExpression::OK_LESS,
            IExpression::OK_LESS_OR_EQUAL,
            IExpression::OK_GREATER,
            IExpression::OK_GREATER_OR_EQUAL
        };

        IType const *bool_type = m_tc.bool_type;
        for (unsigned i = 0; i < 4; ++i) {
            IType const          *ptype;
            ISymbol const        *sym = NULL, *psym = NULL;
            IType_function const *func_type;
            size_t               num_args = 0;
            int                  arg_num;

            sym = m_sym_tab.get_operator_symbol(ops[i]);

// declare operators
#define REL_OPERATORS(args, flags)                                      \
            arg_num = -1;                                               \
            args                                                        \
            func_type = m_tc.create_function(bool_type,                 \
                Type_cache::Function_parameters(m_params, num_args));   \
            m_def_tab.enter_operator_definition(ops[i], sym, func_type);

#include "compilercore_known_defs.h"
        }
    }

    /// enter operator*
    void enter_mul_operators() {
        IType const          *ptype;
        ISymbol const        *sym = NULL, *psym = NULL;
        IType_function const *func_type;
        size_t               num_args = 0;
        int                  arg_num;

        sym = m_sym_tab.get_operator_symbol(IExpression::OK_MULTIPLY);

// declare operators
#define MUL_OPERATOR(ret, args, flags)                          \
        arg_num = -1;                                           \
        args                                                    \
        func_type = m_tc.create_function(m_tc.ret##_type,       \
            Type_cache::Function_parameters(m_params, num_args)); \
        m_def_tab.enter_operator_definition(              \
            IExpression::OK_MULTIPLY, sym, func_type);

#include "compilercore_known_defs.h"
    }

    /// enter operator*=
    void enter_mul_assign_operators() {
        IType const          *ptype;
        ISymbol const        *sym = NULL, *psym = NULL;
        Definition           *def;
        IType_function const *func_type;
        size_t               num_args = 0;
        int                  arg_num;

        sym = m_sym_tab.get_operator_symbol(
            IExpression::OK_MULTIPLY_ASSIGN);

// declare operators
#define MUL_ASSIGN_OPERATOR(ret, args, flags)                   \
        arg_num = -1;                                           \
        args                                                    \
        func_type = m_tc.create_function(m_tc.ret##_type,       \
            Type_cache::Function_parameters(m_params, num_args)); \
        def = m_def_tab.enter_operator_definition(              \
            IExpression::OK_MULTIPLY_ASSIGN, sym, func_type);   \
            def->set_flag(Definition::DEF_OP_LVALUE);

#include "compilercore_known_defs.h"
    }

    /// enter operator=
    void enter_assign_operators() {
        IType const          *ptype;
        ISymbol const        *sym = NULL, *psym = NULL;
        Definition           *def;
        IType_function const *func_type;
        size_t               num_args = 0;
        int                  arg_num;

        sym = m_sym_tab.get_operator_symbol(IExpression::OK_ASSIGN);

// declare operators
#define ASSIGN_OPERATOR(ret, args, flags)                       \
        arg_num = -1;                                           \
        args                                                    \
        func_type = m_tc.create_function(m_tc.ret##_type,       \
            Type_cache::Function_parameters(m_params, num_args)); \
        def = m_def_tab.enter_operator_definition(              \
            IExpression::OK_ASSIGN, sym, func_type);            \
            def->set_flag(Definition::DEF_OP_LVALUE);

#include "compilercore_known_defs.h"
    }

    /// enter all other operators
    void enter_operators() {
        IType const          *ptype;
        ISymbol const        *sym = NULL, *psym = NULL;
        Definition           *def;
        IType_function const *func_type;
        size_t               num_args = 0;
        int                  arg_num;

#define REL_OPERATORS(args, flags)
#define EQ_OPERATORS(args, flags)
#define BINARY_PLUS_MINUS_OPERATORS(ret, args, flags)
#define MUL_OPERATOR(ret, args, flags)
#define MUL_ASSIGN_OPERATOR(ret, args, flags)
#define ASSIGN_OPERATOR(ret, args, flags)

// declare a builtin operator
#define OPERATOR(ret, code, args, flags)                        \
        arg_num = -1;                                           \
        args                                                    \
        sym = m_sym_tab.get_operator_symbol(IExpression::code); \
        func_type = m_tc.create_function(m_tc.ret##_type,       \
            Type_cache::Function_parameters(m_params, num_args)); \
        def = m_def_tab.enter_operator_definition(              \
            IExpression::code, sym, func_type);                 \
        mark_lvalue(def, IExpression::code);

#include "compilercore_known_defs.h"
    }

    /// Mark those operator definitions that require an lvalue first argument.
    void mark_lvalue(Definition *def, IExpression::Operator op) {
        switch (op) {
        case IExpression::OK_PRE_INCREMENT:
        case IExpression::OK_PRE_DECREMENT:
        case IExpression::OK_POST_INCREMENT:
        case IExpression::OK_POST_DECREMENT:
        case IExpression::OK_ASSIGN:
        case IExpression::OK_MULTIPLY_ASSIGN:
        case IExpression::OK_DIVIDE_ASSIGN:
        case IExpression::OK_MODULO_ASSIGN:
        case IExpression::OK_PLUS_ASSIGN:
        case IExpression::OK_MINUS_ASSIGN:
        case IExpression::OK_SHIFT_LEFT_ASSIGN:
        case IExpression::OK_SHIFT_RIGHT_ASSIGN:
        case IExpression::OK_UNSIGNED_SHIFT_RIGHT_ASSIGN:
        case IExpression::OK_BITWISE_OR_ASSIGN:
        case IExpression::OK_BITWISE_XOR_ASSIGN:
        case IExpression::OK_BITWISE_AND_ASSIGN:
            def->set_flag(Definition::DEF_OP_LVALUE);
            break;
        default:
            break;
        }
    }
};

void enter_predefined_entities(
    Module     &module,
    Type_cache &tc,
    bool build_predefined_types)
{
    Entity_builder builder(module, tc, build_predefined_types);

    builder.enter_builtins(module.is_stdlib());
}

#undef T

}  // COMPILER
} // MI
