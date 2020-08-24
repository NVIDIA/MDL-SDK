/******************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <cstdio>

#include "compilercore_allocator.h"
#include "compilercore_def_table.h"
#include "compilercore_modules.h"
#include "compilercore_tools.h"
#include "compilercore_func_hash.h"
#include "compilercore_hash.h"

#undef DOUT

#if 0
#define DOUT(x)     printf x
#else
#define DOUT(x)
#endif

namespace mi {
namespace mdl {

typedef Store<Definition *> Def_store;

namespace {

static char const * const expr_kind_name[] = {
    "EK_INVALID",
    "EK_LITERAL",
    "EK_REFERENCE",
    "EK_UNARY",
    "EK_BINARY",
    "EK_CONDITIONAL",
    "EK_CALL",
    "EK_LET",
};

static char const * const expr_operator_name[] = {
    // unary
    "OK_BITWISE_COMPLEMENT",
    "OK_LOGICAL_NOT",
    "OK_POSITIVE",
    "OK_NEGATIVE",
    "OK_PRE_INCREMENT",
    "OK_PRE_DECREMENT",
    "OK_POST_INCREMENT",
    "OK_POST_DECREMENT",
    "OK_CAST",

    // binary
    "OK_SELECT",
    "OK_ARRAY_INDEX",
    "OK_MULTIPLY",
    "OK_DIVIDE",
    "OK_MODULO",
    "OK_PLUS",
    "OK_MINUS",
    "OK_SHIFT_LEFT",
    "OK_SHIFT_RIGHT",
    "OK_UNSIGNED_SHIFT_RIGHT",
    "OK_LESS",
    "OK_LESS_OR_EQUAL",
    "OK_GREATER_OR_EQUAL",
    "OK_GREATER",
    "OK_EQUAL",
    "OK_NOT_EQUAL",
    "OK_BITWISE_AND",
    "OK_BITWISE_XOR",
    "OK_BITWISE_OR",
    "OK_LOGICAL_AND",
    "OK_LOGICAL_OR",

    // binary assignments
    "OK_ASSIGN",
    "OK_MULTIPLY_ASSIGN",
    "OK_DIVIDE_ASSIGN",
    "OK_MODULO_ASSIGN",
    "OK_PLUS_ASSIGN",
    "OK_MINUS_ASSIGN",
    "OK_SHIFT_LEFT_ASSIGN",
    "OK_SHIFT_RIGHT_ASSIGN",
    "OK_UNSIGNED_SHIFT_RIGHT_ASSIGN",
    "OK_BITWISE_OR_ASSIGN",
    "OK_BITWISE_XOR_ASSIGN",
    "OK_BITWISE_AND_ASSIGN",

    "OK_SEQUENCE",

    // ternary
    "OK_TERNARY",
    // variadic
    "OK_CALL"
};

/// Check if the given type is a user defined type (but not an array).
static bool is_user_type(IType const *type)
{
    if (IType_struct const *s_type = as<IType_struct>(type)) {
        if (s_type->get_predefined_id() == IType_struct::SID_USER) {
            return true;
        }
    }
    else if (IType_enum const *e_type = as<IType_enum>(type)) {
        IType_enum::Predefined_id id = e_type->get_predefined_id();
        if (id == IType_enum::EID_USER || id == IType_enum::EID_TEX_GAMMA_MODE) {
            // although tex::gamma_mode is predefined in the compiler (due to its use
            // in the texture constructor), it IS a user type: There is even MDL code
            // for it
            return true;
        }
    }
    return false;
}

/// Helper class to compute the semantic function hash.
class Function_hasher : public Module_visitor, public ICallgraph_visitor {
public:
    /// Calculate all function hashes.
    bool calculate_hashes();

private:
    /// Visit a node of the call graph.
    void visit_cg_node(Call_node *node, ICallgraph_visitor::Order order) MDL_FINAL;

    bool pre_visit(IDeclaration_variable *var_decl) MDL_FINAL;

    bool pre_visit(IDeclaration_constant *var_decl) MDL_FINAL;

    bool pre_visit(IDeclaration_type_struct *s_decl) MDL_FINAL;

    bool pre_visit(IDeclaration_type_enum *e_decl) MDL_FINAL;

    bool pre_visit(IDeclaration_type_alias *a_decl) MDL_FINAL;

    bool pre_visit(IParameter *param) MDL_FINAL;

    bool pre_visit(IAnnotation_block *block) MDL_FINAL;

    void post_visit(IArgument_named *arg) MDL_FINAL;

    void post_visit(IArgument_positional *arg) MDL_FINAL;

    void post_visit(IType_name *tname) MDL_FINAL;

    IExpression *post_visit(IExpression_invalid *expr) MDL_FINAL;

    IExpression *post_visit(IExpression_literal *expr) MDL_FINAL;

    IExpression *post_visit(IExpression_reference *expr) MDL_FINAL;

    IExpression *post_visit(IExpression_unary *expr) MDL_FINAL;

    IExpression *post_visit(IExpression_binary *expr) MDL_FINAL;

    IExpression *post_visit(IExpression_conditional *expr) MDL_FINAL;

    IExpression *post_visit(IExpression_call *expr) MDL_FINAL;

    IExpression *post_visit(IExpression_let *expr) MDL_FINAL;

    void post_visit(IStatement_invalid *stmt) MDL_FINAL;

    void post_visit(IStatement_compound *stmt) MDL_FINAL;

    void post_visit(IStatement_declaration *stmt) MDL_FINAL;

    void post_visit(IStatement_expression *stmt) MDL_FINAL;

    void post_visit(IStatement_if *stmt) MDL_FINAL;

    void post_visit(IStatement_case *stmt) MDL_FINAL;

    void post_visit(IStatement_switch *stmt) MDL_FINAL;

    void post_visit(IStatement_while *stmt) MDL_FINAL;

    void post_visit(IStatement_do_while *stmt) MDL_FINAL;

    void post_visit(IStatement_for *stmt) MDL_FINAL;

    void post_visit(IStatement_break *stmt) MDL_FINAL;

    void post_visit(IStatement_continue *stmt) MDL_FINAL;

    void post_visit(IStatement_return *stmt) MDL_FINAL;

    /// Hash a type.
    void hash(IType const *t);

    /// Hash a value.
    void hash(IValue const *v);

    /// Hash a definition.
    void hash(IDefinition const *def);

    /// Hash an unsigned value.
    void hash(unsigned v) {
        DOUT(("unsigned %u\n", v));
        m_hasher.update(v);
    }

    /// Hash an integer value.
    void hash(int v) {
        DOUT(("int %d\n", v));
        m_hasher.update(v);
    }

    /// Hash an unsigned value.
    void hash(IExpression::Kind v) {
        DOUT(("IExpression::Kind %s\n", expr_kind_name[v]));
        m_hasher.update(unsigned(v));
    }

    /// Hash an unsigned value.
    void hash(IExpression_unary::Operator v) {
        DOUT(("IExpression_unary::Operator %s\n", expr_operator_name[v]));
        m_hasher.update(unsigned(v));
    }

    /// Hash an unsigned value.
    void hash(IExpression_binary::Operator v) {
        DOUT(("IExpression_binary::Operator %s\n", expr_operator_name[v]));
        m_hasher.update(unsigned(v));
    }

    /// Hash an unsigned value.
    void hash(IStatement::Kind v) {
        DOUT(("IStatement::Kind %u\n", v));
        m_hasher.update(unsigned(v));
    }

    /// Hash an unsigned value.
    void hash(IType::Kind v) {
        DOUT(("IType::Kind %u\n", v));
        m_hasher.update(unsigned(v));
    }

    /// Hash an unsigned value.
    void hash(IType_texture::Shape v) {
        DOUT(("IType_texture::Shape %u\n", v));
        m_hasher.update(unsigned(v));
    }

    /// Hash an unsigned value.
    void hash(IValue::Kind v) {
        DOUT(("IValue::Kind %u\n", v));
        m_hasher.update(unsigned(v));
    }

    /// Hash an unsigned value.
    void hash(IValue_texture::gamma_mode v) {
        DOUT(("IValue_texture::gamma_mode %u\n", v));
        m_hasher.update(unsigned(v));
    }

    /// Hash an unsigned value.
    void hash(IDefinition::Kind v) {
        DOUT(("IDefinition::Kind %u\n", v));
        m_hasher.update(unsigned(v));
    }

    /// Hash an unsigned value.
    void hash(IDefinition::Semantics v) {
        DOUT(("IDefinition::Semantics %u\n", v));
        m_hasher.update(unsigned(v));
    }

    /// Hash a character.
    void hash(char c) {
        DOUT(("char %u\n", unsigned(c)));
        m_hasher.update(c);
    }

    /// Hash a float value.
    void hash(float v) {
        DOUT(("float %f\n", v));
        m_hasher.update(v);
    }

    /// Hash a double value.
    void hash(double v) {
        DOUT(("double %f\n", v));
        m_hasher.update(v);
    }

    /// Hash a string value.
    void hash(char const *s) {
        DOUT(("string '%s'\n", s));
        m_hasher.update(s);
    }

    void hash(unsigned char const *data, size_t l) {
        DOUT(("data "));
        for (size_t i = 0; i < l; ++i) {
            DOUT(("%02x", data[i]));
        }
        DOUT(("\n"));
        m_hasher.update(data, l);
    }

    /// Hash a symbol name.
    void hash(ISymbol const *sym) {
        DOUT(("symbol '%s'\n", sym->get_name()));
        m_hasher.update(sym->get_name());
    }

    /// Reset the hasher.
    void reset(Definition const *def) {
        DOUT(("\nRESET %s\n", def->get_sym()->get_name()));
        m_hasher.restart();
    }

    /// Store the hash value.
    void store(Definition const *def);

public:
    /// Constructor.
    Function_hasher(
        Module                          &mod,
        Call_graph const                &cg,
        Module::Function_hash_set const &base_hashes)
    : m_mod(mod)
    , m_cg(cg)
    , m_base_hashes(base_hashes)
    , m_hasher()
    , m_def2hash_map(0, Def2hash_map::hasher(), Def2hash_map::key_equal(), cg.get_allocator())
    {
    }

private:
    /// The current module.
    Module &m_mod;

    /// The call graph.
    Call_graph const &m_cg;

    /// All known hashes of the ::base module.
    Module::Function_hash_set const &m_base_hashes;

    /// The hasher itself.
    MD5_hasher m_hasher;

    typedef ptr_hash_map<IDefinition const, IModule::Function_hash>::Type Def2hash_map;

    Def2hash_map m_def2hash_map;
};

// Calculate all function hashes.
bool Function_hasher::calculate_hashes()
{
    bool has_base_hash = false;

    m_hasher.restart();

    Call_graph_walker::walk(m_cg, this);

    // copy the hashes to the module
    m_mod.clear_function_hashes();
    for (Def2hash_map::const_iterator it(m_def2hash_map.begin()), end(m_def2hash_map.end());
        it != end;
        ++it)
    {
        IDefinition const            *def = it->first;
        IModule::Function_hash const &hash = it->second;

        m_mod.add_function_hash(def, hash);

        if (m_base_hashes.find(hash) != m_base_hashes.end()) {
            has_base_hash = true;
        }
    }
    return has_base_hash;
}

// Visit a node of the call graph.
void Function_hasher::visit_cg_node(Call_node *node, ICallgraph_visitor::Order order)
{
    if (order == ICallgraph_visitor::POST_ORDER) {
        // compute hashes bottom up, for reuse

        Definition const *def = node->get_definition();

        if (def->has_flag(Definition::DEF_IS_IMPORTED)) {
            // ignore so far
        } else {
            switch (def->get_kind()) {
            case IDefinition::DK_FUNCTION:
                {
                    IType_function const *f_type   = cast<IType_function>(def->get_type());
                    IType const          *ret_type = f_type->get_return_type();

                    if (!is_material_type(ret_type)) {
                        // a real function. We ignore materialsi.e. do not compute a hash for them.
                        IDeclaration const *decl = def->get_declaration();

                        if (decl == NULL) {
                            // compiler generated
                        } else {
                            if (is<IDeclaration_function>(decl)) {
                                IDeclaration_function const *fdecl =
                                    cast<IDeclaration_function>(decl);

                                reset(def);
                                visit(fdecl);
                                store(def);
                            }
                        }
                    }
                }
                break;
            case IDefinition::DK_CONSTRUCTOR:
                // all constructors are compiler generated in MDL, hence it is enough to hash
                // the constructed type
                {
                    IType_function const *f_type   = cast<IType_function>(def->get_type());
                    IType const          *ret_type = f_type->get_return_type();

                    reset(def);
                    hash(ret_type);

                    // there might be copy, default, and elemental constructors, so add
                    // the semantics
                    hash(def->get_semantics());
                    store(def);
                }
                break;
            default:
                break;
            }
        }
    }
}

// Store the hash value.
void Function_hasher::store(Definition const *def)
{
    IModule::Function_hash res;
    m_hasher.final(res.hash);

    DOUT(("FINAL '%s' ", def->get_sym()->get_name()));

    for (size_t i = 0; i < dimension_of(res.hash); ++i) {
        DOUT(("%02x", res.hash[i]));
    }
    DOUT(("\n"));

    m_def2hash_map[def] = res;
}

bool Function_hasher::pre_visit(IDeclaration_variable *var_decl)
{
    for (int i = 0, n = var_decl->get_variable_count(); i < n; ++i) {
        ISimple_name const *v_name = var_decl->get_variable_name(i);
        IDefinition const  *v_def  = v_name->get_definition();

        hash(v_def);
        if (IExpression const *init = var_decl->get_variable_init(i))
            visit(init);
    }
    // do not visit children
    return false;
}

bool Function_hasher::pre_visit(IDeclaration_constant *cnst_decl)
{
    // ignore constants, these are replaced by its values; do not visit children
    return false;
}

bool Function_hasher::pre_visit(IDeclaration_type_struct *s_decl)
{
    // ignore user type declarations; do not visit children
    return false;
}

bool Function_hasher::pre_visit(IDeclaration_type_enum *e_decl)
{
    // ignore user type declarations; do not visit children
    return false;
}

bool Function_hasher::pre_visit(IDeclaration_type_alias *a_decl)
{
    // ignore user type declarations; do not visit children
    return false;
}

bool Function_hasher::pre_visit(IParameter *param)
{
    IDefinition const *def = param->get_name()->get_definition();

    // hash a parameter only by its type and its position, ignore the default initializer
    IType const *p_type = def->get_type();

    hash(p_type);

    // do not visit children
    return false;
}

bool Function_hasher::pre_visit(IAnnotation_block *block)
{
    // ignore annotations completely
    return false;
}

void Function_hasher::post_visit(IArgument_named *arg) {}

void Function_hasher::post_visit(IArgument_positional *arg) {}

void Function_hasher::post_visit(IType_name *tname) {}

IExpression *Function_hasher::post_visit(IExpression_invalid *expr)
{
    hash(expr->get_kind());
    return expr;
}

IExpression *Function_hasher::post_visit(IExpression_literal *expr)
{
    hash(expr->get_kind());
    hash(expr->get_value());
    return expr;
}

IExpression *Function_hasher::post_visit(IExpression_reference *expr)
{
    hash(expr->get_kind());

    if (expr->is_array_constructor()) {
        hash('A');
    } else {
        IDefinition const *def = expr->get_definition();
        hash(def);
    }
    return expr;
}

IExpression *Function_hasher::post_visit(IExpression_unary *expr)
{
    // arguments are already hashed
    hash(expr->get_kind());
    hash(expr->get_operator());
    return expr;
}

IExpression *Function_hasher::post_visit(IExpression_binary *expr)
{
    // arguments are already hashed
    hash(expr->get_kind());
    hash(expr->get_operator());
    return expr;
}

IExpression *Function_hasher::post_visit(IExpression_conditional *expr)
{
    // arguments are already hashed
    hash(expr->get_kind());
    return expr;
}

IExpression *Function_hasher::post_visit(IExpression_call *expr)
{
    // arguments are already hashed
    hash(expr->get_kind());
    return expr;
}

IExpression *Function_hasher::post_visit(IExpression_let *expr)
{
    // decls /expression are already hashed
    hash(expr->get_kind());
    return expr;
}

void Function_hasher::post_visit(IStatement_invalid *stmt)
{
    hash(stmt->get_kind());
}

void Function_hasher::post_visit(IStatement_compound *stmt)
{
    // do nothing here, { s } ans s are equivalent
}

void Function_hasher::post_visit(IStatement_declaration *stmt)
{
    // do nothing, just process the declaration
}

void Function_hasher::post_visit(IStatement_expression *stmt)
{
    // do nothing, just process the expression
}

void Function_hasher::post_visit(IStatement_if *stmt)
{
    // children already processed
    hash(stmt->get_kind());
}

void Function_hasher::post_visit(IStatement_case *stmt)
{
    // children already processed
    hash(stmt->get_kind());
}

void Function_hasher::post_visit(IStatement_switch *stmt)
{
    // children already processed
    hash(stmt->get_kind());
}

void Function_hasher::post_visit(IStatement_while *stmt)
{
    // children already processed
    hash(stmt->get_kind());
}

void Function_hasher::post_visit(IStatement_do_while *stmt)
{
    // children already processed
    hash(stmt->get_kind());
}

void Function_hasher::post_visit(IStatement_for *stmt)
{
    // children already processed
    hash(stmt->get_kind());
}

void Function_hasher::post_visit(IStatement_break *stmt)
{
    hash(stmt->get_kind());
}

void Function_hasher::post_visit(IStatement_continue *stmt)
{
    hash(stmt->get_kind());
}

void Function_hasher::post_visit(IStatement_return *stmt)
{
    // children already processed
    hash(stmt->get_kind());
}

// Hash a type.
void Function_hasher::hash(IType const *tp)
{
    IType::Kind kind = tp->get_kind();

    switch (kind) {
    case IType::TK_ALIAS:
        {
            IType_alias const *a_tp = cast<IType_alias>(tp);
            // ignore the name

            IType::Modifiers m = a_tp->get_type_modifiers();
            if (m != IType::MK_NONE) {
                hash(kind);
                hash(m);
            }
            hash(a_tp->skip_type_alias());
        }
        return;

    case IType::TK_BOOL:
    case IType::TK_INT:
    case IType::TK_FLOAT:
    case IType::TK_DOUBLE:
    case IType::TK_STRING:
    case IType::TK_LIGHT_PROFILE:
    case IType::TK_BSDF:
    case IType::TK_HAIR_BSDF:
    case IType::TK_EDF:
    case IType::TK_VDF:
    case IType::TK_COLOR:
    case IType::TK_BSDF_MEASUREMENT:
        // for all these, it is enough to hash the kind
        hash(kind);
        return;

    case IType::TK_ENUM:
        {
            IType_enum const *e_tp = cast<IType_enum>(tp);
            int              n     = e_tp->get_value_count();

            hash(kind);
            hash(n);

            // do NOT hash the names, just the values
            for (int i = 0; i < n; ++i) {
                ISymbol const *v_name;
                int code;

                e_tp->get_value(i, v_name, code);
                hash(code);
            }
        }
        return;

    case IType::TK_VECTOR:
        {
            IType_vector const *v_tp = cast<IType_vector>(tp);
            IType const        *e_tp = v_tp->get_element_type();

            hash(kind);
            hash(e_tp);
            hash(v_tp->get_size());
        }
        return;

    case IType::TK_MATRIX:
        {
            IType_matrix const *m_tp = cast<IType_matrix>(tp);
            IType const        *e_tp = m_tp->get_element_type();

            hash(kind);
            hash(e_tp);
            hash(m_tp->get_columns());
        }
        return;

    case IType::TK_ARRAY:
        {
            IType_array const *a_tp = cast<IType_array>(tp);
            IType const       *e_tp = a_tp->get_element_type();

            hash(kind);
            hash(e_tp);
            if (a_tp->is_immediate_sized()) {
                hash(a_tp->get_size());
            } else {
                IType_array_size const *size = a_tp->get_deferred_size();
                ISymbol const          *sym  = size->get_size_symbol();

                hash(sym);
            }
        }
        return;

    case IType::TK_FUNCTION:
        {
            IType_function const *f_tp   = cast<IType_function>(tp);
            IType const          *ret_tp = f_tp->get_return_type();

            hash(kind);

            if (ret_tp != NULL) {
                hash(ret_tp);
            }

            hash(f_tp->get_parameter_count());
            for (int i = 0, n = f_tp->get_parameter_count(); i < n; ++i) {
                ISymbol const *p_sym = NULL;
                IType const   *p_tp = NULL;

                f_tp->get_parameter(i, p_tp, p_sym);
                hash(p_tp);
                hash(p_sym);
            }
        }
        return;

    case IType::TK_STRUCT:
        {
            IType_struct const *s_tp = cast<IType_struct>(tp);

            hash(kind);
            hash(s_tp->get_field_count());
            for (int i = 0, n = s_tp->get_field_count(); i < n; ++i) {
                ISymbol const *f_sym = NULL;
                IType const   *f_tp  = NULL;

                s_tp->get_field(i, f_tp, f_sym);
                hash(f_tp);
            }
        }
        return;

    case IType::TK_TEXTURE:
        {
            IType_texture const *t_tp = cast<IType_texture>(tp);

            hash(kind);
            hash(t_tp->get_shape());
        }
        return;

    case IType::TK_INCOMPLETE:
    case IType::TK_ERROR:
        MDL_ASSERT(!"unexpected type kind");
        hash(kind);
        return;
    }
    MDL_ASSERT(!"unexpected type kind");
}

// Hash a value.
void Function_hasher::hash(IValue const *v)
{
    IValue::Kind kind = v->get_kind();

    hash(kind);
    switch (kind) {
    case IValue::VK_BAD:
        return;
    case IValue::VK_BOOL:
        {
            IValue_bool const *b = cast<IValue_bool>(v);
            hash(b->get_value()? 't' : 'f');
        }
        return;
    case IValue::VK_INT:
        {
            IValue_int const *i = cast<IValue_int>(v);
            hash(i->get_value());
        }
        return;
    case IValue::VK_ENUM:
        {
            IValue_enum const *e = cast<IValue_enum>(v);
            hash(e->get_value());
        }
        return;
    case IValue::VK_FLOAT:
        {
            IValue_float const *f = cast<IValue_float>(v);
            hash(f->get_value());
        }
        return;
    case IValue::VK_DOUBLE:
        {
            IValue_double const *d = cast<IValue_double>(v);
            hash(d->get_value());
        }
        return;
    case IValue::VK_STRING:
        {
            IValue_string const *s = cast<IValue_string>(v);
            hash(s->get_value());
        }
        return;

    case IValue::VK_VECTOR:
    case IValue::VK_MATRIX:
    case IValue::VK_ARRAY:
    case IValue::VK_RGB_COLOR:
    case IValue::VK_STRUCT:
        {
            IValue_compound const *c = cast<IValue_compound>(v);
            int n = c->get_component_count();

            hash(n);
            for (int i = 0; i < n; ++i) {
                IValue const *e = c->get_value(i);

                hash(e);
            }
        }
        return;

    case IValue::VK_INVALID_REF:
        // nothing more
        return;
    case IValue::VK_TEXTURE:
        {
            IValue_texture const *t = cast<IValue_texture>(v);

            hash(t->get_string_value());
            hash(t->get_tag_value());
            hash(t->get_tag_version());
            hash(t->get_gamma_mode());
        }
        return;

    case IValue::VK_LIGHT_PROFILE:
    case IValue::VK_BSDF_MEASUREMENT:
        {
            IValue_resource const *r = cast<IValue_resource>(v);

            hash(r->get_string_value());
            hash(r->get_tag_value());
            hash(r->get_tag_version());
        }
        return;
    }
    MDL_ASSERT(!"unsupported value kind");
}

// Hash a definition.
void Function_hasher::hash(IDefinition const *def)
{
    IDefinition::Kind kind = def->get_kind();

    DOUT(("(Definition '%s')\n", def->get_symbol()->get_name()));
    hash(kind);
    switch (kind) {
    case IDefinition::DK_ERROR:
        return;
    case IDefinition::DK_CONSTANT:
    case IDefinition::DK_ENUM_VALUE:
        {
            IValue const *c = def->get_constant_value();
            hash(c);
        }
        return;
    case IDefinition::DK_ANNOTATION:
        MDL_ASSERT(!"unexpected annotation definition inside a hashed value");
        return;
    case IDefinition::DK_TYPE:
        {
            IType const *type = def->get_type();
            hash(type);
        }
        return;
    case IDefinition::DK_FUNCTION:
        {
            Def2hash_map::const_iterator it = m_def2hash_map.find(def);

            if (it != m_def2hash_map.end()) {
                // known function call
                hash(it->second.hash, dimension_of(it->second.hash));
            } else {
                MDL_ASSERT(def->get_property(IDefinition::DP_IS_IMPORTED));

                hash(m_mod.get_owner_module_name(def));
                hash(def->get_symbol());
                hash(def->get_type());
            }
        }
        return;
    case IDefinition::DK_VARIABLE:
        {
            // in MDL, this is always a local variable
            IType const *v_type = def->get_type();
            hash(v_type);

            // we could map the variable name here to something different
            ISymbol const *sym = def->get_symbol();
            hash(sym);
        }
        return;
    case IDefinition::DK_MEMBER:
        {
            IType const *v_type = def->get_type();
            hash(v_type);

            // use the field index
            hash(def->get_field_index());
        }
        return;
    case IDefinition::DK_CONSTRUCTOR:
        {
            Def2hash_map::const_iterator it = m_def2hash_map.find(def);

            if (it != m_def2hash_map.end()) {
                // known function call
                hash(it->second.hash, dimension_of(it->second.hash));
            } else {
                IType const *type = def->get_type();
                hash(type);

                MDL_ASSERT(!is_user_type(type) || def->get_property(IDefinition::DP_IS_IMPORTED));

                // the semantics should be enough to identify the constructor
                IDefinition::Semantics sema = def->get_semantics();
                hash(sema);
            }
        }
        return;
    case IDefinition::DK_PARAMETER:
        {
            IType const *v_type = def->get_type();
            hash(v_type);

            // use the parameter index
            hash(def->get_parameter_index());
        }
        return;
    case IDefinition::DK_ARRAY_SIZE:
        {
            // the type is always int, ignore it
            ISymbol const *sym = def->get_symbol();
            hash(sym);
        }
        return;
    case IDefinition::DK_OPERATOR:
        {
            IType const *type = def->get_type();
            hash(type);

            // the semantics should be enough to identify the operator
            IDefinition::Semantics sema = def->get_semantics();
            hash(sema);
        }
        return;
    case IDefinition::DK_NAMESPACE:
        MDL_ASSERT(!"NYI");
        return;
    }
    MDL_ASSERT(!"unknown definition kind");
}

}  // anonymous

// Compute the hashes.
void Sema_hasher::run(
    IAllocator *alloc,
    Module     *mod,
    MDL        *compiler)
{
    if (mod->is_valid()) {
        Sema_hasher(alloc, mod, compiler).compute_hashes();
    }
}

// Compute the semantic hashes for all exported functions.
void Sema_hasher::compute_hashes()
{
    Module::Function_hash_set hashes(Module::Function_hash_set::key_compare(), get_allocator());

    Module const *base_mod = m_compiler->find_builtin_module(string("::base", get_allocator()));
    if (base_mod != NULL) {
        base_mod->get_all_function_hashes(hashes);
    }

    // first step: recompute the call graph
    for (int i = 0, n = m_mod.get_exported_definition_count(); i < n; ++i) {
        Definition const *def = m_mod.get_exported_definition(i);

        if (def->has_flag(Definition::DEF_IS_IMPORTED)) {
            // ignore so far
            continue;
        }

        switch (def->get_kind()) {
        case IDefinition::DK_FUNCTION:
            {
                // found a function or an material, add it to the call graph:
                // This is necessary, to catch local functions that are called from the material
                // body only
                Definition *d = const_cast<Definition *>(def);

                m_cg.add_node(d);
                m_wq.push(d);

                IType_function const *ftype = cast<IType_function>(def->get_type());
                IType const          *ret_type = ftype->get_return_type();

                if (!is_material_type(ret_type)) {
                    // hashes are only computed for functions, not for materials
                    m_hashes[d] = NULL;
                }
            }
            break;
        default:
            // ignore
            break;
        }
    }

    while (!m_wq.empty()) {
        Definition *def = m_wq.front();

        m_wq.pop();

        if (IDeclaration const *decl = def->get_declaration()) {
            // constructors are functions, but point to s type declaration, so check that here
            if (is<IDeclaration_function>(decl)) {
                IDeclaration_function const *f_decl = cast<IDeclaration_function>(decl);

                if (IStatement const *body = f_decl->get_body()) {
                    // visit the body and build the call graph
                    Def_store curr_node(m_curr_def, def);
                    visit(body);
                }
            }
        }
    }

    m_cg.finalize();

    // now visit the call graph bottom up and calculate the hashes

    Function_hasher fh(m_mod, m_cg, hashes);

    bool has_base_hashes = fh.calculate_hashes();
    if (has_base_hashes) {
        // This is a work-around for the current material converter in iray.
        // Currently, the only module that can be replaced by hashes is ::base
        // However, at the time when the material converter does the swap, base might not be
        // loaded. Hence, if a module references base using hashes, add an import dependency,
        // so base will be loaded.
        m_mod.register_import(base_mod);
    }
}

IExpression *Sema_hasher::post_visit(IExpression_call *call)
{
    IExpression_reference const *ref = cast<IExpression_reference>(call->get_reference());

    if (ref->is_array_constructor()) {
        return call;
    }

    Definition *def = const_cast<Definition *>(impl_cast<Definition>(ref->get_definition()));

    if (def->has_flag(Definition::DEF_IS_IMPORTED)) {
        // ignored so far, so do not add them to the CG
        return call;
    }

    if (def->get_kind() == IDefinition::DK_CONSTRUCTOR) {
        IType_function const *f_type   = cast<IType_function>(def->get_type());
        IType const          *ret_type = f_type->get_return_type();

        if (!is_user_type(ret_type)) {
            // compute hashes only for user type constructors
            return call;
        }
    }

    m_cg.add_call(m_curr_def, def);

    if (m_hashes.find(def) == m_hashes.end()) {
        m_hashes[def] = NULL;
        m_wq.push(def);
    }
    return call;
}

// Constructor.
Sema_hasher::Sema_hasher(
    IAllocator *alloc,
    Module     *mod,
    MDL        *compiler)
: m_mod(*mod)
, m_cg(alloc, mod->get_name(), mod->is_stdlib())
, m_compiler(compiler)
, m_curr_def(NULL)
, m_hashes(0, Sema_hash_map::hasher(), Sema_hash_map::key_equal(), alloc)
, m_wq(Wait_queue::container_type(get_allocator()))
{
}

}  // mdl
}  // mi
