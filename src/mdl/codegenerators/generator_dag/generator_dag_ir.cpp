/******************************************************************************
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <cmath>

#include <mi/mdl/mdl_values.h>

#include "mdl/compiler/compilercore/compilercore_cc_conf.h"
#include "mdl/compiler/compilercore/compilercore_allocator.h"
#include "mdl/compiler/compilercore/compilercore_mdl.h"
#include "mdl/compiler/compilercore/compilercore_tools.h"
#include "mdl/codegenerators/generator_code/generator_code.h"

#include "generator_dag_ir.h"
#include "generator_dag_builder.h"
#include "generator_dag_tools.h"

namespace mi {
namespace mdl {

namespace {

/// Calculate the name hash.
static size_t calc_name_hash(char const *name)
{
    size_t hash = 0;
    for (size_t i = 0; name[i] != '\0'; ++i)
        hash = hash * 9 ^ size_t(name[i]);
    return hash;
}

}

// -------------------------- DAG IR node --------------------------

/// Abstract base class for all expressions.
///
/// \tparam T  the implemented expression interface type
template<typename T>
class Expression_impl : public T {
public:
    /// Get the kind of the expression.
    ///
    /// \returns    The kind of generated code.
    ///
    typename DAG_node::Kind get_kind() const MDL_FINAL { return T::s_kind; }

    /// Get the ID of this DAG IR node.
    size_t get_id() const MDL_FINAL { return m_id; }

protected:
    /// Constructor.
    ///
    /// \param id  The unique ID of this node.
    explicit Expression_impl(size_t id)
    : m_id(id)
    {
    }

private:
    /// The unique id.
    size_t const m_id;
};

/// A constant.
class Constant_impl : public Expression_impl<DAG_constant> {
    typedef Expression_impl<DAG_constant> Base;
    friend class Arena_builder;
public:

    /// Get the value of the constant.
    IValue const *get_value() const MDL_FINAL { return m_value; }

    /// Get the type of the constant.
    IType const *get_type() const MDL_FINAL { return m_value->get_type(); }

    // non-interface methods
    void set_value(IValue const *v) { m_value = v; }

private:
    /// Constructor.
    ///
    /// \param id     The unique ID of this node.
    /// \param value  The value of this constant.
    Constant_impl(size_t id, IValue const *value)
    : Base(id)
    , m_value(value)
    {   
    }

private:
    /// The value of this constant.
    IValue const *m_value;
};

/// A temporary reference.
class Temporary_impl : public Expression_impl<DAG_temporary> {
    typedef Expression_impl<DAG_temporary> Base;
    friend class Arena_builder;
public:
    /// Get the index of the referenced temporary.
    int get_index() const MDL_FINAL { return m_index; }

    // Get the expression of the temporary.
    DAG_node const *get_expr() const MDL_FINAL { return m_node; }

    /// Get the type of the temporary.
    IType const *get_type() const MDL_FINAL { return m_node->get_type(); }

private:
    /// Constructor.
    ///
    /// \param id     The unique ID of this node.
    /// \param node   The DAG node that is "named" by this temporary.
    /// \param index  The index of this temporary.
    Temporary_impl(size_t id, DAG_node const *node, Uint32 index)
    : Base(id)
    , m_node(node), m_index(index)
    {
    }

private:
    /// The node that is "named" by this index.
    DAG_node const * const m_node;

    /// The index of the referenced temporary.
    Uint32 const m_index;
};

/// A call.
class Call_impl : public Expression_impl<DAG_call> {
    typedef Expression_impl<DAG_call> Base;
    friend class Arena_builder;
public:
    /// Get return type.
    /// Get the return type of the called function.
    ///
    /// \returns            The return type of the called function.
    IType const *get_type() const MDL_FINAL { return m_ret_type; }

    /// Get the name of the called function.
    char const *get_name() const MDL_FINAL { return m_name; }

    /// Get the number of arguments.
    int get_argument_count() const MDL_FINAL { return m_arguments.size(); }

    /// Get the name of the parameter corresponding to the argument at position index.
    char const *get_parameter_name(int index) const MDL_FINAL
    {
        if ((index < 0) || (m_parameter_names.size() <= size_t(index)))
            return NULL;
        return m_parameter_names[index];

    }

    /// Get the argument at position index.
    DAG_node const *get_argument(int index) const MDL_FINAL
    {
        if ((index < 0) || (m_arguments.size() <= size_t(index)))
            return NULL;
        return m_arguments[index];
    }

    /// Get the argument for parameter name.
    DAG_node const *get_argument(const char *name) const MDL_FINAL
    {
        for (size_t i = 0, n = m_parameter_names.size(); i < n; ++i)
            if (strcmp(m_parameter_names[i], name) == 0)
                return m_arguments[i];
        return NULL;
    }

    /// Get the semantic of a call if known.
    IDefinition::Semantics get_semantic() const MDL_FINAL { return m_semantic; }

    /// Set the argument expression of a call.
    void set_argument(int index, DAG_node const *arg) MDL_FINAL
    {
        if (0 <= index && size_t(index) < m_arguments.size())
            m_arguments[index] = arg;
    }

    /// Get the name hash.
    size_t get_name_hash() const MDL_FINAL { return m_name_hash; }

private:
    /// Constructor.
    ///
    /// \param  id              The unique ID of this node.
    /// \param  arena           The memory arena use to store parameter names.
    /// \param  name            The name of the called function.
    /// \param  sema            The semantic of the called function.
    /// \param  call_args       The call arguments of the called function.
    /// \param  num_call_args   The number of call arguments.
    /// \param  ret_type        The return type of the called function.
    ///
    Call_impl(
        size_t                        id,
        Memory_arena                  *arena,
        char const                    *name,
        IDefinition::Semantics        sema,
        DAG_call::Call_argument const call_args[],
        size_t                        num_call_args,
        IType const                   *ret_type)
    : Base(id)
    , m_semantic(sema)
    , m_ret_type(ret_type)
    , m_name(name)
    , m_name_hash(calc_name_hash(name) ^ size_t(sema) * 9)
    , m_parameter_names(arena)
    , m_arguments(arena)
    {
        m_parameter_names.resize(num_call_args, NULL);
        m_arguments.resize(num_call_args, NULL);

        for (size_t i = 0; i < num_call_args; ++i) {
            // FIXME: for now, put the string on the arena
            char *n = Arena_strdup(*arena, call_args[i].param_name);
            m_parameter_names[i] = n;
            m_arguments[i]       = call_args[i].arg;
        }

        MDL_ASSERT(sema != operator_to_semantic(IExpression::OK_SELECT));
    }

private:
    /// The semantic of this call.
    IDefinition::Semantics const m_semantic;

    /// The return type of the function call.
    IType const *m_ret_type;

    /// The name of the function.
    char const * const m_name;

    /// The name hash value.
    size_t m_name_hash;

    /// The parameter names of the function.
    Arena_vector<char const *>::Type m_parameter_names;

    /// The arguments of the function.
    Arena_vector<DAG_node const *>::Type m_arguments;
};

/// A parameter reference.
class Parameter_impl : public Expression_impl<DAG_parameter> {
    typedef Expression_impl<DAG_parameter> Base;
    friend class Arena_builder;
public:
    /// Get the type of the referenced parameter.
    IType const *get_type() const MDL_FINAL { return m_type; }

    /// Get the index of the referenced parameter.
    int get_index() const MDL_FINAL { return m_index; }

    // non-interface methods

    /// Set a new index for this parameter.
    void set_index(Uint32 index) { m_index = index; }

private:
    /// Constructor.
    ///
    /// \param id     The unique ID of this node.
    /// \param type   The type of the parameter
    /// \param index  The index of this parameter.
    explicit Parameter_impl(size_t id, IType const *type, Uint32 index)
    : Base(id)
    , m_type(type)
    , m_index(index)
    {
    }

private:
    /// The type of this parameter.
    IType const * const m_type;

    /// The index of the referenced parameter.
    Uint32 m_index;
};

// -------------------------- Expression factory --------------------------

// A hash functor for Expressions.
size_t DAG_node_factory_impl::Hash_dag_node::operator()(
    DAG_node const *node) const
{
    DAG_node::Kind kind = node->get_kind();
    auto           it   = m_temp_name_map.find(node);
    size_t         hash = it != m_temp_name_map.end() ? calc_name_hash(it->second) : 0;

    switch (kind) {
    case DAG_node::EK_CONSTANT:
        {
            DAG_constant const *c = cast<DAG_constant>(node);
            hash = Hash_ptr<IValue>()(c->get_value());
            break;
        }
    case DAG_node::EK_TEMPORARY:
        {
            DAG_temporary const *t = cast<DAG_temporary>(node);
            hash = size_t(t->get_index());
            break;
        }
    case DAG_node::EK_CALL:
        {
            DAG_call const *c = cast<DAG_call>(node);

            hash = c->get_name_hash();
            for (int i = 0, n = c->get_argument_count(); i < n; ++i) {
                hash = hash * 3 ^ Hash_ptr<DAG_node>()(c->get_argument(i));
            }
            break;
        }
    case DAG_node::EK_PARAMETER:
        {
            DAG_parameter const *p = cast<DAG_parameter>(node);
            hash = size_t(p->get_index());
            break;
        }
    }
    return hash ^ (size_t(kind) << 8);
}

// An Equal functor for Expressions.
bool DAG_node_factory_impl::Equal_dag_node::operator()(
    DAG_node const *a, DAG_node const *b) const
{
    DAG_node::Kind kind = a->get_kind();

    if (kind != b->get_kind())
        return false;

    auto it_a = m_temp_name_map.find(a);
    auto it_b = m_temp_name_map.find(b);
    bool has_name_a = it_a != m_temp_name_map.end();
    bool has_name_b = it_b != m_temp_name_map.end();
    if (has_name_a != has_name_b)
        return false;

    if (has_name_a && strcmp(it_a->second, it_b->second) != 0)
        return false;

    switch (kind) {
    case DAG_node::EK_CONSTANT:
        {
            DAG_constant const *ca = cast<DAG_constant>(a);
            DAG_constant const *cb = cast<DAG_constant>(b);

            return ca->get_value() == cb->get_value();
        }
    case DAG_node::EK_TEMPORARY:
        {
            DAG_temporary const *ta = cast<DAG_temporary>(a);
            DAG_temporary const *tb = cast<DAG_temporary>(b);

            bool res = ta->get_index() == tb->get_index();
            MDL_ASSERT(!res || ta->get_expr() == tb->get_expr());
            return res;
    }
    case DAG_node::EK_CALL:
        {
            DAG_call const *ca = cast<DAG_call>(a);
            DAG_call const *cb = cast<DAG_call>(b);

            int n_args = ca->get_argument_count();
            if (n_args != cb->get_argument_count())
                return false;

            if (ca->get_semantic() != cb->get_semantic())
                return false;

            if (strcmp(ca->get_name(), cb->get_name()) != 0)
                return false;

            for (int i = 0; i < n_args; ++i)
                if (ca->get_argument(i) != cb->get_argument(i))
                    return false;

            MDL_ASSERT(ca->get_type() == cb->get_type());
            return true;
        }
    case DAG_node::EK_PARAMETER:
        {
            DAG_parameter const *pa = cast<DAG_parameter>(a);
            DAG_parameter const *pb = cast<DAG_parameter>(b);

            if (pa->get_index() != pb->get_index())
                return false;

            MDL_ASSERT(pa->get_type() == pb->get_type());
            return true;
        }
    }
    return false;
}

// Constructor.
DAG_node_factory_impl::DAG_node_factory_impl(
    IMDL          *mdl,
    Memory_arena  &arena,
    Value_factory &value_factory,
    char const    *internal_space)
: m_builder(arena)
, m_mdl(mi::base::make_handle_dup(mdl))
, m_value_factory(value_factory)
, m_sym_tab(*value_factory.get_type_factory()->get_symbol_table())
, m_internal_space(Arena_strdup(arena, internal_space))
, m_call_evaluator(NULL)
, m_next_id(0)
, m_cse_enabled(true)
, m_opt_enabled(true)
, m_unsafe_math_opt(true)
, m_expose_names_of_let_expressions(false)
, m_inline_allowed(true)
, m_noinline_ignored(false)
, m_needs_state_import(false)
, m_needs_nvidia_df_import(false)
, m_avoid_non_const_gamma(true)
, m_enable_scene_conv_fold(false)
, m_enable_wavelength_fold(false)
, m_mdl_meters_per_scene_unit(1.0f)
, m_state_wavelength_min(380.0f)
, m_state_wavelength_max(780.0f)
, m_temp_name_map(
    0,
    Definition_temporary_name_map::hasher(),
    Definition_temporary_name_map::key_equal(),
    arena.get_allocator())
, m_value_table(
    0,
    Value_table::hasher(m_temp_name_map),
    Value_table::key_equal(m_temp_name_map),
    arena.get_allocator())
{
}

// Create a constant.
DAG_constant const *DAG_node_factory_impl::create_constant(
    IValue const *value)
{
    MDL_ASSERT(value != NULL && !is<IValue_bad>(value));
    DAG_node *res = m_builder.create<Constant_impl>(m_next_id++, value);
    return static_cast<Constant_impl *>(identify_remember(res));
}

// Create a temporary reference.
DAG_temporary const *DAG_node_factory_impl::create_temporary(
    DAG_node const *node,
    int            index)
{
    DAG_node *res = m_builder.create<Temporary_impl>(m_next_id++, node, index);
    return static_cast<Temporary_impl *>(identify_remember(res));
}

/// Check if a DAG node represents the given operator kind.
static bool is_operator(DAG_node const *node, IExpression::Operator op)
{
    if (is<DAG_call>(node)) {
        IDefinition::Semantics sema = cast<DAG_call>(node)->get_semantic();

        return semantic_to_operator(sema) == op;
    }
    return false;
}

/// Check if a DAG expression represents the given operator kind.
static bool is_operator(DAG_node const *node, IExpression_unary::Operator op)
{
    return is_operator(node, IExpression::Operator(op));
}

/// Checks if the given IR node represents a zero.
static bool is_zero(DAG_node const *node)
{
    if (is<DAG_constant>(node)) {
        IValue const *v = cast<DAG_constant>(node)->get_value();
        return v->is_zero();
    }
    return false;
}

/// Checks if the given IR node represents a one.
static bool is_one(DAG_node const *node)
{
    if (is<DAG_constant>(node)) {
        IValue const *v = cast<DAG_constant>(node)->get_value();
        return v->is_one();
    }
    return false;
}

/// Checks if the given IR node is a all-one.
static bool is_all_one(DAG_node const *node)
{
    if (is<DAG_constant>(node)) {
        IValue const *v = cast<DAG_constant>(node)->get_value();
        return v->is_all_one();
    }
    return false;
}

/// Returns true if the given type has NaN or Inf.
static bool has_nan_or_inf(IType const *type)
{
    for (;;) {
        switch (type->get_kind()) {
        case IType::TK_BOOL:
        case IType::TK_INT:
            // boolean and integer do not have NaN's
            return false;
        case IType::TK_FLOAT:
        case IType::TK_DOUBLE:
            // floating point has NaN
            return true;
        case IType::TK_MATRIX:
            // matrices are always made of FP
            return true;
        case IType::TK_VECTOR:
            type = cast<IType_vector>(type)->get_element_type();
            break;
        default:
            // all others have no arithmetic, assume NaN
            return true;
        }
    }
}

/// Checks if a given IR node is known to be finite.
static bool is_finite(DAG_node const *node)
{
    IType const *type = node->get_type()->skip_type_alias();

    if (!has_nan_or_inf(type)) {
        // only finite values
        return true;
    }

    switch (node->get_kind()) {
    case DAG_node::EK_CONSTANT:
        {
            DAG_constant const *c = cast<DAG_constant>(node);
            IValue const       *v = c->get_value();

            return v->is_finite();
        }
    case DAG_node::EK_TEMPORARY:
        // should not happen, but if ...
        {
            DAG_temporary const *t = cast<DAG_temporary>(node);
            return is_finite(t->get_expr());
        }
    case DAG_node::EK_CALL:
        {
            DAG_call const *call = cast<DAG_call>(node);

            switch (call->get_semantic()) {
            case IDefinition::DS_INTRINSIC_STATE_POSITION:
            case IDefinition::DS_INTRINSIC_STATE_NORMAL:
            case IDefinition::DS_INTRINSIC_STATE_GEOMETRY_NORMAL:
            case IDefinition::DS_INTRINSIC_STATE_MOTION:
            case IDefinition::DS_INTRINSIC_STATE_TEXTURE_SPACE_MAX:
            case IDefinition::DS_INTRINSIC_STATE_TEXTURE_COORDINATE:
            case IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_U:
            case IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_V:
            case IDefinition::DS_INTRINSIC_STATE_TANGENT_SPACE:
            case IDefinition::DS_INTRINSIC_STATE_GEOMETRY_TANGENT_U:
            case IDefinition::DS_INTRINSIC_STATE_GEOMETRY_TANGENT_V:
            case IDefinition::DS_INTRINSIC_STATE_DIRECTION:
            case IDefinition::DS_INTRINSIC_STATE_ANIMATION_TIME:
            case IDefinition::DS_INTRINSIC_STATE_WAVELENGTH_BASE:
            case IDefinition::DS_INTRINSIC_STATE_TRANSFORM:
            case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_POINT:
            case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_VECTOR:
            case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_NORMAL:
            case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_SCALE:
            case IDefinition::DS_INTRINSIC_STATE_ROUNDED_CORNER_NORMAL:
                // for all state functions it is guaranteed, that they return a finite value
                return true;
            default:
                // cannot determine
                return false;
            }
        }
    case DAG_node::EK_PARAMETER:
        // cannot determine
        return false;
    }
    return false;
}

// Get the field index from a getter function call name.
int DAG_node_factory_impl::get_field_index(
    IType_compound const *c_type,
    char const           *call_name)
{
    char const *p   = strstr(call_name, ".mdle::");
    char const *dot = strchr(p != NULL ? (p + 7) : call_name, '.');
    if (dot != NULL) {
        ++dot;
        if (char const *n = strchr(dot, '(')) {
            string field(dot, n - dot, get_allocator());

            if (IType_struct const *s_type = as<IType_struct>(c_type))
                return s_type->find_field(field.c_str());
            if (IType_vector const *v_type = as<IType_vector>(c_type)) {
                if (field.size() != 1)
                    return -1;
                int index = -1;

                switch (field[0]) {
                case 'x': index = 0; break;
                case 'y': index = 1; break;
                case 'z': index = 2; break;
                case 'w': index = 3; break;
                default:
                    break;
                }
                if (index < v_type->get_size())
                    return index;
            }
        }
    }
    return -1;
}

/// Check if two types are equal after aliases are skipped.
static bool equal_op_types(IType const *t, IType const *s)
{
    return t->skip_type_alias() == s->skip_type_alias();
}

/// Check if the given DAG IR node represents a call to state::normal().
static bool is_state_normal_call(DAG_node const *node)
{
    if (is<DAG_call>(node))
        return cast<DAG_call>(node)->get_semantic() == IDefinition::DS_INTRINSIC_STATE_NORMAL;
    return false;
}

/// Check if the given DAG IR node represents a float 1.0f constant.
static bool is_float_one(DAG_node const *node)
{
    if (is<DAG_constant>(node)) {
        IValue const *v = cast<DAG_constant>(node)->get_value();
        if (is<IValue_float>(v))
            return v->is_one();
    }
    return false;
}

/// Check if the given DAG IR node represents a float 0.0f constant.
static bool is_float_zero(DAG_node const *node)
{
    if (is<DAG_constant>(node)) {
        IValue const *v = cast<DAG_constant>(node)->get_value();
        if (is<IValue_float>(v))
            return v->is_zero();
    }
    return false;
}

// Build a call to a conversion from a ::tex::gamma value to int.
DAG_node const *DAG_node_factory_impl::build_gamma_conv(
    DAG_node const *x)
{
    char const *name = "::tex::int(::tex::gamma_mode)";

    DAG_call::Call_argument args[1];

    args[0].arg = x;
    args[0].param_name = "x";

    IType_factory *fact = m_value_factory.get_type_factory();

    return create_call(
        name,
        IDefinition::DS_CONV_OPERATOR,
        args,
        1,
        fact->create_int());
}

// Build a call to a operator== for a ::tex::gamma value.
DAG_node const *DAG_node_factory_impl::build_gamma_equal(
    DAG_node const *x,
    DAG_node const *y)
{
    char const *name = "operator==(int,int)";

    DAG_call::Call_argument args[2];

    args[0].arg = build_gamma_conv(x);
    args[0].param_name = "x";

    args[1].arg = build_gamma_conv(y);
    args[1].param_name = "y";

    IType_factory *fact = m_value_factory.get_type_factory();

    return create_call(
        name,
        operator_to_semantic(IExpression::OK_EQUAL),
        args,
        2,
        fact->create_bool());
}

// Build a call to a ternary operator for a texture.
DAG_node const *DAG_node_factory_impl::build_texture_ternary(
    DAG_node const *cond,
    DAG_node const *true_expr,
    DAG_node const *false_expr)
{
    IType const *ret_type = true_expr->get_type()->skip_type_alias();

    DAG_call::Call_argument args[3];

    args[0].arg        = cond;
    args[0].param_name = "cond";

    args[1].arg        = true_expr;
    args[1].param_name = "true_exp";

    args[2].arg        = false_expr;
    args[2].param_name = "false_exp";

    return create_call(
        get_ternary_operator_signature(),
        operator_to_semantic(IExpression::OK_TERNARY),
        args,
        dimension_of(args),
        ret_type);
}

// Avoid non-const gamma textures.
DAG_node const *DAG_node_factory_impl::do_avoid_non_const_gamma(
    IType_texture const *tex_type,
    DAG_constant const  *url,
    DAG_node const      *gamma)
{

    // there are only 3 possible gamma values. Transform this into
    // gamma = gamma_default ?
    //   T(..., gamma_default) :
    //   (gamma = gamma_linear ? T(..., gamma_linear) : T(..., gamma_srgb)

    IType_enum const *e_tp = cast<IType_enum>(gamma->get_type());
    MDL_ASSERT(e_tp->get_predefined_id() == IType_enum::EID_TEX_GAMMA_MODE);

    IValue_string const *v_string = cast<IValue_string>(url->get_value());

    IValue_texture const *v_tex_default = m_value_factory.create_texture(
        tex_type, v_string->get_value(), IValue_texture::gamma_default, 0, 0);
    DAG_node const    *c_tex_default    = create_constant(v_tex_default);

    IValue_texture const *v_tex_linear = m_value_factory.create_texture(
        tex_type, v_string->get_value(), IValue_texture::gamma_linear, 0, 0);
    DAG_node const    *c_tex_linear    = create_constant(v_tex_linear);

    IValue_texture const *v_tex_srgb = m_value_factory.create_texture(
        tex_type, v_string->get_value(), IValue_texture::gamma_srgb, 0, 0);
    DAG_node const    *c_tex_srgb  = create_constant(v_tex_srgb);

    IValue_enum const *v_default = m_value_factory.create_enum(e_tp, 0);
    DAG_node const    *c_default = create_constant(v_default);

    IValue_enum const *v_linear  = m_value_factory.create_enum(e_tp, 1);
    DAG_node const    *c_linear  = create_constant(v_linear);

    // f = gamma == gamma_linear ? tex_linear : tex_srgb;
    DAG_node const *f = build_texture_ternary(
        build_gamma_equal(gamma, c_linear),
        c_tex_linear,
        c_tex_srgb);

    // gamma == gamma_default ? tex_defaul : f
    return build_texture_ternary(
        build_gamma_equal(gamma, c_default),
        c_tex_default,
        f);
}

/// Check if a given node represents a float to color cast.
static bool is_color_from_float(DAG_node const *n)
{
    if (DAG_constant const *c = as<DAG_constant>(n)) {
        if (IValue_rgb_color const *rgb = as<IValue_rgb_color>(c->get_value())) {
            IValue_float const *r = rgb->get_value(0);
            IValue_float const *g = rgb->get_value(1);

            if (r != g)
                return false;

            IValue_float const *b = rgb->get_value(2);

            return r == b;
        }
    }
    if (DAG_call const *c = as<DAG_call>(n)) {
        if (c->get_semantic() == IDefinition::DS_CONV_CONSTRUCTOR &&
            strcmp(c->get_name(), "color(float)") == 0) {
            return true;
        }
    }
    return false;
}

// Unwrap a float to color cast, i.e. return a node computing a float from
// a node computing a color.
DAG_node const *DAG_node_factory_impl::unwrap_float_to_color(DAG_node const *n)
{
    if (DAG_constant const *c = as<DAG_constant>(n)) {
        if (IValue_rgb_color const *rgb = as<IValue_rgb_color>(c->get_value())) {
            IValue_float const *f = rgb->get_value(0);
            return create_constant(f);
        }
    }
    if (DAG_call const *c = as<DAG_call>(n)) {
        if (c->get_semantic() == IDefinition::DS_CONV_CONSTRUCTOR &&
            strcmp(c->get_name(), "color(float)") == 0) {
            return c->get_argument(0);
        }
    }

    MDL_ASSERT(!"cannot unwrap color to float");
    return NULL;
}

// Create a call.
DAG_node const *DAG_node_factory_impl::create_call(
    char const                    *name,
    IDefinition::Semantics        sema,
    DAG_call::Call_argument const call_args[],
    int                           num_call_args,
    IType const                   *ret_type)
{
    // beware: Annotations are created as calls, then the return type is NULL
    ret_type = ret_type != NULL ? ret_type->skip_type_alias() : ret_type;

    // handle things independent of optimization switches first:
    // - MDL 1.X to MDL 1.Y conversions
    // - scene unit conversion
    // - texture constructors
    switch (sema) {
        case IDefinition::DS_ELEM_CONSTRUCTOR:
            if (num_call_args == 6 && is_material_type(ret_type)) {
                DAG_call::Call_argument n_call_args[7];

                Type_factory &tf = *m_value_factory.get_type_factory();

                MDL_ASSERT(strcmp(
                    name,
                    "material$1.4(bool,material_surface,material_surface,"
                    "color,material_volume,material_geometry)") == 0);

                n_call_args[0] = call_args[0];
                n_call_args[1] = call_args[1];
                n_call_args[2] = call_args[2];
                n_call_args[3] = call_args[3];
                n_call_args[4] = call_args[4];
                n_call_args[5] = call_args[5];

                n_call_args[6].param_name = "hair";
                n_call_args[6].arg        =
                    create_constant(m_value_factory.create_invalid_ref(tf.create_hair_bsdf()));

                // map to material 1.5 constructor
                name = "material(bool,material_surface,material_surface,"
                       "color,material_volume,material_geometry,hair_bsdf)";
                return create_call(
                    name,
                    IDefinition::DS_ELEM_CONSTRUCTOR,
                    n_call_args,
                    7,
                    ret_type);
            }
            break;
    case IDefinition::DS_TEXTURE_CONSTRUCTOR:
        if (m_avoid_non_const_gamma && num_call_args == 2) {
            DAG_node const *url   = call_args[0].arg;
            DAG_node const *gamma = call_args[1].arg;

            if (is<DAG_constant>(url) && !is<DAG_constant>(gamma)) {
                return do_avoid_non_const_gamma(
                    as<IType_texture>(ret_type),
                    cast<DAG_constant>(url),
                    gamma);
            }
        }
        break;
    case IDefinition::DS_INTRINSIC_DF_MEASURED_EDF:
        if (num_call_args == 4) {
            // this transformation will reference state::texture_tangent_u(), state must
            // be imported
            m_needs_state_import = true;

            DAG_call::Call_argument tu_call_args[1];
            DAG_call::Call_argument n_call_args[6];
            IType_factory *type_fact = m_value_factory.get_type_factory();

            // MDL 1.0 -> 1.2: insert the multiplier and tangent_u parameters
            tu_call_args[0].param_name = "index";
            tu_call_args[0].arg        = create_constant(m_value_factory.create_int(0));
            DAG_node const *tu_call =
                create_call(
                    "::state::texture_tangent_u(int)",
                    IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_U,
                    tu_call_args,
                    1,
                    type_fact->create_vector(type_fact->create_float(), 3));

            n_call_args[0]            = call_args[0];
            n_call_args[1].param_name = "multiplier";
            n_call_args[1].arg        = create_constant(m_value_factory.create_float(1.0f));
            n_call_args[2]            = call_args[1];
            n_call_args[3]            = call_args[2];
            n_call_args[4].param_name = "tangent_u";
            n_call_args[4].arg        = tu_call;
            n_call_args[5]            = call_args[3];

            MDL_ASSERT(
                strcmp(name, "::df::measured_edf$1.0(light_profile,bool,float3x3,string)") == 0);
            name = "::df::measured_edf(light_profile,float,bool,float3x3,float3,string)";
            return create_call(name, sema, n_call_args, 6, ret_type);
        } else if (num_call_args == 5) {
            // this transformation will reference state::texture_tangent_u(), state must
            // be imported
            m_needs_state_import = true;

            DAG_call::Call_argument tu_call_args[1];
            DAG_call::Call_argument n_call_args[6];
            IType_factory *type_fact = m_value_factory.get_type_factory();

            // MDL 1.1 -> 1.2: insert tangent_u parameter
            tu_call_args[0].param_name = "index";
            tu_call_args[0].arg        = create_constant(m_value_factory.create_int(0));
            DAG_node const *tu_call =
                create_call(
                "::state::texture_tangent_u(int)",
                IDefinition::DS_INTRINSIC_STATE_TEXTURE_TANGENT_U,
                tu_call_args,
                1,
                type_fact->create_vector(type_fact->create_float(), 3));

            n_call_args[0]            = call_args[0];
            n_call_args[1]            = call_args[1];
            n_call_args[2]            = call_args[2];
            n_call_args[3]            = call_args[3];
            n_call_args[4].param_name = "tangent_u";
            n_call_args[4].arg        = tu_call;
            n_call_args[5]            = call_args[4];

            MDL_ASSERT(
                strcmp(
                    name,
                    "::df::measured_edf$1.1(light_profile,float,bool,float3x3,string)") == 0);
            name = "::df::measured_edf(light_profile,float,bool,float3x3,float3,string)";
            return create_call(name, sema, n_call_args, 6, ret_type);
        }
        break;
    case IDefinition::DS_INTRINSIC_DF_FRESNEL_LAYER:
        if (strcmp(name, "::df::fresnel_layer$1.3(color,float,bsdf,bsdf,float3)") == 0) {
            // MDL 1.3 -> 1.4: convert "half-colored" to full colored
            DAG_call::Call_argument n_call_args[5];

            n_call_args[0] = call_args[0];
            n_call_args[1] = call_args[1];
            n_call_args[2] = call_args[2];
            n_call_args[3] = call_args[3];
            n_call_args[4] = call_args[4];

            // wrap the second parameter by an color constructor
            DAG_call::Call_argument c_call_args[1];

            c_call_args[0].arg        = n_call_args[1].arg;
            c_call_args[0].param_name = "value";

            IType const *color_tp = n_call_args[0].arg->get_type()->skip_type_alias();

            n_call_args[1].arg = create_constructor_call(
                "color(float)", IDefinition::DS_CONV_CONSTRUCTOR, c_call_args, 1, color_tp);

            name = "::df::color_fresnel_layer(color,color,bsdf,bsdf,float3)";
            return create_call(
                name, IDefinition::DS_INTRINSIC_DF_COLOR_FRESNEL_LAYER, n_call_args, 5, ret_type);
        }
        break;
    case IDefinition::DS_INTRINSIC_DF_SPOT_EDF:
        if (num_call_args == 4) {
            // MDL 1.0 -> 1.1: insert spread parameter
            DAG_call::Call_argument n_call_args[5];

            // insert the spread parameter
            n_call_args[0]            = call_args[0];
            n_call_args[1].param_name = "spread";
            n_call_args[1].arg        = create_constant(
                m_value_factory.create_float(float(M_PI)));
            n_call_args[2]            = call_args[1];
            n_call_args[3]            = call_args[2];
            n_call_args[4]            = call_args[3];

            MDL_ASSERT(strcmp(name, "::df::spot_edf$1.0(float,bool,float3x3,string)") == 0);
            name = "::df::spot_edf(float,float,bool,float3x3,string)";
            return create_call(name, sema, n_call_args, 5, ret_type);
        }
        break;
    case IDefinition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF:
        if (num_call_args == 6) {
            // MDL 1.5 -> 1.6: insert multiscatter_tint parameter
            DAG_call::Call_argument n_call_args[7];
            IValue_float const *zero = m_value_factory.create_float(0.0f);

            // insert the multiscatter_tint parameter
            n_call_args[0] = call_args[0];
            n_call_args[1] = call_args[1];
            n_call_args[2] = call_args[2];
            n_call_args[3].param_name = "multiscatter_tint";
            n_call_args[3].arg = create_constant(
                m_value_factory.create_rgb_color(zero, zero, zero));
            n_call_args[4] = call_args[3];
            n_call_args[5] = call_args[4];
            n_call_args[6] = call_args[5];

            MDL_ASSERT(strcmp(name,
                "::df::simple_glossy_bsdf$1.5"
                "(float,float,color,float3,::df::scatter_mode,string)") == 0);
            name =
                "::df::simple_glossy_bsdf"
                "(float,float,color,color,float3,::df::scatter_mode,string)";
            return create_call(name, sema, n_call_args, 7, ret_type);
        }
        break;
    case IDefinition::DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF:
        if (num_call_args == 5) {
            // MDL 1.5 -> 1.6: insert multiscatter_tint parameter
            DAG_call::Call_argument n_call_args[6];
            IValue_float const *zero = m_value_factory.create_float(0.0f);

            // insert the multiscatter_tint parameter
            n_call_args[0] = call_args[0];
            n_call_args[1] = call_args[1];
            n_call_args[2] = call_args[2];
            n_call_args[3].param_name = "multiscatter_tint";
            n_call_args[3].arg = create_constant(
                m_value_factory.create_rgb_color(zero, zero, zero));
            n_call_args[4] = call_args[3];
            n_call_args[5] = call_args[4];

            MDL_ASSERT(strcmp(name,
                "::df::backscattering_glossy_reflection_bsdf$1.5"
                "(float,float,color,float3,string)") == 0);
            name =
                "::df::backscattering_glossy_reflection_bsdf"
                "(float,float,color,color,float3,string)";
            return create_call(name, sema, n_call_args, 6, ret_type);
        }
        break;
    case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF:
        if (num_call_args == 6) {
            // MDL 1.5 -> 1.6: insert multiscatter_tint parameter
            DAG_call::Call_argument n_call_args[7];
            IValue_float const *zero = m_value_factory.create_float(0.0f);

            // insert the multiscatter_tint parameter
            n_call_args[0] = call_args[0];
            n_call_args[1] = call_args[1];
            n_call_args[2] = call_args[2];
            n_call_args[3].param_name = "multiscatter_tint";
            n_call_args[3].arg = create_constant(
                m_value_factory.create_rgb_color(zero, zero, zero));
            n_call_args[4] = call_args[3];
            n_call_args[5] = call_args[4];
            n_call_args[6] = call_args[5];

            MDL_ASSERT(strcmp(name,
                "::df::microfacet_beckmann_smith_bsdf$1.5"
                "(float,float,color,float3,::df::scatter_mode,string)") == 0);
            name =
                "::df::microfacet_beckmann_smith_bsdf"
                "(float,float,color,color,float3,::df::scatter_mode,string)";
            return create_call(name, sema, n_call_args, 7, ret_type);
        }
        break;
    case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF:
        if (num_call_args == 6) {
            // MDL 1.5 -> 1.6: insert multiscatter_tint parameter
            DAG_call::Call_argument n_call_args[7];
            IValue_float const *zero = m_value_factory.create_float(0.0f);

            // insert the multiscatter_tint parameter
            n_call_args[0] = call_args[0];
            n_call_args[1] = call_args[1];
            n_call_args[2] = call_args[2];
            n_call_args[3].param_name = "multiscatter_tint";
            n_call_args[3].arg        = create_constant(
                m_value_factory.create_rgb_color(zero, zero, zero));
            n_call_args[4] = call_args[3];
            n_call_args[5] = call_args[4];
            n_call_args[6] = call_args[5];

            MDL_ASSERT(strcmp(name,
                "::df::microfacet_ggx_smith_bsdf$1.5"
                "(float,float,color,float3,::df::scatter_mode,string)") == 0);
            name =
                "::df::microfacet_ggx_smith_bsdf"
                "(float,float,color,color,float3,::df::scatter_mode,string)";
            return create_call(name, sema, n_call_args, 7, ret_type);
        }
        break;
    case IDefinition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF:
        if (num_call_args == 6) {
            // MDL 1.5 -> 1.6: insert multiscatter_tint parameter
            DAG_call::Call_argument n_call_args[7];
            IValue_float const *zero = m_value_factory.create_float(0.0f);

            // insert the multiscatter_tint parameter
            n_call_args[0] = call_args[0];
            n_call_args[1] = call_args[1];
            n_call_args[2] = call_args[2];
            n_call_args[3].param_name = "multiscatter_tint";
            n_call_args[3].arg = create_constant(
                m_value_factory.create_rgb_color(zero, zero, zero));
            n_call_args[4] = call_args[3];
            n_call_args[5] = call_args[4];
            n_call_args[6] = call_args[5];

            MDL_ASSERT(strcmp(name,
                "::df::microfacet_beckmann_vcavities_bsdf$1.5"
                "(float,float,color,float3,::df::scatter_mode,string)") == 0);
            name =
                "::df::microfacet_beckmann_vcavities_bsdf"
                "(float,float,color,color,float3,::df::scatter_mode,string)";
            return create_call(name, sema, n_call_args, 7, ret_type);
        }
        break;
    case IDefinition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF:
         if (num_call_args == 6) {
             // MDL 1.5 -> 1.6: insert multiscatter_tint parameter
             DAG_call::Call_argument n_call_args[7];
             IValue_float const *zero = m_value_factory.create_float(0.0f);

             // insert the multiscatter_tint parameter
             n_call_args[0] = call_args[0];
             n_call_args[1] = call_args[1];
             n_call_args[2] = call_args[2];
             n_call_args[3].param_name = "multiscatter_tint";
             n_call_args[3].arg        = create_constant(
                 m_value_factory.create_rgb_color(zero, zero, zero));
             n_call_args[4] = call_args[3];
             n_call_args[5] = call_args[4];
             n_call_args[6] = call_args[5];

             MDL_ASSERT(strcmp(name,
                 "::df::microfacet_ggx_vcavities_bsdf$1.5"
                 "(float,float,color,float3,::df::scatter_mode,string)") == 0);
             name =
                 "::df::microfacet_ggx_vcavities_bsdf"
                 "(float,float,color,color,float3,::df::scatter_mode,string)";
             return create_call(name, sema, n_call_args, 7, ret_type);
         }
         break;
    case IDefinition::DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF:
        if (num_call_args == 5) {
            // MDL 1.5 -> 1.6: insert multiscatter_tint parameter
            DAG_call::Call_argument n_call_args[6];
            IValue_float const *zero = m_value_factory.create_float(0.0f);

            // insert the multiscatter_tint parameter
            n_call_args[0] = call_args[0];
            n_call_args[1] = call_args[1];
            n_call_args[2] = call_args[2];
            n_call_args[3].param_name = "multiscatter_tint";
            n_call_args[3].arg = create_constant(
                m_value_factory.create_rgb_color(zero, zero, zero));
            n_call_args[4] = call_args[3];
            n_call_args[5] = call_args[4];

            MDL_ASSERT(strcmp(name,
                "::df::ward_geisler_moroder_bsdf$1.5"
                "(float,float,color,float3,string)") == 0);
            name =
                "::df::ward_geisler_moroder_bsdf"
                "(float,float,color,color,float3,string)";
            return create_call(name, sema, n_call_args, 6, ret_type);
        }
        break;
     case IDefinition::DS_INTRINSIC_STATE_METERS_PER_SCENE_UNIT:
        if (m_enable_scene_conv_fold) {
            return create_constant(
                m_value_factory.create_float(m_mdl_meters_per_scene_unit));
        }
        break;
    case IDefinition::DS_INTRINSIC_STATE_SCENE_UNITS_PER_METER:
        if (m_enable_scene_conv_fold) {
            return create_constant(
                m_value_factory.create_float(1.0f / m_mdl_meters_per_scene_unit));
        }
        break;
    case IDefinition::DS_INTRINSIC_STATE_ROUNDED_CORNER_NORMAL:
        if (num_call_args == 2) {
            DAG_call::Call_argument n_call_args[3];

            // MDL 1.2 -> 1.3: insert the roundness parameter
            n_call_args[0]            = call_args[0];
            n_call_args[1]            = call_args[1];
            n_call_args[2].param_name = "roundness";
            n_call_args[2].arg        = create_constant(m_value_factory.create_float(1.0f));

            MDL_ASSERT(strcmp(name, "::state::rounded_corner_normal$1.2(float,bool)") == 0);
            name = "::state::rounded_corner_normal(float,bool,float)";
            return create_call(name, sema, n_call_args, 3, ret_type);
        }
        break;
    case IDefinition::DS_INTRINSIC_STATE_WAVELENGTH_MIN:
        if (m_enable_wavelength_fold) {
            return create_constant(
                m_value_factory.create_float(m_state_wavelength_min));
        }
        break;
    case IDefinition::DS_INTRINSIC_STATE_WAVELENGTH_MAX:
        if (m_enable_wavelength_fold) {
            return create_constant(
                m_value_factory.create_float(m_state_wavelength_max));
        }
        break;
    case IDefinition::DS_INTRINSIC_TEX_WIDTH:
        if (num_call_args == 1 && is_tex_2d(call_args[0].arg->get_type())) {
            DAG_call::Call_argument n_call_args[2];

            // MDL 1.3 -> 1.4: insert the uv_tile parameter
            n_call_args[0] = call_args[0];
            n_call_args[1].param_name = "uv_tile";
            n_call_args[1].arg        = create_constant(create_int2_zero(m_value_factory));

            MDL_ASSERT(strcmp(name, "::tex::width$1.3(texture_2d)") == 0);
            name = "::tex::width(texture_2d,int2)";
            return create_call(name, sema, n_call_args, 2, ret_type);
        }
        break;
    case IDefinition::DS_INTRINSIC_TEX_HEIGHT:
        if (num_call_args == 1 && is_tex_2d(call_args[0].arg->get_type())) {
            DAG_call::Call_argument n_call_args[2];

            // MDL 1.3 -> 1.4: insert the uv_tile parameter
            n_call_args[0] = call_args[0];
            n_call_args[1].param_name = "uv_tile";
            n_call_args[1].arg        = create_constant(create_int2_zero(m_value_factory));

            MDL_ASSERT(strcmp(name, "::tex::height$1.3(texture_2d)") == 0);
            name = "::tex::height(texture_2d,int2)";
            return create_call(name, sema, n_call_args, 2, ret_type);
        }
        break;
    case IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT:
        if (num_call_args == 2 && is_tex_2d(call_args[0].arg->get_type())) {
            DAG_call::Call_argument n_call_args[3];

            // MDL 1.3 -> 1.4: insert the uv_tile parameter
            n_call_args[0] = call_args[0];
            n_call_args[1] = call_args[1];
            n_call_args[2].param_name = "uv_tile";
            n_call_args[2].arg = create_constant(create_int2_zero(m_value_factory));

            MDL_ASSERT(strcmp(name, "::tex::texel_float$1.3(texture_2d,int2)") == 0);
            name = "::tex::texel_float(texture_2d,int2,int2)";
            return create_call(name, sema, n_call_args, 3, ret_type);
        }
        break;
    case IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT2:
        if (num_call_args == 2 && is_tex_2d(call_args[0].arg->get_type())) {
            DAG_call::Call_argument n_call_args[3];

            // MDL 1.3 -> 1.4: insert the uv_tile parameter
            n_call_args[0] = call_args[0];
            n_call_args[1] = call_args[1];
            n_call_args[2].param_name = "uv_tile";
            n_call_args[2].arg = create_constant(create_int2_zero(m_value_factory));

            MDL_ASSERT(strcmp(name, "::tex::texel_float2$1.3(texture_2d,int2)") == 0);
            name = "::tex::texel_float2(texture_2d,int2,int2)";
            return create_call(name, sema, n_call_args, 3, ret_type);
        }
        break;
    case IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT3:
        if (num_call_args == 2 && is_tex_2d(call_args[0].arg->get_type())) {
            DAG_call::Call_argument n_call_args[3];

            // MDL 1.3 -> 1.4: insert the uv_tile parameter
            n_call_args[0] = call_args[0];
            n_call_args[1] = call_args[1];
            n_call_args[2].param_name = "uv_tile";
            n_call_args[2].arg = create_constant(create_int2_zero(m_value_factory));

            MDL_ASSERT(strcmp(name, "::tex::texel_float3$1.3(texture_2d,int2)") == 0);
            name = "::tex::texel_float3(texture_2d,int2,int2)";
            return create_call(name, sema, n_call_args, 3, ret_type);
        }
        break;
    case IDefinition::DS_INTRINSIC_TEX_TEXEL_FLOAT4:
        if (num_call_args == 2 && is_tex_2d(call_args[0].arg->get_type())) {
            DAG_call::Call_argument n_call_args[3];

            // MDL 1.3 -> 1.4: insert the uv_tile parameter
            n_call_args[0] = call_args[0];
            n_call_args[1] = call_args[1];
            n_call_args[2].param_name = "uv_tile";
            n_call_args[2].arg = create_constant(create_int2_zero(m_value_factory));

            MDL_ASSERT(strcmp(name, "::tex::texel_float4$1.3(texture_2d,int2)") == 0);
            name = "::tex::texel_float4(texture_2d,int2,int2)";
            return create_call(name, sema, n_call_args, 3, ret_type);
        }
        break;
    case IDefinition::DS_INTRINSIC_TEX_TEXEL_COLOR:
        if (num_call_args == 2 && is_tex_2d(call_args[0].arg->get_type())) {
            DAG_call::Call_argument n_call_args[3];

            // MDL 1.3 -> 1.4: insert the uv_tile parameter
            n_call_args[0] = call_args[0];
            n_call_args[1] = call_args[1];
            n_call_args[2].param_name = "uv_tile";
            n_call_args[2].arg = create_constant(create_int2_zero(m_value_factory));

            MDL_ASSERT(strcmp(name, "::tex::texel_color$1.3(texture_2d,int2)") == 0);
            name = "::tex::texel_color(texture_2d,int2,int2)";
            return create_call(name, sema, n_call_args, 3, ret_type);
        }
        break;
    default:
        break;
    }

    if (m_opt_enabled && all_args_without_name(call_args, num_call_args)) {

        if (semantic_is_operator(sema)) {
            IExpression::Operator op = semantic_to_operator(sema);

            if (op == IExpression::OK_ARRAY_INDEX) {
                if (is<DAG_constant>(call_args[1].arg)) {
                    DAG_node const *base = call_args[0].arg;

                    if (is<DAG_call>(base)) {
                        // check if arg[0] is an elemental constructor
                        DAG_call const *a = cast<DAG_call>(base);

                        IDefinition::Semantics sema = a->get_semantic();
                        if (sema == IDefinition::DS_ELEM_CONSTRUCTOR ||
                            sema == IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR)
                        {
                            // T(x_0, ..., x_n, ..., x_m)[n] ==> x_n
                            IType const *ret_type = a->get_type()->skip_type_alias();

                            DAG_constant const *i  = cast<DAG_constant>(call_args[1].arg);
                            IValue_int const   *iv = cast<IValue_int>(i->get_value());
                            int                idx = iv->get_value();

                            switch (ret_type->get_kind()) {
                            case IType::TK_VECTOR:
                            case IType::TK_MATRIX:
                            case IType::TK_ARRAY:
                                if (0 <= idx && idx < a->get_argument_count()) {
                                    return a->get_argument(idx);
                                }
                                // this CAN happen, because MDL does not enforce valid array indexes
                                break;
                            default:
                                // this should NOT happen
                                MDL_ASSERT(!"Unexpected index on non-indexable type");
                                break;
                            }
                        } else if (sema == IDefinition::DS_CONV_CONSTRUCTOR) {
                            // vectorX(v)[n] ==> v, iff type(v) == element_type(vectorX)
                            IType_vector const *v_tp = as<IType_vector>(a->get_type());
                            if (v_tp != NULL && a->get_argument_count() == 1) {
                                DAG_node const *node    = a->get_argument(0);
                                IType const    *node_tp = node->get_type();

                                if (ret_type->skip_type_alias() == node_tp->skip_type_alias()) {
                                    DAG_constant const *i  = cast<DAG_constant>(call_args[1].arg);
                                    IValue_int const   *iv = cast<IValue_int>(i->get_value());
                                    int                idx = iv->get_value();

                                    if (0 <= idx && idx < v_tp->get_size()) {
                                        return node;
                                    }
                                    // this CAN happen, because MDL does not enforce valid
                                    // array indexes
                                }
                            }
                        }
                    } else if (is<DAG_constant>(base)) {
                        DAG_constant const      *i  = cast<DAG_constant>(call_args[1].arg);
                        IValue_int_valued const *iv = cast<IValue_int_valued>(i->get_value());
                        int                     idx = iv->get_value();

                        IValue const *v = cast<DAG_constant>(base)->get_value();
                        v = v->extract(&m_value_factory, idx);
                        if (!is<IValue_bad>(v))
                            return create_constant(v);
                    }
                }
            }
            if (op != IExpression::OK_CALL)
                return create_operator_call(name, op, call_args, ret_type);
        } else if (sema == IDefinition::DS_CONV_OPERATOR || mi::mdl::is_constructor(sema)) {
            return create_constructor_call(name, sema, call_args, num_call_args, ret_type);
        }

        switch (sema) {
        case IDefinition::DS_UNKNOWN:
            // user defined
            break;
        case IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS:
            {
                // T(..., f: x, ...).f ==> x
                DAG_node const *arg = call_args[0].arg;

                while (arg->get_kind() == DAG_node::EK_CALL &&
                    cast<DAG_call>(arg)->get_semantic() ==
                    operator_to_semantic(IExpression_unary::OK_CAST))
                {
                    // simply skip casts, our construction ensures, that the layout is compatible
                    arg = cast<DAG_call>(arg)->get_argument(0);
                }

                if (DAG_call const *call = as<DAG_call>(arg)) {
                    // check if arg is an elemental constructor
                    if (call->get_semantic() == IDefinition::DS_ELEM_CONSTRUCTOR) {
                        // T(..., f: x, ...).f ==> x
                        if (IType_compound const *c_type = as<IType_compound>(call->get_type())) {
                            int field_index = get_field_index(c_type, name);
                            if (DAG_node const *node = call->get_argument(field_index)) {
                                return node;
                            }
                        }
                    }
                } else if (DAG_constant const *c = as<DAG_constant>(arg)) {
                    // T(..., f: x, ...).f == > x
                    IValue const *v = c->get_value();
                    if (IType_compound const *c_type = as<IType_compound>(v->get_type())) {
                        int idx = get_field_index(c_type, name);

                        v = v->extract(&m_value_factory, idx);
                        if (!is<IValue_bad>(v)) {
                            return create_constant(v);
                        }
                    }
                }
            }
            break;
        case IDefinition::DS_INTRINSIC_DAG_ARRAY_LENGTH:
            {
                DAG_node const *arg = call_args[0].arg;

                // skip temporaries
                while (is<DAG_temporary>(arg)) {
                    DAG_temporary const *tmp = cast<DAG_temporary>(arg);
                    arg = tmp->get_expr();
                }

                switch (arg->get_kind()) {
                case DAG_node::EK_CONSTANT:
                    // get the length of an array constant, easy
                    {
                        DAG_constant const *c      = cast<DAG_constant>(arg);
                        IValue const       *val    = c->get_value();
                        IType_array const  *a_type = cast<IType_array>(val->get_type());

                        MDL_ASSERT(
                            a_type->is_immediate_sized() && "array constant size not immediate");
                        int size = a_type->get_size();
                        val = m_value_factory.create_int(size);
                        return create_constant(val);
                    }
                case DAG_node::EK_CALL:
                    // get the length of a call result array
                    {
                        DAG_call const    *call   = cast<DAG_call>(arg);
                        IType_array const *a_type = cast<IType_array>(call->get_type());

                        // at this point, type binding MUST have removed all deferred sized arrays
                        MDL_ASSERT(
                            a_type->is_immediate_sized() && "array constant size not immediate");
                        int size = a_type->get_size();
                        IValue const *val = m_value_factory.create_int(size);
                        return create_constant(val);
                    }
                case DAG_node::EK_PARAMETER:
                    // get the length of a parameter, for class compilation this will be immediate
                    // sized when we reach here during DAG instantiation
                    {
                        DAG_parameter const *param  = cast<DAG_parameter>(arg);
                        IType_array const   *a_type = cast<IType_array>(param->get_type());

                        if (!a_type->is_immediate_sized()) {
                            // not instantiated yet, cannot optimize
                            break;
                        }

                        int size = a_type->get_size();
                        IValue const *val = m_value_factory.create_int(size);
                        return create_constant(val);
                    }
                default:
                    // should not happen
                    MDL_ASSERT(!"Array length argument neither call nor constant nor parameter");
                    break;
                }
            }
            break;

        case IDefinition::DS_INTRINSIC_DF_NORMALIZED_MIX:
        case IDefinition::DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX:
            if (num_call_args == 1) {
                DAG_node const *components = call_args[0].arg;
                char const     *p_name     = call_args[0].param_name;
                bool           final       = false;
                DAG_node const *reduced    = remove_zero_components(components, final);

                if (final)
                    return reduced;
                if (DAG_node const *res = create_mix_call(name, sema, reduced, p_name, ret_type))
                    return res;
            }
            break;
        case IDefinition::DS_INTRINSIC_DF_CLAMPED_MIX:
        case IDefinition::DS_INTRINSIC_DF_COLOR_CLAMPED_MIX:
            if (num_call_args == 1) {
                DAG_node const *components = call_args[0].arg;
                char const     *p_name     = call_args[0].param_name;
                bool           final       = false;
                DAG_node const *reduced    = remove_clamped_components(components, final);

                if (final)
                    return reduced;
                if (DAG_node const *res = create_mix_call(name, sema, reduced, p_name, ret_type))
                    return res;
            }
            break;
        case IDefinition::DS_INTRINSIC_DF_TINT:
            switch (num_call_args) {
            case 2:
                {
                    DAG_node const *tint = call_args[0].arg;
                    if (is<DAG_constant>(tint)) {
                        IValue const *v = cast<DAG_constant>(tint)->get_value();

                        if (is<IValue_rgb_color>(v)) {
                            if (v->is_all_one()) {
                                // df::tint(color(1.0), x) ==> x
                                return call_args[1].arg;
                            }
                            if (v->is_zero()) {
                                // df::tint(color(0.0), x) ==> df()
                                return create_default_df_constructor(cast<IType_df>(ret_type));
                            }
                        }
                    }
                }
                break;
            case 3:
                {
                    DAG_node const *reflection_tint   = call_args[0].arg;
                    DAG_node const *transmission_tint = call_args[1].arg;
                    if (is<DAG_constant>(reflection_tint) && is<DAG_constant>(transmission_tint)) {
                        IValue const *r_v = cast<DAG_constant>(reflection_tint)->get_value();
                        IValue const *t_v = cast<DAG_constant>(transmission_tint)->get_value();

                        if (is<IValue_rgb_color>(r_v) && is<IValue_rgb_color>(t_v)) {
                            if (r_v->is_all_one() && t_v->is_all_one()) {
                                // df::tint(color(1.0), color(1.0), x) ==> x
                                return call_args[2].arg;
                            }
                            if (r_v->is_zero() && t_v->is_zero()) {
                                // df::tint(color(0.0), color(0.0), x) ==> df()
                                return create_default_df_constructor(cast<IType_df>(ret_type));
                            }
                        }
                    }
                }
                break;
            }
            break;
        case IDefinition::DS_INTRINSIC_DF_FRESNEL_LAYER:
            if (num_call_args == 5) {
                DAG_node const *weight = call_args[1].arg;
                if (is<DAG_constant>(weight)) {
                    IValue const *v = cast<DAG_constant>(weight)->get_value();

                    if (is<IValue_float>(v) && v->is_zero()) {
                        // df::fresnel_layer(weight: 0.0, base: x) ==> x
                        DAG_node const *base = call_args[3].arg;
                        return base;
                    }
                }
            }
            break;
        case IDefinition::DS_INTRINSIC_DF_COLOR_FRESNEL_LAYER:
            if (num_call_args == 5) {
                DAG_node const *weight = call_args[1].arg;
                if (is<DAG_constant>(weight)) {
                    IValue const *v = cast<DAG_constant>(weight)->get_value();

                    if (is<IValue_rgb_color>(v) && v->is_zero()) {
                        // df::color_fresnel_layer(weight: color(0.0), base: x) ==> x
                        DAG_node const *base = call_args[3].arg;
                        return base;
                    }
                }
                DAG_node const *ior = call_args[0].arg;
                if (is_color_from_float(ior) && is_color_from_float(weight)) {
                    // df::color_fresnel_layer(ior: color(ior), weight: color(weight), ...) ==>
                    // df::fresnel_layer(ior: ior, weight: weight, ...)
                    DAG_call::Call_argument n_call_args[5];

                    n_call_args[0] = call_args[0];
                    n_call_args[1] = call_args[1];
                    n_call_args[2] = call_args[2];
                    n_call_args[3] = call_args[3];
                    n_call_args[4] = call_args[4];

                    n_call_args[0].arg = unwrap_float_to_color(ior);
                    n_call_args[1].arg = unwrap_float_to_color(weight);

                    name = "::df::fresnel_layer(float,float,bsdf,bsdf,float3)";
                    return create_call(
                        name, IDefinition::DS_INTRINSIC_DF_FRESNEL_LAYER, n_call_args, 5, ret_type);
                }
            }
            break;
        case IDefinition::DS_INTRINSIC_DF_WEIGHTED_LAYER:
            if (num_call_args == 4) {
                DAG_node const *weight = call_args[0].arg;
                if (is<DAG_constant>(weight)) {
                    IValue const *v = cast<DAG_constant>(weight)->get_value();

                    if (is<IValue_float>(v)) {
                        if (v->is_zero()) {
                            // df::weigted_layer(weight: 0.0f, base: x) ==> x
                            DAG_node const *base = call_args[2].arg;
                            return base;
                        }
                        if (v->is_one()) {
                            DAG_node const *normal = call_args[3].arg;
                            if (is_state_normal_call(normal)) {
                                // df::weigted_layer(
                                //      weight: 1.0f, layer: x, normal: state::normal()) ==> x
                                DAG_node const *layer = call_args[1].arg;
                                return layer;
                            }
                        }
                    }
                }
            }
            break;
        case IDefinition::DS_INTRINSIC_DF_COLOR_WEIGHTED_LAYER:
            if (num_call_args == 4) {
                DAG_node const *weight = call_args[0].arg;
                if (is<DAG_constant>(weight)) {
                    IValue const *v = cast<DAG_constant>(weight)->get_value();

                    if (is<IValue_rgb_color>(v)) {
                        if (v->is_zero()) {
                            // df::weigted_layer(weight: color(0.0f), base: x) ==> x
                            DAG_node const *base = call_args[2].arg;
                            return base;
                        }
                        if (v->is_all_one()) {
                            DAG_node const *normal = call_args[3].arg;
                            if (is_state_normal_call(normal)) {
                                // df::weigted_layer(
                                //     weight: color(1.0f), layer: x, normal: state::normal()) ==> x
                                DAG_node const *layer = call_args[1].arg;
                                return layer;
                            }
                        }
                    }
                }
            }
            break;
        case IDefinition::DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER:
            if (num_call_args == 7) {
                DAG_node const *weight = call_args[3].arg;
                if (is<DAG_constant>(weight)) {
                    IValue const *v = cast<DAG_constant>(weight)->get_value();

                    if (is<IValue_float>(v)) {
                        if (v->is_zero()) {
                            // df::custom_curve_layer(weight: 0.0, base: x) ==> x
                            DAG_node const *base = call_args[5].arg;
                            return base;
                        }
                        if (v->is_one()) {
                            DAG_node const *normal = call_args[6].arg;
                            if (is_state_normal_call(normal)) {
                                DAG_node const *exponent = call_args[2].arg;
                                if (is_float_zero(exponent)) {
                                    // df::custom_curve_layer(
                                    //      weight: 1.0,
                                    //      exponent: 0.0,
                                    //      layer: x,
                                    //      normal: state::normal()) ==> x
                                    DAG_node const *layer = call_args[4].arg;
                                    return layer;
                                }

                                DAG_node const *normal_reflectivity  = call_args[0].arg;
                                DAG_node const *grazing_reflectivity = call_args[1].arg;

                                if (is_float_one(normal_reflectivity) &&
                                    is_float_one(grazing_reflectivity))
                                {
                                    // df::custom_curve_layer(
                                    //      weight: 1.0,
                                    //      normal_reflectivity: 1.0,
                                    //      grazing_reflectivity: 1.0,
                                    //      layer: x,
                                    //      normal: state::normal()) ==> x
                                    DAG_node const *layer = call_args[4].arg;
                                    return layer;
                                }
                            }
                        }
                    }
                }
            }
            break;
        case IDefinition::DS_INTRINSIC_DF_COLOR_CUSTOM_CURVE_LAYER:
            if (num_call_args == 7) {
                DAG_node const *weight = call_args[3].arg;
                if (is<DAG_constant>(weight)) {
                    IValue const *v = cast<DAG_constant>(weight)->get_value();

                    if (is<IValue_rgb_color>(v)) {
                        if (v->is_zero()) {
                            // df::custom_curve_layer(weight: color(0.0), base: x) ==> x
                            DAG_node const *base = call_args[5].arg;
                            return base;
                        }
                        if (v->is_all_one()) {
                            DAG_node const *normal = call_args[6].arg;
                            if (is_state_normal_call(normal)) {
                                DAG_node const *exponent = call_args[2].arg;
                                if (is_float_zero(exponent)) {
                                    // df::custom_curve_layer(
                                    //      weight: color(1.0),
                                    //      exponent: 0.0,
                                    //      layer: x,
                                    //      normal: state::normal()) ==> x
                                    DAG_node const *layer = call_args[4].arg;
                                    return layer;
                                }

                                DAG_node const *normal_reflectivity  = call_args[0].arg;
                                DAG_node const *grazing_reflectivity = call_args[1].arg;

                                if (is_float_one(normal_reflectivity) &&
                                    is_float_one(grazing_reflectivity))
                                {
                                    // df::custom_curve_layer(
                                    //      weight: color(1.0),
                                    //      normal_reflectivity: 1.0,
                                    //      grazing_reflectivity: 1.0,
                                    //      layer: x,
                                    //      normal: state::normal()) ==> x
                                    DAG_node const *layer = call_args[4].arg;
                                    return layer;
                                }
                            }
                        }
                    }
                }
            }
            break;
        case IDefinition::DS_INTRINSIC_DF_LIGHT_PROFILE_ISVALID:
        case IDefinition::DS_INTRINSIC_DF_BSDF_MEASUREMENT_ISVALID:
        case IDefinition::DS_INTRINSIC_TEX_TEXTURE_ISVALID:
            if (num_call_args == 1 && is<DAG_constant>(call_args[0].arg)) {
                // Note: we fold here only invalid textures and let others to be folded
                // by the integration. This allows "missing" resources to be flagged as invalid.
                IValue const *r = cast<DAG_constant>(call_args[0].arg)->get_value();
                if (is<IValue_invalid_ref>(r)) {
                    IValue const *v = m_value_factory.create_bool(false);
                    return create_constant(v);
                } else {
                    IValue const *res = evaluate_intrinsic_function(sema, &r, 1);
                    if (res != NULL)
                        return create_constant(res);
                }
            }
            break;
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM:
            if (num_call_args == 2 &&
                is<DAG_constant>(call_args[0].arg) && is<DAG_constant>(call_args[1].arg))
            {
                IValue const *a = cast<DAG_constant>(call_args[0].arg)->get_value();
                IValue const *b = cast<DAG_constant>(call_args[1].arg)->get_value();
                if (equal_coordinate_space(a, b, m_internal_space)) {
                    IValue const *v = create_identity_matrix(m_value_factory);
                    return create_constant(v);
                }
            }
            break;
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_POINT:
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_VECTOR:
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_NORMAL:
        case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_SCALE:
            if (num_call_args == 3 &&
                is<DAG_constant>(call_args[0].arg) && is<DAG_constant>(call_args[1].arg))
            {
                IValue const *a = cast<DAG_constant>(call_args[0].arg)->get_value();
                IValue const *b = cast<DAG_constant>(call_args[1].arg)->get_value();
                
                if (equal_coordinate_space(a, b, m_internal_space)) {
                    return call_args[2].arg;
                }
            }
            break;
        default:
            {
                // known, maybe from math
                VLA<IValue const *> arguments(get_allocator(), num_call_args);

                int i;
                for (i = 0; i < num_call_args; ++i) {
                    DAG_node const *arg = call_args[i].arg;

                    if (!is<DAG_constant>(arg))
                        break;
                    arguments[i] = cast<DAG_constant>(arg)->get_value();
                }

                if (i >= num_call_args) {
                    IValue const *res =
                        evaluate_intrinsic_function(sema, arguments.data(), num_call_args);

                    if (res != NULL)
                        return create_constant(res);
                }
            }
            break;
        }
    }

    DAG_node *res = alloc_call(name, sema, call_args, num_call_args, ret_type);

    return static_cast<Call_impl *>(identify_remember(res));
}

// Create a parameter reference.
DAG_parameter const *DAG_node_factory_impl::create_parameter(
    IType const *type,
    int         index)
{
    DAG_node *res = m_builder.create<Parameter_impl>(m_next_id++, type, index);
    return static_cast<Parameter_impl *>(identify_remember(res));
}

// Get the type factory associated with this expression factory.
Type_factory *DAG_node_factory_impl::get_type_factory()
{
    return m_value_factory.get_type_factory();
}

// Get the value factory associated with this expression factory.
Value_factory *DAG_node_factory_impl::get_value_factory()
{
    return &m_value_factory;
}

// Create a float4x4 identity matrix.
IValue_matrix const *DAG_node_factory_impl::create_identity_matrix(
    IValue_factory &value_factory)
{
    IType_factory *type_factory = value_factory.get_type_factory();
    IType_float const *float_type = type_factory->create_float();
    IType_vector const *float4_type = type_factory->create_vector(float_type, 4);
    IType_matrix const *float4x4_type = type_factory->create_matrix(float4_type, 4);
    IValue const *zero = value_factory.create_float(0.0f);
    IValue const *one = value_factory.create_float(1.0f);
    IValue const *row_0[4] = { one, zero, zero, zero };
    IValue const *row_1[4] = { zero, one, zero, zero };
    IValue const *row_2[4] = { zero, zero, one, zero };
    IValue const *row_3[4] = { zero, zero, zero, one };
    IValue const *columns[4] = {
        value_factory.create_vector(float4_type, row_0, 4),
        value_factory.create_vector(float4_type, row_1, 4),
        value_factory.create_vector(float4_type, row_2, 4),
        value_factory.create_vector(float4_type, row_3, 4)
    };
    return value_factory.create_matrix(float4x4_type, columns, 4);
}

/// Check if the given expression is of matrix type.
static bool is_matrix_typed(IType const *type) {
    if (is_deriv_type(type))
        type = get_deriv_base_type(type);
    return as<IType_matrix>(type) != NULL;
}

/// Check if the given expression is of vector type.
static bool is_vector_typed(IType const *type) {
    if (is_deriv_type(type))
        type = get_deriv_base_type(type);
    return as<IType_vector>(type) != NULL;
}

/// Check if the given expression is of matrix type.
static bool is_matrix_typed(DAG_node const *node) {
    return is_matrix_typed(node->get_type());
}

/// Check if the given expression is of vector type.
static bool is_vector_typed(DAG_node const *node) {
    return is_vector_typed(node->get_type());
}

// Normalize the arguments of a binary expression for better CSE support.
bool DAG_node_factory_impl::normalize(
    IExpression_binary::Operator &op,
    DAG_node const               *&l,
    DAG_node const               *&r)
{
    bool swap_args = false;

    switch (op) {
    case IExpression_binary::OK_MULTIPLY:
        {
            IType const *l_tp = l->get_type()->skip_type_alias();
            IType const *r_tp = r->get_type()->skip_type_alias();

            if (is_matrix_typed(l_tp) || is_matrix_typed(r_tp)) {
                // matrix multiplication is not symmetric
                break;
            }

            // Prefer vector * float regardless of other normalization rules later.
            // The reason is simple: otherwise a vector*float inside a material class could be
            // turned into a float*vector inside a material instance, which might not be available
            // in the Db because it was never referenced ...
            if (is_vector_typed(r_tp) && !is_vector_typed(l_tp)) {
                // swap
                swap_args = true;
                break;
            } else if (is_vector_typed(l_tp) && !is_vector_typed(r_tp)) {
                // do nothing
                break;
            }

            // same for color
            if (is<IType_color>(r_tp) && !is<IType_color>(l_tp)) {
                // swap
                swap_args = true;
                break;
            } else if (is<IType_color>(l_tp) && !is<IType_color>(r_tp)) {
                // do nothing
                break;
            }
        }
        // fall through symmetric cases
    case IExpression_binary::OK_PLUS:
    case IExpression_binary::OK_EQUAL:
    case IExpression_binary::OK_NOT_EQUAL:
    case IExpression_binary::OK_BITWISE_AND:
    case IExpression_binary::OK_BITWISE_XOR:
    case IExpression_binary::OK_BITWISE_OR:
        // symmetric ones
    case IExpression_binary::OK_LOGICAL_AND:
    case IExpression_binary::OK_LOGICAL_OR:
        // && and || are strict inside materials, hence they are symmetric
        {
            if (is<DAG_constant>(l) && !is<DAG_constant>(r)) {
                // put constant to right
                swap_args = true;
            } else if (l->get_id() > r->get_id()) {
                swap_args = true;
            } else {
                // ensure ID's are really unique
                MDL_ASSERT(l == r || l->get_id() != r->get_id());
            }
            break;
        }
    case IExpression_binary::OK_GREATER_OR_EQUAL:
        // normalize a >= b into b <= a
        op = IExpression_binary::OK_LESS_OR_EQUAL;
        swap_args = true;
        break;
    case IExpression_binary::OK_GREATER:
        // normalize a > b into b < a
        op = IExpression_binary::OK_LESS;
        swap_args = true;
        break;
    default:
        break;
    }
    if (swap_args) {
        DAG_node const *t = l;
        l = r;
        r = t;

        return true;
    }
    return false;
}

// Attempt to apply a unary operator to a value.
IValue const *DAG_node_factory_impl::apply_unary_op(
    IValue_factory                    &value_factory,
    IExpression_unary::Operator const op,
    IValue const                      *value)
{
    IValue const *res = NULL;
    switch (op) {
    case IExpression_unary::OK_BITWISE_COMPLEMENT:
        res = value->bitwise_not(&value_factory);
        break;
    case IExpression_unary::OK_LOGICAL_NOT:
        res = value->logical_not(&value_factory);
        break;
    case IExpression_unary::OK_POSITIVE:
        return value;
    case IExpression_unary::OK_NEGATIVE:
        res = value->minus(&value_factory);
        break;
    default:
        return NULL;
    }
    if (!is<IValue_bad>(res))
        return res;
    return NULL;
}

// Attempt to apply a binary operator to two values.
IValue const *DAG_node_factory_impl::apply_binary_op(
    IValue_factory                     &value_factory,
    IExpression_binary::Operator const op,
    IValue const                       *left,
    IValue const                       *right)
{
    IValue const *res = NULL;

    switch (op) {
    case IExpression_binary::OK_MULTIPLY:
        res = left->multiply(&value_factory, right);
        break;
    case IExpression_binary::OK_DIVIDE:
        res = left->divide(&value_factory, right);
        break;
    case IExpression_binary::OK_MODULO:
        res = left->modulo(&value_factory, right);
        break;
    case IExpression_binary::OK_PLUS:
        res = left->add(&value_factory, right);
        break;
    case IExpression_binary::OK_MINUS:
        res = left->sub(&value_factory, right);
        break;
    case IExpression_binary::OK_SHIFT_LEFT:
        res = left->shl(&value_factory, right);
        break;
    case IExpression_binary::OK_SHIFT_RIGHT:
        res = left->asr(&value_factory, right);
        break;
    case IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT:
        res = left->lsr(&value_factory, right);
        break;
    case IExpression_binary::OK_LESS:
        {
            IValue::Compare_results cr = left->compare(right);
            res = value_factory.create_bool(
                cr == IValue::CR_LT);
        }
        break;
    case IExpression_binary::OK_LESS_OR_EQUAL:
        {
            IValue::Compare_results cr = left->compare(right);
            res = value_factory.create_bool(
                (cr & IValue::CR_LE) != 0 && (cr & IValue::CR_UO) == 0);
        }
        break;
    case IExpression_binary::OK_GREATER_OR_EQUAL:
        {
            IValue::Compare_results cr = left->compare(right);
            res = value_factory.create_bool(
                (cr & IValue::CR_GE) != 0 && (cr & IValue::CR_UO) == 0);
        }
        break;
    case IExpression_binary::OK_GREATER:
        {
            IValue::Compare_results cr = left->compare(right);
            res = value_factory.create_bool(
                cr == IValue::CR_GT);
        }
        break;
    case IExpression_binary::OK_EQUAL:
        {
            IValue::Compare_results cr = left->compare(right);
            res = value_factory.create_bool(
                cr == IValue::CR_EQ);
        }
        break;
    case IExpression_binary::OK_NOT_EQUAL:
        {
            IValue::Compare_results cr = left->compare(right);
            res = value_factory.create_bool(
                (cr & IValue::CR_UEQ) == 0);
        }
        break;
    case IExpression_binary::OK_BITWISE_AND:
        res = left->bitwise_and(&value_factory, right);
        break;
    case IExpression_binary::OK_BITWISE_XOR:
        res = left->bitwise_xor(&value_factory, right);
        break;
    case IExpression_binary::OK_BITWISE_OR:
        res = left->bitwise_or(&value_factory, right);
        break;
    case IExpression_binary::OK_LOGICAL_AND:
        res = left->logical_and(&value_factory, right);
        break;
    case IExpression_binary::OK_LOGICAL_OR:
        res = left->logical_or(&value_factory, right);
        break;
    default:
        return NULL;
    }
    if (!is<IValue_bad>(res))
        return res;
    return NULL;
}

// Convert a value to a value of type target_type.
IValue const *DAG_node_factory_impl::convert(
    IValue_factory &value_factory,
    IType const    *target_type,
    IValue const   *value)
{
    IValue const *res = value->convert(&value_factory, target_type);
    if (!is<IValue_bad>(res))
        return res;
    return NULL;
}

/// Check if the given compound type has hidden fields.
static bool have_hidden_fields(IType_compound const *c_type) {
    if (IType_struct const *s_type = as<IType_struct>(c_type)) {
        // currently, only the material emission type has hidden fields
        return s_type->get_predefined_id() == IType_struct::SID_MATERIAL_EMISSION;
    }
    return false;
}

// Evaluate a constructor call.
IValue const *DAG_node_factory_impl::evaluate_constructor(
    IValue_factory         &value_factory,
    IDefinition::Semantics sema,
    IType const            *ret_type,
    Value_vector const     &arguments)
{
    switch (sema) {
    case IDefinition::DS_COPY_CONSTRUCTOR:
        MDL_ASSERT(arguments.size() == 1);
        return arguments[0];
    case IDefinition::DS_CONV_CONSTRUCTOR:
    case IDefinition::DS_CONV_OPERATOR:
        MDL_ASSERT(arguments.size() == 1);
        return convert(value_factory, ret_type, arguments[0]);
    case IDefinition::DS_ELEM_CONSTRUCTOR:
        // an element wise constructor build a value from all its argument values
        if (IType_compound const *c_type = as<IType_compound>(ret_type)) {
            if (IType_struct const *s_type = as<IType_struct>(ret_type)) {
                if (s_type->get_predefined_id() == IType_struct::SID_MATERIAL) {
                    // do not fold the material, later code expects it unfolded
                    break;
                }
            }

            size_t n_fields = c_type->get_compound_size();
            size_t n_args   = arguments.size();

            if (n_fields != n_args && !have_hidden_fields(c_type)) {
                // cannot fold
                MDL_ASSERT(!"unexpected elemental constructor");
                break;
            }

            VLA<IValue const *> values(get_allocator(), n_fields);

            for (size_t i = 0; i < n_args; ++i) {
                values[i] = arguments[i];
            }

            // fill hidden fields by their defaults
            bool failed = false;
            for (size_t i = n_args; i < n_fields; ++i) {
                IType const *f_type = c_type->get_compound_type(int(i));

                if (IType_enum const *e_type = as<IType_enum>(f_type)) {
                    // for enum types, the default is always the first one
                    values[i] = value_factory.create_enum(e_type, 0);
                } else {
                    failed = true;
                    break;
                }
            }

            if (!failed) {
                return value_factory.create_compound(c_type, values.data(), n_fields);
            }
        }
        // cannot fold
        break;
    case IDefinition::DS_COLOR_SPECTRUM_CONSTRUCTOR:
        // a color is constructed from a spectrum
#if 0
        {
            MDL_ASSERT(arguments.size() == 2);

            IValue_array const *wavelengths = cast<IValue_array>(arguments[0]);
            IValue_array const *aplitudes   = cast<IValue_array>(arguments[1]);

            return value_factory.create_spectrum_color(wavelengths, aplitudes);
        }
#else
        // cannot fold
        break;
#endif
    case IDefinition::DS_MATRIX_ELEM_CONSTRUCTOR:
        // a matrix element wise constructor builds a matrix from element values
        {
            IType_matrix const *m_type = cast<IType_matrix>(ret_type);
            IType_vector const *v_type = m_type->get_element_type();

            IValue const *column_vals[4];
            size_t n_cols = m_type->get_columns();
            size_t n_rows = v_type->get_size();

            MDL_ASSERT(arguments.size() == n_cols * n_rows);

            size_t idx = 0;
            for (size_t col = 0; col < n_cols; ++col) {
                IValue const *row_vals[4];
                for (size_t row = 0; row < n_rows; ++row, ++idx) {
                    row_vals[row] = arguments[idx];
                }
                column_vals[col] = value_factory.create_vector(v_type, row_vals, n_rows);
            }
            return value_factory.create_matrix(m_type, column_vals, n_cols);
        }
    case IDefinition::DS_MATRIX_DIAG_CONSTRUCTOR:
        // a matrix diagonal constructor builds a matrix from zeros and
        // its only argument
        MDL_ASSERT(arguments.size() == 1);
        return convert(value_factory, cast<IType_matrix>(ret_type), arguments[0]);
    case IDefinition::DS_INVALID_REF_CONSTRUCTOR:
        // this constructor creates an invalid reference.
        MDL_ASSERT(arguments.size() == 0);
        if (IType_reference const *r_type = as<IType_reference>(ret_type))
            return value_factory.create_invalid_ref(r_type);
        break;
    case IDefinition::DS_DEFAULT_STRUCT_CONSTRUCTOR:
        // This is the default constructor for a struct
        // Because it has no arguments, it was either folded already by the core
        // or is not constant at all, in which case we cannot fold it
        MDL_ASSERT(arguments.size() == 0);
        break;
    case IDefinition::DS_TEXTURE_CONSTRUCTOR:
        // this constructor creates a texture
        {
            MDL_ASSERT(arguments.size() == 2);

            IType_texture const  *tex_type = cast<IType_texture>(ret_type);

            IValue_string const *sval  = cast<IValue_string>(arguments[0]);
            IValue_enum const   *gamma = cast<IValue_enum>(arguments[1]);

            return value_factory.create_texture(
                tex_type,
                sval->get_value(),
                IValue_texture::gamma_mode(gamma->get_value()),
                /*value_tag=*/0,
                /*value_version=*/0);
        }
    case IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR:
        {
            IType_array const *a_type = cast<IType_array>(ret_type->skip_type_alias());
            size_t n_args = arguments.size();
            IValue const * const * args = n_args > 0 ? &arguments[0] : NULL;
            return value_factory.create_array(a_type, args, n_args);
        }
        break;
    default:
        break;
    }
    return NULL;
}

// Evaluate an intrinsic function call.
IValue const *DAG_node_factory_impl::evaluate_intrinsic_function(
    IDefinition::Semantics sema,
    IValue const * const   arguments[],
    size_t                 n_args) const
{
    switch (sema) {
    case IDefinition::DS_UNKNOWN:
        return NULL;
    case IDefinition::DS_INTRINSIC_STATE_TRANSFORM:
        if (n_args == 2 &&
            equal_coordinate_space(arguments[0], arguments[1], m_internal_space))
        {
            return create_identity_matrix(m_value_factory);
        }
        break;
    case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_POINT:
    case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_VECTOR:
    case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_NORMAL:
    case IDefinition::DS_INTRINSIC_STATE_TRANSFORM_SCALE:
        if (n_args == 3 &&
            equal_coordinate_space(arguments[0], arguments[1], m_internal_space))
        {
            return arguments[2];
        }
        break;
    default:
        {
            if (m_call_evaluator != NULL) {
                // try call evaluator first
                IValue const *res = m_call_evaluator->evaluate_intrinsic_function(
                    &m_value_factory, sema, arguments, n_args);
                if (!is<IValue_bad>(res))
                    return res;
            }

            // try compiler evaluator
            IValue const *res = m_mdl->evaluate_intrinsic_function(
                &m_value_factory, sema, arguments, n_args);
            if (!is<IValue_bad>(res))
                return res;
        }
        break;
    }
    return NULL;
}

// Enable the folding of scene unit conversion functions.
void DAG_node_factory_impl::enable_unit_conv_fold(float mdl_meters_per_scene_unit)
{
    m_enable_scene_conv_fold    = true;
    m_mdl_meters_per_scene_unit = mdl_meters_per_scene_unit;
}

// Enable the folding of state::wavelength_[min|max] functions.
void DAG_node_factory_impl::enable_wavelength_fold(
    float wavelength_min,
    float wavelength_max)
{
    m_enable_wavelength_fold = true;
    m_state_wavelength_min   = wavelength_min;
    m_state_wavelength_max   = wavelength_max;
}


// Check if this node factory owns the given DAG node.
bool DAG_node_factory_impl::is_owner(DAG_node const *n) const
{
    return m_builder.get_arena()->contains(n);
}

// Adds a name to a given DAG node.
void DAG_node_factory_impl::add_node_name(
    DAG_node const *node,
    char const     *name)
{
    m_temp_name_map[node] = Arena_strdup(*m_builder.get_arena(), name);
}

// Return true iff all arguments are without name.
bool DAG_node_factory_impl::all_args_without_name(
    DAG_node const *args[],
    size_t         n_args) const
{
    if (!m_expose_names_of_let_expressions) {
        MDL_ASSERT(m_temp_name_map.empty());
        return true;
    }

    for (size_t i = 0; i < n_args; ++i) {
        if (m_temp_name_map.find(args[i]) != m_temp_name_map.end())
            return false;
    }
    return true;
}

// Return true iff all arguments are without name.
bool DAG_node_factory_impl::all_args_without_name(
    DAG_call::Call_argument const args[],
    size_t                        n_args) const
{
    if (!m_expose_names_of_let_expressions) {
        MDL_ASSERT(m_temp_name_map.empty());
        return true;
    }

    for (size_t i = 0; i < n_args; ++i) {
        if (m_temp_name_map.find(args[i].arg) != m_temp_name_map.end())
            return false;
    }
    return true;
}

DAG_node const *DAG_node_factory_impl::shallow_copy(DAG_node const *node)
{
    No_CSE_scope scope(*this);

    switch (node->get_kind())
    {
        case DAG_node::EK_CONSTANT:
        {
            DAG_constant const *c     = cast<DAG_constant>(node);
            IValue const       *value = c->get_value();
            return create_constant(value);
        }
        case DAG_node::EK_PARAMETER:
        {
            DAG_parameter const *p    = cast<DAG_parameter>(node);
            IType const         *type = p->get_type();
            int                 index = p->get_index();
            return create_parameter(type, index);
        }
        case DAG_node::EK_TEMPORARY:
        {
            DAG_temporary const *t    = cast<DAG_temporary>(node);
            DAG_node const      *expr = t->get_expr();
            int                 index = t->get_index();
            return create_temporary(expr, index);
        }
        case DAG_node::EK_CALL:
        {
            DAG_call const         *call    = cast<DAG_call>(node);
            int                    n_params = call->get_argument_count();
            IDefinition::Semantics sema     = call->get_semantic();
            char const             *name    = call->get_name();
            IType const            *type    = call->get_type();

            VLA<DAG_call::Call_argument> args(get_allocator(), n_params);

            for (int i = 0; i < n_params; ++i) {
                args[i].param_name = call->get_parameter_name(i);
                args[i].arg        = call->get_argument(i);
            }

            return create_call(name, sema, args.data(), args.size(), type);
        }
    }

    MDL_ASSERT(!"Unsupported DAG node kind");
    return node;
}

// Create an operator call.
DAG_node const *
DAG_node_factory_impl::create_operator_call(
    char const                    *name,
    IExpression::Operator         op,
    DAG_call::Call_argument const call_args[],
    IType const                   *ret_type)
{
    // Note for all promotes: We do not do them yet if the promoted entity is NOT a constant.
    // iray does not profit from such a transformation, because we would change one unsupported
    // operator call by one unsupported conversion call.
    if (op <= IExpression::OK_UNARY_LAST) {
        // is a unary operator
        IExpression_unary::Operator uop = IExpression_unary::Operator(op);

        DAG_node const *arg = call_args[0].arg;

        if (is<DAG_constant>(arg)) {
            IValue const *a = cast<DAG_constant>(arg)->get_value();

            IValue const *v = apply_unary_op(m_value_factory, uop, a);
            if (v != NULL)
                return create_constant(v);
        }

        // check for idempotent operators
        switch (uop) {
        case IExpression_unary::OK_BITWISE_COMPLEMENT:
        case IExpression_unary::OK_LOGICAL_NOT:
        case IExpression_unary::OK_NEGATIVE:
            // -(-(x)) ==> x
            // !(!(x)) ==> x
            // ~(~(x)) ==> x
            if (is_operator(arg, uop)) {
                return cast<DAG_call>(arg)->get_argument(0);
            }
            break;
        case IExpression_unary::OK_POSITIVE:
            // +x ==> x
            return arg;
        case IExpression_unary::OK_CAST:
            // cast<T>(cast<S>(x) ==> cast<T>(x)
            if (is_operator(arg, uop)) {
                arg = cast<DAG_call>(arg)->get_argument(0);
            }
            break;
        default:
            break;
        }

        DAG_node *res = alloc_call(name, operator_to_semantic(op), call_args, 1, ret_type);

        return static_cast<Call_impl *>(identify_remember(res));
    } else if (op <= IExpression::OK_BINARY_LAST) {
        // binary operator
        IExpression_binary::Operator bop = IExpression_binary::Operator(op);

        DAG_node const *left  = call_args[0].arg;
        DAG_node const *right = call_args[1].arg;

        if (is<DAG_constant>(left) && is<DAG_constant>(right)) {
            IValue const *l = cast<DAG_constant>(left)->get_value();
            IValue const *r = cast<DAG_constant>(right)->get_value();

            IValue const *v = apply_binary_op(m_value_factory, bop, l, r);
            if (v != NULL)
                return create_constant(v);
        }

        bool args_swapped = normalize(bop, left, right);

        // check for arithmetic identities
        switch (bop) {
        case IExpression_binary::OK_LESS_OR_EQUAL:
        case IExpression_binary::OK_GREATER_OR_EQUAL:
        case IExpression_binary::OK_EQUAL:
            if (left == right) {
                if (m_unsafe_math_opt || is_finite(left)) {
                    // x <= x == true
                    // x >= x == true
                    // x == x == true
                    return create_constant(m_value_factory.create_bool(true));
                }
            }
            break;
        case IExpression_binary::OK_LESS:
        case IExpression_binary::OK_GREATER:
            if (left == right) {
                if (m_unsafe_math_opt || is_finite(left)) {
                    // x < x == false
                    // x > x == false
                    return create_constant(m_value_factory.create_bool(false));
                }
            }
            break;
        case IExpression_binary::OK_NOT_EQUAL:
            if (left == right) {
                // do not kill the NaN check: really do this only for finite values
                if (is_finite(left)) {
                    // x != x == false
                    return create_constant(m_value_factory.create_bool(false));
                }
            }
            break;
        case IExpression_binary::OK_BITWISE_AND:
            if (left == right) {
                // x & x ==> x
                return left;
            }
            if (is_zero(right)) {
                // x & 0 ==> 0
                return right;
            }
            if (is_all_one(right)) {
                // x & 1...1 ==> x
                return left;
            }
            break;
        case IExpression_binary::OK_BITWISE_XOR:
            if (left == right) {
                // x ^ x ==> 0
                IValue const *v = m_value_factory.create_zero(ret_type);
                if (!is<IValue_bad>(v))
                    return create_constant(v);
            }
            if (is_zero(right)) {
                // x ^ 0 ==> x
                return left;
            }
            break;
        case IExpression_binary::OK_BITWISE_OR:
            if (left == right) {
                // x | x ==> x
                return left;
            }
            if (is_zero(right)) {
                // x | 0 ==> x
                return left;
            }
            if (is_all_one(right)) {
                // x | 1...1 ==> 1...1
                return right;
            }
            break;
        case IExpression_binary::OK_LOGICAL_AND:
            if (left == right) {
                // x && x == x
                return left;
            }
            if (is_zero(right)) {
                // x && false ==> PROMOTE(false)
                IValue const *v =
                    m_value_factory.create_bool(false)->convert(&m_value_factory, ret_type);
                if (!is<IValue_bad>(v))
                    return create_constant(v);
            }
            if (is_one(right)) {
                // x && true ==> PROMOTE(x)
                if (equal_op_types(ret_type, left->get_type()))
                    return left;
            }
            break;
        case IExpression_binary::OK_LOGICAL_OR:
            if (left == right) {
                // x || x ==> x
                return left;
            }
            if (is_zero(right)) {
                // x || false ==> PROMOTE(x)
                if (equal_op_types(ret_type, left->get_type()))
                    return left;
            }
            if (is_one(right)) {
                // x || true ==> PROMOTE(true)
                IValue const *v =
                    m_value_factory.create_bool(true)->convert(&m_value_factory, ret_type);
                if (!is<IValue_bad>(v))
                    return create_constant(v);
            }
            break;
        case IExpression_binary::OK_SHIFT_LEFT:
        case IExpression_binary::OK_SHIFT_RIGHT:
        case IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT:
            if (is_zero(right)) {
                // x SHIFT 0 ==> x
                return left;
            }
            break;
        case IExpression_binary::OK_PLUS:
        case IExpression_binary::OK_MINUS:
            if (is_zero(right)) {
                // x + 0 ==> PROMOTE(x)
                // x - 0 ==> PROMOTE(x)
                if (equal_op_types(ret_type, left->get_type()))
                    return left;
            }
            break;
        case IExpression_binary::OK_MULTIPLY:
            if (is_one(right)) {
                // x * 1 ==> PROMOTE(x)
                if (is_matrix_typed(left) && is_vector_typed(right)) {
                    // does not work for matrix * vector, vector(1) is NOT the
                    // neutral element
                } else {
                    if (equal_op_types(ret_type, left->get_type()))
                        return left;
                }
            } else if (is_zero(right)) {
                if (m_unsafe_math_opt || is_finite(left)) {
                    // x * 0 ==> PROMOTE(0)
                    IValue const *zero = m_value_factory.create_zero(ret_type);
                    return create_constant(zero);
                }
            }
            if (is_one(left)) { // Matrix mult is not symmetric ...
                // 1 * x ==> PROMOTE(x)
                if (is_vector_typed(left) && is_matrix_typed(right)) {
                    // does not work for vector * matrix, vector(1) is NOT the
                    // neutral element
                } else {
                    if (equal_op_types(ret_type, right->get_type()))
                        return right;
                }
            } else if (is_zero(left)) {
                if (m_unsafe_math_opt || is_finite(right)) {
                    // 0 * x ==> PROMOTE(0)
                    IValue const *zero = m_value_factory.create_zero(ret_type);
                    return create_constant(zero);
                }
            }
            break;
        case IExpression_binary::OK_DIVIDE:
            if (is_one(right)) {
                // x / 1 ==> PROMOTE(x)
                if (equal_op_types(ret_type, left->get_type()))
                    return left;
            }
            break;
        case IExpression_binary::OK_MODULO:
            if (is_one(right)) {
                // modulo is only defined on integer in MDL, so it is safe to do:
                // x % 1 ==> PROMOTE(0)
                IValue const *zero = m_value_factory.create_zero(left->get_type());
                MDL_ASSERT(!is<IValue_bad>(zero));
                return create_constant(zero);
            }
            break;
        default:
            break;
        }

        string op_name(get_allocator());
        if (args_swapped &&
            left->get_type()->skip_type_alias() != right->get_type()->skip_type_alias()) {
            // Fixup the name: must be different due to normalization
            string tmp(name, get_allocator());

            size_t pos = tmp.find('(');
            if (pos != string::npos) {
                size_t comma = tmp.find(',', pos + 1);
                size_t len   = tmp.size();

                op_name = tmp.substr(0, pos + 1) +
                    tmp.substr(comma + 1, len - comma - 2) +
                    ',' +
                    tmp.substr(pos + 1, comma - pos - 1) +
                    ')';
                name = op_name.c_str();
            }
        }

        // argument order might have changed, so create a new a argument vector
        DAG_call::Call_argument new_args[2];
        new_args[0].arg        = left;
        new_args[0].param_name = call_args[0].param_name;
        new_args[1].arg        = right;
        new_args[1].param_name = call_args[1].param_name;

        DAG_node *res = alloc_call(name, operator_to_semantic(op), new_args, 2, ret_type);

        return static_cast<Call_impl *>(identify_remember(res));
    } else {
        MDL_ASSERT(op == IExpression::OK_TERNARY);

        return create_ternary_call(call_args[0].arg, call_args[1].arg, call_args[2].arg, ret_type);
    }
}

// Converts a constant into a elemental constructor.
DAG_call const *DAG_node_factory_impl::value_to_constructor(
    DAG_constant const *c)
{
    IValue_struct const *v = cast<IValue_struct>(c->get_value());

    IType_struct const *type = v->get_type();

    int n_fields = type->get_field_count();
    Small_VLA<DAG_call::Call_argument, 8> args(get_allocator(), n_fields);

    Name_printer printer(get_allocator(), m_mdl.get());

    printer.print(type->get_symbol()->get_name());
    printer.print('(');

    for (int i = 0; i < n_fields; ++i) {
        ISymbol const *f_sym;
        IType const *f_type;

        type->get_field(i, f_type, f_sym);

        args[i].param_name = f_sym->get_name();
        args[i].arg = create_constant(v->get_value(i));

        if (i != 0)
            printer.print(',');

        printer.print(f_type->skip_type_alias());
    }

    printer.print(')');

    Type_factory &tf = *m_value_factory.get_type_factory();

    IType const *res_type = tf.import(type);
    res_type = tf.create_alias(res_type, /*name=*/NULL, IType::MK_UNIFORM);

    No_OPT_scope scope(*this);

    return (DAG_call const *)create_call(
        printer.get_line().c_str(),
        IDefinition::DS_ELEM_CONSTRUCTOR,
        args.data(),
        args.size(),
        type);
}

// Try to move a ternary operator down.
DAG_node const *DAG_node_factory_impl::move_ternary_down(
    DAG_node const *cond,
    DAG_node const *t_expr,
    DAG_node const *f_expr,
    IType const    *ret_type)
{
    DAG_call const *t_call = NULL;
    DAG_call const *f_call = NULL;

    DAG_constant const *t_const = NULL;
    DAG_constant const *f_const = NULL;

    if (is<DAG_constant>(t_expr)) {
        if (is<DAG_constant>(f_expr)) {
            // both are const, convert to calls is possible first
            t_const = cast<DAG_constant>(t_expr);
            f_const = cast<DAG_constant>(f_expr);

            t_call = value_to_constructor(t_const);
            f_call = value_to_constructor(f_const);
        } else {
            // only t_expr must be converted
            t_const = cast<DAG_constant>(t_expr);

            if (!is<DAG_call>(f_expr))
                return NULL;
            f_call = cast<DAG_call>(f_expr);

            if (f_call->get_semantic() != IDefinition::DS_ELEM_CONSTRUCTOR)
                return NULL;

            t_call = value_to_constructor(t_const);
        }
    } else {
        if (is<DAG_constant>(f_expr)) {
            // only f_expr must be converted
            f_const = cast<DAG_constant>(f_expr);

            if (!is<DAG_call>(t_expr))
                return NULL;
            t_call = cast<DAG_call>(t_expr);

            if (t_call->get_semantic() != IDefinition::DS_ELEM_CONSTRUCTOR)
                return NULL;

            f_call = value_to_constructor(f_const);
        } else {
            // both are non-const, easy

            if (!is<DAG_call>(t_expr) || !is<DAG_call>(f_expr))
                return NULL;

            t_call = cast<DAG_call>(t_expr);
            f_call = cast<DAG_call>(f_expr);

            if (strcmp(t_call->get_name(), f_call->get_name()) != 0) {
                // different calls
                return NULL;
            }
        }
    }

    if (t_call != NULL && f_call != NULL) {
        MDL_ASSERT(strcmp(t_call->get_name(), f_call->get_name()) == 0);

        int n_args = t_call->get_argument_count();
        MDL_ASSERT(n_args == f_call->get_argument_count());

        Small_VLA<DAG_call::Call_argument, 8> args(get_allocator(), n_args);

        for (int i = 0; i < n_args; ++i) {
            DAG_node const *t_arg = t_call->get_argument(i);
            DAG_node const *f_arg = f_call->get_argument(i);

            args[i].param_name = t_call->get_parameter_name(i);
            args[i].arg = t_arg == f_arg ?
                t_arg : create_ternary_call(cond, t_arg, f_arg, t_arg->get_type());
        }

        {
            // switch optimization off, so we are sure no one will "rebuild" this call
            No_OPT_scope no_opt(*this);

            return create_call(
                t_call->get_name(),
                t_call->get_semantic(),
                args.data(),
                args.size(),
                t_call->get_type());
        }
    }

    return NULL;
}

// Create a ternary operator call.
DAG_node const *
DAG_node_factory_impl::create_ternary_call(
    DAG_node const *cond,
    DAG_node const *t_expr,
    DAG_node const *f_expr,
    IType const    *ret_type)
{
    if (t_expr == f_expr) {
        // x ? T : T ==> T
        return t_expr;
    }

    if (is<DAG_constant>(cond)) {
        DAG_constant const *c = cast<DAG_constant>(cond);
        IValue_bool const  *b = cast<IValue_bool>(c->get_value());

        if (b->get_value()) {
            // true case
            return t_expr;
        } else {
            // false case
            return f_expr;
        }
    }

    if (is_material_type_or_sub_type(ret_type)) {
        // we cannot switch over material subtypes, try to move the ternary operator down
        DAG_node const *res = move_ternary_down(cond, t_expr, f_expr, ret_type);

        if (res != NULL)
            return res;

        // failed
    }

    // really bad, an operator? that cannot be resolved

    IAllocator *alloc = get_allocator();

    IType const *type = t_expr->get_type()->skip_type_alias();

    string name("", alloc);

    // special handling for internal derivative types
    if (is_deriv_type(type)) {
        // prefix name with '#' and use base type for the rest
        name += '#';
        type = get_deriv_base_type(type);
    }
    name += get_ternary_operator_signature();

    DAG_call::Call_argument args[3];

    args[0].arg = cond;
    args[0].param_name = "cond";
    args[1].arg = t_expr;
    args[1].param_name = "true_exp";
    args[2].arg = f_expr;
    args[2].param_name = "false_exp";

    DAG_node *res = alloc_call(
        name.c_str(),
        operator_to_semantic(IExpression::OK_TERNARY),
        args,
        dimension_of(args),
        ret_type);

    return static_cast<Call_impl *>(identify_remember(res));
}

/// Get the name of a conversion constructor if there is one.
static char const *conv_constructor_name(IType const *tp)
{
    tp = tp->skip_type_alias();
    if (is<IType_color>(tp)) {
        return "color(float)";
    }
    if (IType_vector const *v_tp = as<IType_vector>(tp)) {
        IType_atomic const *e_tp = v_tp->get_element_type();
        int                n     = v_tp->get_size();

        switch (e_tp->get_kind()) {
        case IType::TK_BOOL:
            switch (n) {
            case 2: return "bool2(bool)";
            case 3: return "bool3(bool)";
            case 4: return "bool4(bool)";
            }
            break;
        case IType::TK_INT:
            switch (n) {
            case 2: return "int2(int)";
            case 3: return "int3(int)";
            case 4: return "int4(int)";
            }
            break;
        case IType::TK_FLOAT:
            switch (n) {
            case 2: return "float2(float)";
            case 3: return "float3(float)";
            case 4: return "float4(float)";
            }
            break;
        case IType::TK_DOUBLE:
            switch (n) {
            case 2: return "double2(double)";
            case 3: return "double3(double)";
            case 4: return "double4(double)";
            }
            break;
        default:
            break;
        }
    }
    return NULL;
}

// Create a constructor call.
DAG_node const *
DAG_node_factory_impl::create_constructor_call(
    char const                    *name,
    IDefinition::Semantics        sema,
    DAG_call::Call_argument const call_args[],
    int                           num_call_args,
    IType const                   *ret_type)
{
    if (sema == IDefinition::DS_COPY_CONSTRUCTOR) {
        // Remove copy constructor. This will not remove any names, so no check is needed here.
        return call_args[0].arg;
    }

    Value_vector values(num_call_args, NULL, get_allocator());

    bool all_args_const = true;
    for (int i = 0; i < num_call_args; ++i) {
        DAG_node const *arg = call_args[i].arg;
        if (is<DAG_constant>(arg)) {
            values[i] = cast<DAG_constant>(arg)->get_value();
        } else {
            all_args_const = false;
            break;
        }
    }

    if (all_args_const && all_args_without_name(call_args, num_call_args)) {
        if (IValue const *v = evaluate_constructor(
                m_value_factory, sema, ret_type, values)) {
            return create_constant(v);
        }
    }

    // handle MDL 1.0 to MDL 1.1 conversions
    DAG_node *res = NULL;
    if (sema == IDefinition::DS_ELEM_CONSTRUCTOR) {
        if (IType_struct const *s_type = as<IType_struct>(ret_type)) {
            if (s_type->get_predefined_id() == IType_struct::SID_MATERIAL_EMISSION) {
                if (num_call_args == 2) {
                    // convert the MDL 1.0 constructor into an MDL 1.1 constructor
                    DAG_call::Call_argument new_call_args[3];

                    IType_enum const *e_type =
                        m_value_factory.get_type_factory()->
                        get_predefined_enum(IType_enum::EID_INTENSITY_MODE);
                    IValue const *v = m_value_factory.create_enum(e_type, 0);

                    new_call_args[0] = call_args[0];
                    new_call_args[1] = call_args[1];
                    new_call_args[2].param_name = "mode";
                    new_call_args[2].arg = create_constant(v);

                    MDL_ASSERT(strcmp(name, "material_emission$1.0(edf,color)") == 0);
                    name = "material_emission(edf,color,intensity_mode)";
                    res = alloc_call(name, sema, new_call_args, 3, ret_type);
                }
            }
        }
    }

    if (m_opt_enabled && res == NULL) {
        if (strcmp(name, "color(float3)") == 0) {
            // color(float3(x)) ==> color(x)
            DAG_node const *f3 = call_args[0].arg;

            if (DAG_call const *c = as<DAG_call>(f3)) {
                if (strcmp(c->get_name(), "float3(float)") == 0) {
                    DAG_node const *x = c->get_argument(0);

                    DAG_call::Call_argument n_call_args[1];

                    n_call_args[0].arg = x;
                    n_call_args[0].param_name = "value";

                    res = alloc_call(
                        "color(float)",
                        IDefinition::DS_CONV_CONSTRUCTOR,
                        n_call_args, 1,
                        ret_type);
                }
            }
        } else if (sema == IDefinition::DS_ELEM_CONSTRUCTOR) {
            DAG_node const *x;
            bool           all_same = false;

            if (is<IType_vector>(ret_type->skip_type_alias())) {
                // vectorX(x,...,x)) ==> vectorX(x)
                x = call_args[0].arg;
                all_same = true;
                for (int i = 1; i < num_call_args; ++i) {
                    if (x != call_args[i].arg) {
                        all_same = false;
                        break;
                    }
                }

                // check for vectorX(a.x, a.y, a.z, ...) ==> a
                if (!all_same) {
                    if (DAG_call const *xc = as<DAG_call>(x)) {
                        if (xc->get_semantic() ==
                            operator_to_semantic(IExpression::OK_ARRAY_INDEX))
                        {
                            x = xc->get_argument(0);

                            if (x->get_type()->skip_type_alias()== ret_type->skip_type_alias()) {
                                unsigned num_fields = 0;
                                for (int i = 0; i < num_call_args; ++i) {
                                    DAG_call const *arg = as<DAG_call>(call_args[i].arg);
                                    if (arg == NULL)
                                        break;
                                    if (arg->get_semantic() !=
                                        operator_to_semantic(IExpression::OK_ARRAY_INDEX))
                                        break;
                                    if (arg->get_argument(0) != x)
                                        break;

                                    DAG_constant const *c = as<DAG_constant>(arg->get_argument(1));
                                    if (c == NULL)
                                        break;
                                    IValue_int const *iv = cast<IValue_int>(c->get_value());
                                    if (iv->get_value() != i)
                                        break;
                                    ++num_fields;
                                }

                                if (num_fields == num_call_args) {
                                    // all elements are used
                                    return x;
                                }
                            }
                        }
                    }
                }
            } else if (strcmp(name, "color(float,float,float)") == 0) {
                // color(x,x,x) ==> color(x)
                x = call_args[0].arg;
                if (x == call_args[1].arg && x == call_args[2].arg) {
                    all_same = true;
                }
            }

            if (all_same) {
                DAG_call::Call_argument n_call_args[1];

                n_call_args[0].arg = x;
                n_call_args[0].param_name = "value";

                res = alloc_call(
                    conv_constructor_name(ret_type),
                    IDefinition::DS_CONV_CONSTRUCTOR,
                    n_call_args, 1,
                    ret_type);
            }
        }
    }

    if (res == NULL)
        res = alloc_call(name, sema, call_args, num_call_args, ret_type);

    return static_cast<Call_impl *>(identify_remember(res));
}

// Creates an invalid reference (i.e. a call to the a default df constructor).
DAG_node const *DAG_node_factory_impl::create_default_df_constructor(
    IType_df const *df_type)
{
    IValue const *invalid_ref = m_value_factory.create_invalid_ref(df_type);
    return create_constant(invalid_ref);
}

// Remove zero weighted components for df::normalized_mix() and color_normalized_mix().
DAG_node const *DAG_node_factory_impl::remove_zero_components(
    DAG_node const *components,
    bool           &is_finale_result)
{
    is_finale_result = false;
    if (is<DAG_call>(components)) {
        DAG_call const *c = cast<DAG_call>(components);

        if (c->get_semantic() != IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR)
            return components;

        int n_args = c->get_argument_count();
        VLA<DAG_call::Call_argument> f_args(get_allocator(), n_args);

        size_t j = 0;
        for (int i = 0; i < n_args; ++i) {
            DAG_node const *arg = c->get_argument(i);

            f_args[j++].arg      = arg;
            f_args[i].param_name = c->get_parameter_name(i);

            if (is<DAG_call>(arg)) {
                DAG_call const *c_arg = cast<DAG_call>(arg);

                if (c_arg->get_semantic() == IDefinition::DS_ELEM_CONSTRUCTOR) {
                    DAG_node const *w = c_arg->get_argument("weight");
                    if (w == NULL)
                        continue;

                    if (is<DAG_constant>(w)) {
                        DAG_constant const *wc = cast<DAG_constant>(w);
                        IValue const       *v  = wc->get_value();

                        switch (v->get_kind()) {
                        case IValue::VK_FLOAT:
                            {
                                IValue_float const *wv = cast<IValue_float>(v);

                                if (wv->get_value() <= 0.0f) {
                                    // a weight smaller or equal zero can be removed
                                    --j;
                                }
                            }
                            break;

                        case IValue::VK_RGB_COLOR:
                            {
                                IValue_rgb_color const *wv = cast<IValue_rgb_color>(v);

                                if (wv->get_value(0)->get_value() <= 0.0f &&
                                    wv->get_value(1)->get_value() <= 0.0f &&
                                    wv->get_value(2)->get_value() <= 0.0f)
                                {
                                    // a weight smaller or equal zero can be removed
                                    --j;
                                }
                            }
                            break;

                        default:
                            continue;
                        }
                    }
                }
            }
        }

        if (j < n_args) {
            // something was deleted
            if (j == 0) {
                // all components are deleted
                return NULL;
            } else if (j == 1) {
                DAG_node const *arg = f_args[0].arg;

                if (is<DAG_call>(arg)) {
                    DAG_call const *c_arg = cast<DAG_call>(arg);

                    if (c_arg->get_semantic() == IDefinition::DS_ELEM_CONSTRUCTOR) {
                        if (DAG_constant const *wc =
                               as<DAG_constant>(c_arg->get_argument("weight")))
                        {
                            if (IValue_float const *wv = as<IValue_float>(wc->get_value())) {
                                if (wv->is_one()) {
                                    // only ONE element with weight 1.0f
                                    DAG_node const *res = c_arg->get_argument("component");

                                    if (res != NULL) {
                                        is_finale_result = true;
                                        return res;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // else reduced
            IType_factory *tf = m_value_factory.get_type_factory();
            IType_array const *a_type = cast<IType_array>(c->get_type());
            IType const       *e_type = a_type->get_element_type();
            IType const       *r_type = tf->create_array(e_type, j);

            return create_call(
                c->get_name(),
                IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR,
                f_args.data(),
                j,
                r_type);
        }
    }
    return components;
}

// Remove clamped components for df::clamped_mix() and df::color_clamped_mix().
DAG_node const *DAG_node_factory_impl::remove_clamped_components(
    DAG_node const *components,
    bool           &is_finale_result)
{
    is_finale_result = false;
    if (is<DAG_call>(components)) {
        DAG_call const *c = cast<DAG_call>(components);

        if (c->get_semantic() != IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR)
            return components;

        int n_args = c->get_argument_count();
        VLA<DAG_call::Call_argument> f_args(get_allocator(), n_args);

        size_t j = 0;
        float sum_mono = 0.0f, sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
        bool known_sum = true;
        for (int i = 0; i < n_args; ++i) {
            DAG_node const *arg = c->get_argument(i);

            f_args[j++].arg      = arg;
            f_args[i].param_name = c->get_parameter_name(i);

            if (is<DAG_call>(arg)) {
                DAG_call const *c_arg = cast<DAG_call>(arg);

                if (c_arg->get_semantic() == IDefinition::DS_ELEM_CONSTRUCTOR) {
                    DAG_node const *w = c_arg->get_argument("weight");
                    if (w == NULL) {
                        // this really should not happen
                        return components;
                    }

                    if (is<DAG_constant>(w)) {
                        DAG_constant const *wc = cast<DAG_constant>(w);
                        IValue const       *v = wc->get_value();

                        switch (v->get_kind()) {
                        case IValue::VK_RGB_COLOR:
                            {
                                IValue_rgb_color const *wv = cast<IValue_rgb_color>(v);

                                float weight_r = wv->get_value(0)->get_value();
                                float weight_g = wv->get_value(1)->get_value();
                                float weight_b = wv->get_value(2)->get_value();
                                if (weight_r <= 0.0f && weight_g <= 0.0f && weight_b <= 0.0f) {
                                    // a weight smaller or equal zero can be removed
                                    --j;
                                    continue;
                                }
                                sum_r += weight_r;
                                sum_g += weight_g;
                                sum_b += weight_b;
                                if ((weight_r >= 1.0f && weight_g >= 1.0f && weight_b >= 1.0f) ||
                                    (known_sum && sum_r >= 1.0f && sum_g >= 1.0f && sum_b >= 1.0f))
                                {
                                    // reached clamping
                                    break;
                                }
                            }
                            break;

                        case IValue::VK_FLOAT:
                            {
                                IValue_float const *wv = cast<IValue_float>(v);

                                float weight = wv->get_value();
                                if (weight <= 0.0f) {
                                    // a weight smaller or equal zero can be removed
                                    --j;
                                    continue;
                                }
                                sum_mono += weight;
                                if (weight >= 1.0 || (known_sum && sum_mono >= 1.0f)) {
                                    // reached clamping
                                    break;
                                }
                            }
                            break;

                        default:
                            // this really should not happen
                            return components;
                        }
                    } else {
                        // one weight is NOT a constant
                        known_sum = false;
                    }
                }
            }
        }

        if (j < n_args) {
            // something was deleted
            if (j == 0) {
                // all components are deleted
                return NULL;
            } else if (j == 1) {
                DAG_node const *arg = f_args[0].arg;

                if (is<DAG_call>(arg)) {
                    DAG_call const *c_arg = cast<DAG_call>(arg);

                    if (c_arg->get_semantic() == IDefinition::DS_ELEM_CONSTRUCTOR) {
                        if (DAG_constant const *wc =
                            as<DAG_constant>(c_arg->get_argument("weight")))
                        {
                            if (IValue_float const *wv = as<IValue_float>(wc->get_value())) {
                                if (wv->get_value() >= 1.0f) {
                                    // only ONE element with weight >= 1.0 (which
                                    // will clamped to 1.0)
                                    DAG_node const *res = c_arg->get_argument("component");

                                    if (res != NULL) {
                                        is_finale_result = true;
                                        return res;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // else reduced
            IType_factory *tf = m_value_factory.get_type_factory();
            IType_array const *a_type = cast<IType_array>(c->get_type());
            IType const       *e_type = a_type->get_element_type();
            IType const       *r_type = tf->create_array(e_type, j);

            return create_call(
                c->get_name(),
                IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR,
                f_args.data(),
                j,
                r_type);
        }
    }
    return components;
}

// Returns node or an identical IR node.
DAG_node *DAG_node_factory_impl::identify_remember(
    DAG_node *node)
{
    if (!m_cse_enabled)
        return node;

    Value_table::iterator it = m_value_table.find(node);
    if (it == m_value_table.end()) {
        m_value_table.insert(node);
        return node;
    }
    // already known, drop this and return the other
    size_t id = node->get_id();
    if (id + 1 == m_next_id) {
        // recover the ID
        --m_next_id;
    }
    m_builder.get_arena()->drop(node);
    return *it;
}

// Create a df::*_mix() call.
DAG_node const *DAG_node_factory_impl::create_mix_call(
    char const             *name,
    IDefinition::Semantics sema,
    DAG_node const         *call_arg,
    char const             *param_name,
    IType const            *ret_type)
{
    if (call_arg != NULL) {
        DAG_call::Call_argument n_arg[1];

        n_arg[0].arg        = call_arg;
        n_arg[0].param_name = param_name;

        DAG_node *res = alloc_call(name, sema, n_arg, 1, ret_type);

        return static_cast<Call_impl *>(identify_remember(res));
    } else {
        // transform into invalid ref constructor
        IType_df const *df_type = cast<IType_df>(ret_type);
        return create_default_df_constructor(df_type);
    }
    return NULL;
}

// Allocate a Call node.
Call_impl *DAG_node_factory_impl::alloc_call(
    char const                    *name,
    IDefinition::Semantics        sema,
    DAG_call::Call_argument const call_args[],
    size_t                        num_call_args,
    IType const                   *ret_type)
{
    ISymbol const *shared = m_sym_tab.get_shared_symbol(name);
    return m_builder.create<Call_impl>(
        m_next_id++,
        m_builder.get_arena(),
        shared->get_name(),
        sema,
        call_args,
        num_call_args,
        ret_type);
}

// Set the index of an parameter.
void set_parameter_index(DAG_parameter *param, Uint32 param_idx)
{
    Parameter_impl *p = static_cast<Parameter_impl *>(param);
    p->set_index(param_idx);
}

} // mdl
} // mi

