/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "mdl/compiler/compilercore/compilercore_memory_arena.h"
#include "mdl/compiler/compilercore/compilercore_array_ref.h"

#include "compiler_hlsl_assert.h"
#include "compiler_hlsl_declarations.h"
#include "compiler_hlsl_definitions.h"
#include "compiler_hlsl_exprs.h"
#include "compiler_hlsl_types.h"
#include "compiler_hlsl_values.h"

namespace mi {
namespace mdl {
namespace hlsl {

// ------------------------------- Expr -------------------------------

// Constructor.
Expr::Expr(
    Location const &loc,
    Type           *type)
: Base()
, m_type(type)
, m_loc(loc)
, m_in_parenthesis(false)
{
}

// Get the type.
Type *Expr::get_type() const
{
    return m_type;
}

// Set the type of this expression.
void Expr::set_type(Type *type)
{
    m_type = type;
}

// Fold this expression into a constant value if possible.
Value *Expr::fold(Value_factory &factory) const
{
    return factory.get_bad();
}

// ------------------------------- Expr_invalid -------------------------------

/// Constructor.
Expr_invalid::Expr_invalid(
    Location const &loc,
    Type           *type)
: Base(loc, type)
{
    HLSL_ASSERT(is<Type_error>(type));
}

// Get the kind of expression.
Expr::Kind Expr_invalid::get_kind() const
{
    return s_kind;
}

// ------------------------------- Expr_literal -------------------------------

// Constructor.
Expr_literal::Expr_literal(
    Location const &loc,
    Value          *value)
: Base(loc, value->get_type())
, m_value(value)
{
}

// Get the kind of expression.
Expr::Kind Expr_literal::get_kind() const
{
    return s_kind;
}

// Fold this expression into a constant value if possible.
Value *Expr_literal::fold(Value_factory &) const
{
    return m_value;
}

// ------------------------------- Expr_ref -------------------------------

// Constructor.
Expr_ref::Expr_ref(
    Type_name *name)
: Base(name->get_location(), name->get_type())
, m_name(name)
, m_def(NULL)
{
}

// Get the kind of expression.
Expr::Kind Expr_ref::get_kind() const
{
    return s_kind;
}

// ------------------------------- Expr_unary -------------------------------

// Constructor.
Expr_unary::Expr_unary(
    Location const &loc,
    Operator       op,
    Expr           *arg)
: Base(loc, arg->get_type())
, m_arg(arg)
, m_op(op)
{
}

// Get the kind of expression.
Expr::Kind Expr_unary::get_kind() const
{
    return s_kind;
}

// Get the type.
Type *Expr_unary::get_type() const
{
    // the type of an unary expression is always the type of its argument
    return m_arg->get_type();
}

// Set the type of this expression.
void Expr_unary::set_type(Type *type)
{
    // the type of an unary expression is always the type of its argument
    m_arg->set_type(type);
}

// Fold this expression into a constant value if possible.
Value *Expr_unary::fold(
    Value_factory &factory) const
{
    Value *value = m_arg->fold(factory);
    if (!is<Value_bad>(value)) {
        switch (m_op) {
        case OK_BITWISE_COMPLEMENT:
            return value->bitwise_not(factory);
        case OK_LOGICAL_NOT:
            return value->logical_not(factory);
        case OK_POSITIVE:
            return value;
        case OK_NEGATIVE:
            return value->minus(factory);
        case OK_PRE_INCREMENT:
        case OK_PRE_DECREMENT:
        case OK_POST_INCREMENT:
        case OK_POST_DECREMENT:
            // these cannot operate on constants
            HLSL_ASSERT(!"inplace op on constant");
            break;
        case OK_POINTER_DEREF:
            // cannot fold
            break;
        }
    }
    return Base::fold(factory);
}

// ------------------------------- Expr_binary -------------------------------

// Constructor.
Expr_binary::Expr_binary(
    Operator op,
    Expr     *lhs,
    Expr     *rhs)
: Base(lhs->get_location(), lhs->get_type())  // FIXME
, m_lhs(lhs)
, m_rhs(rhs)
, m_op(op)
{
}

// Get the kind of expression.
Expr::Kind Expr_binary::get_kind() const
{
    return s_kind;
}

// Fold this binary expression into a constant value if possible.
Value *Expr_binary::fold(
    Value_factory &factory) const
{
    Value *lhs = m_lhs->fold(factory);
    if (is<Value_bad>(lhs))
        return lhs;

#if 0
    if (m_op == OK_SELECT) {
        if (Expr_ref *ref = as<Expr_ref>(m_rhs)) {
            if (IDefinition const *def = ref->get_definition()) {
                if (def->get_kind() == IDefinition::DK_MEMBER) {
                    size_t index = def->get_field_index();
                    return lhs->extract(factory, index);
                }
            }
        }
    }
#endif

    Value *rhs = m_rhs->fold(factory);
    if (is<Value_bad>(rhs))
        return rhs;

    switch (m_op) {
    case OK_SELECT:
        // handled above
        break;
    case OK_ARROW:
        // cannot fold
        break;
    case OK_ARRAY_SUBSCRIPT:
        if (Value_compound *a_value = as<Value_compound>(lhs)) {
            size_t index   = 0;
            bool  can_fold = false;

            if (Value_two_complement<32> *i_value = as<Value_two_complement<32> >(rhs)) {
                index = i_value->get_value_unsigned();
                can_fold = true;
            } else if (Value_two_complement<12> *i_value = as<Value_two_complement<12> >(rhs)) {
                index = i_value->get_value_unsigned();
                can_fold = true;
            } else if (Value_two_complement<16> *i_value = as<Value_two_complement<16> >(rhs)) {
                index = i_value->get_value_unsigned();
                can_fold = true;
            }

            if (can_fold) {
                if (index >= a_value->get_component_count()) {
                    // out of bounds
                    return factory.get_bad();
                } else {
                    return a_value->extract(factory, index);
                }
            }
        }
        // cannot fold
        break;
    case OK_MULTIPLY:
        return lhs->multiply(factory, rhs);
    case OK_DIVIDE:
        return lhs->divide(factory, rhs);
    case OK_MODULO:
        return lhs->modulo(factory, rhs);
    case OK_PLUS:
        return lhs->add(factory, rhs);
    case OK_MINUS:
        return lhs->sub(factory, rhs);
    case OK_SHIFT_LEFT:
        return lhs->shl(factory, rhs);
    case OK_SHIFT_RIGHT:
        return lhs->shr(factory, rhs);
    case OK_LESS:
        {
            unsigned res = lhs->compare(rhs);
            return factory.get_bool(res == Value::CR_LT);
        }
    case OK_LESS_OR_EQUAL:
        {
            unsigned res = lhs->compare(rhs);
            return factory.get_bool(
                (res & Value::CR_LE) != 0 &&
                (res & Value::CR_UO) == 0);
        }
    case OK_GREATER_OR_EQUAL:
        {
            unsigned res = lhs->compare(rhs);
            return factory.get_bool(
                (res & Value::CR_GE) != 0 &&
                (res & Value::CR_UO) == 0);
        }
    case OK_GREATER:
        {
            unsigned res = lhs->compare(rhs);
            return factory.get_bool(res == Value::CR_GT);
        }
    case OK_EQUAL:
        {
            unsigned res = lhs->compare(rhs);
            return factory.get_bool(res == Value::CR_EQ);
        }
    case OK_NOT_EQUAL:
        {
            unsigned res = lhs->compare(rhs);
            return factory.get_bool((res & Value::CR_UEQ) == 0);
        }
    case OK_BITWISE_AND:
        return lhs->bitwise_and(factory, rhs);
    case OK_BITWISE_OR:
        return lhs->bitwise_or(factory, rhs);
    case OK_BITWISE_XOR:
        return lhs->bitwise_xor(factory, rhs);
    case OK_LOGICAL_AND:
        return lhs->logical_and(factory, rhs);
    case OK_LOGICAL_OR:
        return lhs->logical_or(factory, rhs);
    case OK_LOGICAL_XOR:
        return lhs->logical_xor(factory, rhs);
    case OK_ASSIGN:
    case OK_MULTIPLY_ASSIGN:
    case OK_DIVIDE_ASSIGN:
    case OK_MODULO_ASSIGN:
    case OK_PLUS_ASSIGN:
    case OK_MINUS_ASSIGN:
    case OK_SHIFT_LEFT_ASSIGN:
    case OK_SHIFT_RIGHT_ASSIGN:
    case OK_BITWISE_AND_ASSIGN:
    case OK_BITWISE_XOR_ASSIGN:
    case OK_BITWISE_OR_ASSIGN:
    case OK_SEQUENCE:
        // cannot fold
        break;
    }
    return Base::fold(factory);
}

// ------------------------------- Expr_condition -------------------------------

/// Get the common type for ?:.
static Type *common_type(Type *a, Type *b)
{
    if (a == b)
        return a;
    if (a == NULL || b == NULL)
        return NULL;
    if (is<Type_error>(a))
        return a;
    if (is<Type_error>(b))
        return b;
#if 0
    Type::Modifiers a_mod = a->get_type_modifiers();
    Type::Modifiers b_mod = a->get_type_modifiers();

    if ((a_mod & Type::MK_CONST) != 0)
        return b;
    if ((b_mod & Type::MK_CONST) != 0)
        return a;

    if ((a_mod & Type::MK_UNIFORM) != 0)
        return b;
    if ((b_mod & Type::MK_UNIFORM) != 0)
        return a;

    // one must be varying here
    if ((a_mod & Type::MK_VARYING) != 0)
        return a;
    HLSL_ASSERT((b_mod & Type::MK_UNIFORM) != 0);
#endif
    return b;
}

// Constructor.
Expr_conditional::Expr_conditional(
    Expr *cond,
    Expr *true_expr,
    Expr *false_expr)
: Base(cond->get_location(), common_type(true_expr->get_type(), false_expr->get_type()))
, m_cond(cond)
, m_true_expr(true_expr)
, m_false_expr(false_expr)
{
}

// Get the kind of expression.
Expr::Kind Expr_conditional::get_kind() const
{
    return s_kind;
}

// Fold this binary expression into a constant value if possible.
Value *Expr_conditional::fold(
    Value_factory &factory) const
{
    Value *cond = m_cond->fold(factory);
    if (Value_bool *b = as<Value_bool>(cond)) {
        if (b->get_value())
            return m_true_expr->fold(factory);
        else
            return m_false_expr->fold(factory);
    }
    Value *true_val = m_true_expr->fold(factory);
    if (!is<Value_bad>(true_val)) {
        Value *false_val = m_false_expr->fold(factory);

        if (true_val == false_val)
            return true_val;
    }
    return Base::fold(factory);
}


// ------------------------------- Expr_call -------------------------------

/// Get the return type from an callee.
static Type *type_from_callee(Expr *callee)
{
    if (Type *tp = callee->get_type()) {
        if (Type_function *f_tp = as<Type_function>(tp)) {
            return f_tp->get_return_type();
        }
    }
    return NULL;
}

// Constructor.
Expr_call::Expr_call(
    Memory_arena            *arena,
    Expr                    *callee,
    Array_ref<Expr *> const &args)
: Base(callee->get_location(), type_from_callee(callee))
, m_callee(callee)
, m_args(args.size(), (Expr *)NULL, arena)
, m_is_typecast(false)
{
    for (size_t i = 0, n = args.size(); i < n; ++i)
        m_args[i] = args[i];
}

// Get the kind of expression.
Expr::Kind Expr_call::get_kind() const
{
    return s_kind;
}

// Get the argument at index.
Expr *Expr_call::get_argument(size_t index)
{
    if (index < m_args.size())
        return m_args[index];
    HLSL_ASSERT(!"index out of range");
    return NULL;
}

// Get the argument at index.
Expr const *Expr_call::get_argument(size_t index) const
{
    Expr_call *c = const_cast<Expr_call *>(this);
    return c->get_argument(index);
}

// Set an argument at index.
void Expr_call::set_argument(size_t index, Expr *arg)
{
    HLSL_ASSERT(index < m_args.size() && "index out of range");

    if (index < m_args.size())
        m_args[index] = arg;
}

// Mark this call as a typecast.
void Expr_call::set_typecast(bool flag)
{
    HLSL_ASSERT(m_args.size() == 1 && "Typecasts must have exactly one argument");
    m_is_typecast = flag;
}

// Fold this call expression into a constant value if possible.
Value *Expr_call::fold(Value_factory &factory) const
{
#if 0
    if (Expr_ref *ref = as<Expr_ref(m_callee)) {
        if (ref->is_array_constructor()) {
            // array constructor: fold all of its arguments
            IType_array const *a_type = as<IType_array>(get_type());
            int n = a_type->get_size();

            Module const *mod = impl_cast<Module>(module);
            IAllocator *alloc = mod->get_allocator();

            VLA<Value const *> values(alloc, n);

            if (n == get_argument_count()) {
                for (int i = 0; i < n; ++i) {
                    IArgument const   *arg  = get_argument(i);
                    IExpression const *expr = arg->get_argument_exp();
                    Value const      *val  = expr->fold(module, handler);
                    if (is<Value_bad>(val))
                        return val;
                    values[i] = val;
                }
            } else {
                // array default constructor
                Value const *def_val = mod->create_default_value(a_type->get_element_type());
                for (int i = 0; i < n; ++i) {
                    values[i] = def_val;
                }
            }
            return factory.get_array(a_type, values, n);
        }

        if (IDefinition const *def = ref->get_definition()) {
            switch (def->get_semantics()) {
            case IDefinition::DS_UNKNOWN:
                // cannot fold;
                break;
            case IDefinition::DS_COPY_CONSTRUCTOR:
                // a copy constructor copies its only argument
                {
                    if (ref->is_array_constructor()) {
                        Module const *mod = impl_cast<Module>(module);
                        IAllocator *alloc = mod->get_allocator();

                        int n = get_argument_count();
                        VLA<Value const *> values(alloc, n);

                        for (int i = 0; i < n; ++i) {
                            IArgument const   *arg  = get_argument(i);
                            IExpression const *expr = arg->get_argument_exp();
                            Value const      *val  = expr->fold(module, handler);

                            if (is<Value_bad>(val))
                                return val;
                            values[i] = val;
                        }
                        IType_array const *a_type = cast<IType_array>(get_type());
                        return factory.get_array(a_type, values, n);
                    } else {
                        HLSL_ASSERT(get_argument_count() == 1);
                        IExpression const *expr = get_argument(0)->get_argument_exp();
                        return expr->fold(module, handler);
                    }
                }
            case IDefinition::DS_CONV_CONSTRUCTOR:
            case IDefinition::DS_CONV_OPERATOR:
                // a conversion constructor/operator converts its only argument
                {
                    IType_function const *func_type = cast<IType_function>(def->get_type());
                    IType const          *dst_type  = func_type->get_return_type();

                    IExpression const *expr = get_argument(0)->get_argument_exp();
                    Value const      *val  = expr->fold(module, handler);
                    return val->convert(factory, dst_type);
                }
            case IDefinition::DS_ELEM_CONSTRUCTOR:
                // an element wise constructor build a value from all its argument values
                {
                    Module const *mod = impl_cast<Module>(module);
                    IAllocator *alloc = mod->get_allocator();

                    IType_function const *func_type = cast<IType_function>(def->get_type());
                    IType const          *dst_type  = func_type->get_return_type();

                    if (IType_compound const *c_type = as<IType_compound>(dst_type)) {
                        int n_fields = c_type->get_compound_size();
                        int n_args   = get_argument_count();

                        HLSL_ASSERT(n_fields == n_args || have_hidden_fields(c_type));

                        VLA<Value const *> values(alloc, n_fields);

                        for (int i = 0; i < n_args; ++i) {
                            IArgument const   *arg  = get_argument(i);
                            IExpression const *expr = arg->get_argument_exp();
                            Value const      *val  = expr->fold(module, handler);
                            if (is<Value_bad>(val))
                                return val;
                            values[i] = val;
                        }

                        // fill hidden fields by their defaults
                        bool failed = false;
                        for (int i = n_args; i < n_fields; ++i) {
                            IType const *f_type = c_type->get_compound_type(i);

                            if (IType_enum const *e_type = as<IType_enum>(f_type)) {
                                // for enum types, the default is always the first one
                                values[i] =
                                    module->get_value_factory()->create_enum(e_type, 0);
                            } else {
                                failed = true;
                                break;
                            }
                        }

                        if (!failed)
                            return factory.get_compound(c_type, values, n_fields);
                    }
                    // cannot fold
                    break;
                }
            case IDefinition::DS_COLOR_SPECTRUM_CONSTRUCTOR:
                // a color is constructed from a spectrum
                {
                    Value const *wavelengths, *aplitudes;

                    IArgument const   *arg0  = get_argument(0);
                    IExpression const *expr0 = arg0->get_argument_exp();
                    wavelengths = expr0->fold(module, handler);
                    if (is<Value_bad>(wavelengths))
                        return wavelengths;

                    IArgument const   *arg1  = get_argument(1);
                    IExpression const *expr1 = arg1->get_argument_exp();
                    aplitudes = expr1->fold(module, handler);
                    if (is<Value_bad>(aplitudes))
                        return aplitudes;

                    return factory.get_color(
                        cast<Value_array>(wavelengths), cast<Value_array>(aplitudes));
                }
            case IDefinition::DS_MATRIX_ELEM_CONSTRUCTOR:
                // a matrix element wise constructor builds a matrix from element values
                {

                    IType_function const *func_type = cast<IType_function>(def->get_type());
                    IType_matrix const   *m_type    =
                        cast<IType_matrix>(func_type->get_return_type());
                    IType_vector const   *v_type    = m_type->get_element_type();

                    Value const *column_vals[4];
                    size_t n_cols = m_type->get_columns();
                    size_t n_rows = v_type->get_size();

                    int idx = 0;
                    for (size_t col = 0; col < n_cols; ++col) {
                        Value const *row_vals[4];
                        for (size_t row = 0; row < n_rows; ++row, ++idx) {
                            IArgument const   *arg  = get_argument(idx);
                            IExpression const *expr = arg->get_argument_exp();
                            Value const      *val  = expr->fold(module, handler);
                            if (is<Value_bad>(val))
                                return val;
                            row_vals[row] = val;
                        }
                        column_vals[col] = factory.get_vector(v_type, row_vals, n_rows);
                    }
                    return factory.get_matrix(m_type, column_vals, n_cols);
                }
                break;
            case IDefinition::DS_MATRIX_DIAG_CONSTRUCTOR:
                // a matrix diagonal constructor builds a matrix from zeros and
                // its only argument
                {
                    IType_function const *func_type = cast<IType_function>(def->get_type());
                    IType_matrix const   *m_type    =
                        cast<IType_matrix>(func_type->get_return_type());
                    IType_vector const   *v_type    = m_type->get_element_type();
                    IType_atomic const   *a_type    = v_type->get_element_type();

                    IExpression const *expr = get_argument(0)->get_argument_exp();
                    Value const *val = expr->fold(module, handler);
                    Value const *zero;

                    if (a_type->get_kind() == IType::TK_FLOAT) {
                        zero = factory.get_float(0.0f);
                    } else {
                        zero = factory.get_double(0.0);
                    }

                    Value const *column_vals[4];
                    size_t n_cols = m_type->get_columns();
                    size_t n_rows = v_type->get_size();

                    for (size_t col = 0; col < n_cols; ++col) {
                        Value const *row_vals[4];
                        for (size_t row = 0; row < n_rows; ++row) {
                            row_vals[row] = col == row ? val : zero;
                        }
                        column_vals[col] = factory.get_vector(v_type, row_vals, n_rows);
                    }
                    return factory.get_matrix(m_type, column_vals, n_cols);
                }
            case IDefinition::DS_INVALID_REF_CONSTRUCTOR:
                // this constructor creates an invalid reference.
                {
                    IType_function const *func_type = cast<IType_function>(def->get_type());
                    IType const          *dst_type  = func_type->get_return_type();

                    if (IType_reference const *r_type = as<IType_reference>(dst_type))
                        return factory.get_invalid_ref(r_type);
                    break;
                }
            case IDefinition::DS_DEFAULT_STRUCT_CONSTRUCTOR:
                // this is the default constructor for a struct
                {
                    Definition const *c_def = impl_cast<Definition>(def);
                        
                    if (c_def->has_flag(Definition::DEF_IS_CONST_CONSTRUCTOR)) {
                        IType_function const *func_type = cast<IType_function>(def->get_type());
                        IType const          *dst_type  = func_type->get_return_type();
                        Module const         *mod       = impl_cast<Module>(module);

                        return mod->create_default_value(dst_type);
                    }
                    break;
                }
            case IDefinition::DS_TEXTURE_CONSTRUCTOR:
                // this constructor creates a texture
                {
                    IType_function const *ftype    = cast<IType_function>(def->get_type());
                    IType_texture const  *tex_type =
                        cast<IType_texture>(ftype->get_return_type());

                    IExpression const *expr0 = get_argument(0)->get_argument_exp();
                    Value const      *val0  = expr0->fold(module, handler);

                    IExpression const *expr1 = get_argument(1)->get_argument_exp();
                    Value const      *val1  = expr1->fold(module, handler);

                    if (is<Value_string>(val0) && is<Value_enum>(val1)) {
                        Value_string const *sval  = cast<Value_string>(val0);
                        Value_enum const   *gamma = cast<Value_enum>(val1);
                        return factory.get_texture(
                            tex_type,
                            sval->get_value(),
                            Value_texture::gamma_mode(gamma->get_value()),
                            /*tag_value=*/0,
                            /*tag_version=*/0);
                    }
                }

            default:
                // unsupported
                break;
            }
        }
    }
#endif
    return Base::fold(factory);
}

// -------------------------------------- Expr_compound --------------------------------------

// Constructor.
Expr_compound::Expr_compound(
    Memory_arena   *arena,
    Location const &loc)
: Base(loc, NULL)
, m_elems(arena)
{
}

// Get the kind of expression.
Expr::Kind Expr_compound::get_kind() const
{
    return s_kind;
}

// Fold this call expression into a constant value if possible.
Value *Expr_compound::fold(Value_factory &factory) const
{
    // FIXME: NYI
    return Base::fold(factory);
}

// Get the element at index.
Expr *Expr_compound::get_element(size_t index)
{
    if (index < m_elems.size())
        return m_elems[index];
    HLSL_ASSERT(!"index out of range");
    return NULL;
}

// Get the element at index.
Expr const *Expr_compound::get_element(size_t index) const
{
    Expr_compound *c = const_cast<Expr_compound *>(this);
    return c->get_element(index);
}

// Add an element
void Expr_compound::add_element(Expr *elem)
{
    HLSL_ASSERT(elem != NULL);
    m_elems.push_back(elem);
}

// Set an element at index.
void Expr_compound::set_element(size_t index, Expr *elem)
{
    HLSL_ASSERT(index < m_elems.size() && "index out of range");
    HLSL_ASSERT(elem != NULL);

    if (index < m_elems.size())
        m_elems[index] = elem;
}

// -------------------------------------- Expr_factory --------------------------------------

// Constructs a new expression factory.
Expr_factory::Expr_factory(
    Memory_arena  &arena,
    Value_factory &vf)
: Base()
, m_builder(arena)
, m_tf(vf.get_type_factory())
, m_vf(vf)
{
}

// Create a new invalid expression.
Expr_invalid *Expr_factory::create_invalid(Location const &loc)
{
    return m_builder.create<Expr_invalid>(loc, m_tf.get_error());
}

// Create a new literal expression.
Expr_literal *Expr_factory::create_literal(
    Location const &loc,
    Value          *value)
{
    return m_builder.create<Expr_literal>(loc, value);
}

// Create a new reference expression.
Expr *Expr_factory::create_reference(
    Type_name *name)
{
    MDL_ASSERT(name != NULL);

    return m_builder.create<Expr_ref>(name);
}

// Create a new unary expression.
Expr *Expr_factory::create_unary(
    Location const       &loc,
    Expr_unary::Operator op,
    Expr                 *arg)
{
    HLSL_ASSERT(arg != NULL);
    if (op == Expr_unary::OK_POSITIVE)
        return arg;
    if (op == Expr_unary::OK_LOGICAL_NOT) {
        if (Expr_unary *unary = as<Expr_unary>(arg)) {
            if (unary->get_operator() == Expr_unary::OK_LOGICAL_NOT) {
                return unary->get_argument();
            }
        }
        if (Expr_binary *bin = as<Expr_binary>(arg)) {
            Type *left_type = bin->get_left_argument()->get_type();
            if (left_type != NULL && is<Type_int>(left_type)) {
                Expr_binary::Operator new_op = Expr_binary::OK_SEQUENCE;
                switch (bin->get_operator()) {
                case Expr_binary::OK_LESS:
                    new_op = Expr_binary::OK_GREATER_OR_EQUAL;
                    break;
                case Expr_binary::OK_LESS_OR_EQUAL:
                    new_op = Expr_binary::OK_GREATER;
                    break;
                case Expr_binary::OK_EQUAL:
                    new_op = Expr_binary::OK_NOT_EQUAL;
                    break;
                case Expr_binary::OK_NOT_EQUAL:
                    new_op = Expr_binary::OK_EQUAL;
                    break;
                case Expr_binary::OK_GREATER:
                    new_op = Expr_binary::OK_LESS_OR_EQUAL;
                    break;
                case Expr_binary::OK_GREATER_OR_EQUAL:
                    new_op = Expr_binary::OK_LESS;
                    break;
                default:
                    break;
                }
                if (new_op != Expr_binary::OK_SEQUENCE) {
                    return create_binary(
                        new_op, bin->get_left_argument(), bin->get_right_argument());
                }
            }
        }
    }

    Expr_unary *res = m_builder.create<Expr_unary>(loc, op, arg);
    Value *val = res->fold(m_vf);
    if (!is<Value_bad>(val))
        return create_literal(loc, val);
    return res;
}

// Create a new binary expression.
Expr *Expr_factory::create_binary(
    Expr_binary::Operator op,
    Expr                  *left,
    Expr                  *right)
{
    HLSL_ASSERT(left != NULL && right != NULL);
    HLSL_ASSERT(
        op != Expr_binary::OK_SELECT ||
        (right->get_kind()== Expr::EK_INVALID || right->get_kind()== Expr::EK_REFERENCE));
    Expr_binary *res = m_builder.create<Expr_binary>(op, left, right);
    Value *val = res->fold(m_vf);
    if (!is<Value_bad>(val))
        return create_literal(res->get_location(), val);
    return res;
}

// Create a new conditional expression.
Expr *Expr_factory::create_conditional(
    Expr           *cond,
    Expr           *true_expr,
    Expr           *false_expr)
{
    HLSL_ASSERT(cond != NULL && true_expr != NULL && false_expr != NULL);
    Expr  *res      = NULL;
    Value *cond_val = cond->fold(m_vf);

    if (Value_bool *bcond = as<Value_bool>(cond_val)) {
        res = bcond->get_value() ? true_expr : false_expr;
    } else {
        res = m_builder.create<Expr_conditional>(cond, true_expr, false_expr);
    }
    Value *val = res->fold(m_vf);
    if (!is<Value_bad>(val))
        return create_literal(res->get_location(), val);
    return res;
}

// Create a new call expression.
Expr *Expr_factory::create_call(
    Expr                    *callee,
    Array_ref<Expr *> const &args)
{
    HLSL_ASSERT(callee != NULL);
    Expr_call *res = m_builder.create<Expr_call>(m_builder.get_arena(), callee, args);
    Value *val = res->fold(m_vf);
    if (!is<Value_bad>(val))
        return create_literal(res->get_location(), val);
    return res;
}

// Create a new typecast call expression.
Expr *Expr_factory::create_typecast(
    Expr *callee,
    Expr *arg)
{
    HLSL_ASSERT(callee != NULL);
    Expr_call *res = m_builder.create<Expr_call>(m_builder.get_arena(), callee, arg);
    res->set_typecast(true);

    Value *val = res->fold(m_vf);
    if (!is<Value_bad>(val))
        return create_literal(res->get_location(), val);
    return res;
}

// Create a new (empty) compound initializer expression.
Expr *Expr_factory::create_compound(
    Location const          &loc,
    Array_ref<Expr *> const &args)
{
    Expr_compound *res = m_builder.create<Expr_compound>(m_builder.get_arena(), loc);

    for (Expr *arg : args) {
        res->add_element(arg);
    }
    return res;
}

}  // hlsl
}  // mdl
}  // mi

