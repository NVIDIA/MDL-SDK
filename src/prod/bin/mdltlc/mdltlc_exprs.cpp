/******************************************************************************
 * Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

#include <mdl/compiler/compilercore/compilercore_memory_arena.h>
#include <mdl/compiler/compilercore/compilercore_array_ref.h>

#include "mdltlc_exprs.h"
#include "mdltlc_types.h"
#include "mdltlc_values.h"
#include "mdltlc_compilation_unit.h"

static char const * unary_token(Expr_unary::Operator op) {
    char const *name = "<BUG>";
    switch (op) {
    case Expr_unary::Operator::OK_BITWISE_COMPLEMENT:
        name = "~";
        break;
    case Expr_unary::Operator::OK_LOGICAL_NOT:
        name = "!";
        break;
    case Expr_unary::Operator::OK_POSITIVE:
        name = "+";
        break;
    case Expr_unary::Operator::OK_NEGATIVE:
        name = "-";
        break;
    case Expr_unary::Operator::OK_PRE_INCREMENT:
        name = "++";
        break;
    case Expr_unary::Operator::OK_PRE_DECREMENT:
        name = "--";
        break;
    case Expr_unary::Operator::OK_POST_INCREMENT:
        name = "++";
        break;
    case Expr_unary::Operator::OK_POST_DECREMENT:
        name = "--";
        break;
    case Expr_unary::Operator::OK_OPTION:
        name = "option";
        break;
    case Expr_unary::Operator::OK_NONODE:
        name = "nonode";
        break;
    case Expr_unary::Operator::OK_MATCH:
        name = "match";
        break;
    case Expr_unary::Operator::OK_IF_GUARD:
        name = "if";
        break;
    case Expr_unary::Operator::OK_MAYBE_GUARD:
        name = "maybe";
        break;
    }
    return name;
}

static char const * binary_token(Expr_binary::Operator op) {
    char const *name = "<BUG>";
    switch (op) {
    case Expr_binary::Operator::OK_SELECT:
        name =".";
        break;
    case Expr_binary::Operator::OK_ARRAY_SUBSCRIPT:
        /* Not allowed, but to return something that might help in
         * debugging... */
        name = "<BUG with []>";
        break;
    case Expr_binary::Operator::OK_MULTIPLY:
        name = "*";
        break;
    case Expr_binary::Operator::OK_DIVIDE:
        name = "/";
        break;
    case Expr_binary::Operator::OK_MODULO:
        name = "%";
        break;
    case Expr_binary::Operator::OK_PLUS:
        name = "+";
        break;
    case Expr_binary::Operator::OK_MINUS:
        name = "-";
        break;
    case Expr_binary::Operator::OK_SHIFT_LEFT:
        name = "<<";
        break;
    case Expr_binary::Operator::OK_SHIFT_RIGHT:
        name = ">>";
        break;
    case Expr_binary::Operator::OK_SHIFT_RIGHT_ARITH:
        name = ">>>";
        break;
    case Expr_binary::Operator::OK_LESS:
        name = "<";
        break;
    case Expr_binary::Operator::OK_LESS_OR_EQUAL:
        name = "<=";
        break;
    case Expr_binary::Operator::OK_GREATER_OR_EQUAL:
        name = ">=";
        break;
    case Expr_binary::Operator::OK_GREATER:
        name = ">";
        break;
    case Expr_binary::Operator::OK_EQUAL:
        name = "==";
        break;
    case Expr_binary::Operator::OK_NOT_EQUAL:
        name = "!=";
        break;
    case Expr_binary::Operator::OK_BITWISE_AND:
        name = "&";
        break;
    case Expr_binary::Operator::OK_BITWISE_OR:
        name = "|";
        break;
    case Expr_binary::Operator::OK_BITWISE_XOR:
        name = "^";
        break;
    case Expr_binary::Operator::OK_LOGICAL_AND:
        name = "&&";
        break;
    case Expr_binary::Operator::OK_LOGICAL_OR:
        name = "||";
        break;
    case Expr_binary::Operator::OK_ASSIGN:
        name = "=";
        break;
    case Expr_binary::Operator::OK_TILDE:
        name = "~";
        break;
    }
    return name;
}

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

void Expr::maybe_pp_type(pp::Pretty_print &p) const {
    if ((p.flags() & pp::Pretty_print::Flags::PRINT_TYPES) == 0)
        return;
    p.colon();
    get_type()->pp(p);
}

// ------------------------------- Expr_invalid -------------------------------

/// Constructor.
Expr_invalid::Expr_invalid(
    Location const &loc,
    Type           *type)
  : Base(loc, type)
{
    MDL_ASSERT(is<Type_error>(type));
}

// Get the kind of expression.
Expr::Kind Expr_invalid::get_kind() const
{
    return s_kind;
}

// Get the priority of the expression (for pretty-printing).
int Expr_invalid::get_prio() const
{
    return 0;
}

// Get the number of sub-expressions.
size_t Expr_invalid::get_sub_expression_count() const
{
    return 0;
}

// Get the i'th subexpression.
Expr *Expr_invalid::get_sub_expression(size_t i)
{
    return NULL;
}

// Get the i'th subexpression.
Expr const *Expr_invalid::get_sub_expression(size_t i) const
{
    return NULL;
}

void Expr_invalid::pp(pp::Pretty_print &p) const {
    p.string("<invalid>");
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

// Get the priority of the expression (for pretty-printing).
int Expr_literal::get_prio() const
{
    return 0;
}

// Get the number of sub-expressions.
size_t Expr_literal::get_sub_expression_count() const
{
    return 0;
}

// Get the i'th subexpression.
Expr *Expr_literal::get_sub_expression(size_t i)
{
    return NULL;
}

// Get the i'th subexpression.
Expr const *Expr_literal::get_sub_expression(size_t i) const
{
    return NULL;
}

void Expr_literal::pp(pp::Pretty_print &p) const {
    switch (m_value->get_kind()) {
    case Value::VK_BOOL: {
        Value_bool *s = as<Value_bool>(m_value);
        if (s->get_value()) {
            p.string("true");
        } else {
            p.string("false");
        }
        break;
    }
    case Value::VK_INT: {
        Value_int *s = as<Value_int>(m_value);
        p.integer(s->get_value());

        break;
    }
    case Value::VK_FLOAT: {
        Value_float *s = as<Value_float>(m_value);
        p.floating_point(s->get_value());
        break;
    }
    case Value::VK_STRING: {
        Value_string *s = as<Value_string>(m_value);
        p.string(s->get_value());
        break;
    }
    }
}

// ------------------------------- Expr_ref -------------------------------

// Constructor.
Expr_ref::Expr_ref(
    Location const &loc,
    Type *type,
    Symbol *symbol)
  : Base(loc, type)
  , m_symbol(symbol)
{
}

// Get the kind of expression.
Expr::Kind Expr_ref::get_kind() const
{
    return s_kind;
}

// Get the priority of the expression (for pretty-printing).
int Expr_ref::get_prio() const
{
    return 0;
}

// Get the number of sub-expressions.
size_t Expr_ref::get_sub_expression_count() const
{
    return 0;
}

// Get the i'th subexpression.
Expr *Expr_ref::get_sub_expression(size_t i)
{
    return NULL;
}

// Get the i'th subexpression.
Expr const *Expr_ref::get_sub_expression(size_t i) const
{
    return NULL;
}

Symbol const* Expr_ref::get_name() {
    return m_symbol;
}

Symbol const* Expr_ref::get_name() const {
    return m_symbol;
}

void Expr_ref::pp(pp::Pretty_print &p) const {
    p.string(m_symbol->get_name());
}

// ------------------------------- Expr_type_annotation -------------------------------

// Constructor.
Expr_type_annotation::Expr_type_annotation(
    Location const &loc,
    Type *type,
    Expr *arg,
    Symbol *type_name)
    : Base(loc, type)
    , m_arg(arg)
    , m_type_name(type_name)
{
}

// Get the kind of expression.
Expr::Kind Expr_type_annotation::get_kind() const
{
    return s_kind;
}

// Get the priority of the expression (for pretty-printing).
int Expr_type_annotation::get_prio() const
{
    return 0;
}

// Get the number of sub-expressions.
size_t Expr_type_annotation::get_sub_expression_count() const
{
    return 1;
}

// Get the i'th subexpression.
Expr *Expr_type_annotation::get_sub_expression(size_t i)
{
    return i == 0 ? m_arg : NULL;
}

// Get the i'th subexpression.
Expr const *Expr_type_annotation::get_sub_expression(size_t i) const
{
    return i == 0 ? m_arg : NULL;
}

void Expr_type_annotation::pp(pp::Pretty_print &p) const {
    m_arg->pp(p);
    p.string("@");
    m_type->pp(p);
}

// ------------------------------- Expr_unary -------------------------------

// Constructor.
Expr_unary::Expr_unary(
    Location const &loc,
    Type *type,
    Operator op,
    Expr *arg)
    : Base(loc, type)
    , m_arg(arg)
    , m_op(op)
{
}

// Get the kind of expression.
Expr::Kind Expr_unary::get_kind() const
{
    return s_kind;
}

// Get the priority of the expression (for pretty-printing).
int Expr_unary::get_prio() const
{
    return 0;
}

// Get the number of sub-expressions.
size_t Expr_unary::get_sub_expression_count() const
{
    return 1;
}

// Get the i'th subexpression.
Expr *Expr_unary::get_sub_expression(size_t i)
{
    return i == 0 ? m_arg : NULL;
}

// Get the i'th subexpression.
Expr const *Expr_unary::get_sub_expression(size_t i) const
{
    return i == 0 ? m_arg : NULL;
}

void Expr_unary::pp(pp::Pretty_print &p) const {

    switch (m_op) {
    case OK_POST_INCREMENT:
        m_arg->pp(p);
        p.string(unary_token(m_op));
        break;
    case OK_POST_DECREMENT:
        m_arg->pp(p);
        p.string(unary_token(m_op));
        break;
    case OK_BITWISE_COMPLEMENT:
    case OK_LOGICAL_NOT:
    case OK_POSITIVE:
    case OK_NEGATIVE:
    case OK_PRE_INCREMENT:
    case OK_PRE_DECREMENT:
        p.string(unary_token(m_op));
        m_arg->pp(p);
        break;
    case OK_OPTION: {
        p.string(unary_token(m_op));
        p.lparen();
        m_arg->pp(p);
        p.rparen();
        break;
    }
    case OK_NONODE: {
        p.string(unary_token(m_op));
        p.lparen();
        m_arg->pp(p);
        p.rparen();
        break;
    }
    case OK_MATCH: {
        p.string(unary_token(m_op));
        p.lparen();
        m_arg->pp(p);
        p.rparen();
        break;
    }
    case OK_IF_GUARD: {
        p.string(unary_token(m_op));
        p.space();
        p.lparen();
        m_arg->pp(p);
        p.rparen();
        break;
    }
    case OK_MAYBE_GUARD: {
        p.string(unary_token(m_op));
        p.space();
        p.lparen();
        m_arg->pp(p);
        p.rparen();
        break;
    }
    }
}

// ------------------------------- Expr_binary -------------------------------

// Constructor.
Expr_binary::Expr_binary(
    Location const &loc,
    Type *type,
    Operator op,
    Expr     *lhs,
    Expr     *rhs)
    : Base(loc, type)
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

// Get the priority of the expression (for pretty-printing).
int Expr_binary::get_prio() const
{
    switch (m_op) {
    case OK_SELECT:
    case OK_ARRAY_SUBSCRIPT:
        return 0;
    case OK_MULTIPLY:
    case OK_DIVIDE:
    case OK_MODULO:
        return 1;
    case OK_PLUS:
    case OK_MINUS:
        return 2;
    case OK_SHIFT_LEFT:
    case OK_SHIFT_RIGHT:
    case OK_SHIFT_RIGHT_ARITH:
        return 3;
    case OK_LESS:
    case OK_LESS_OR_EQUAL:
    case OK_GREATER_OR_EQUAL:
    case OK_GREATER:
        return 4;
    case OK_EQUAL:
    case OK_NOT_EQUAL:
        return 5;
    case OK_BITWISE_AND:
        return 6;
    case OK_BITWISE_XOR:
        return 7;
    case OK_BITWISE_OR:
        return 8;
    case OK_LOGICAL_AND:
        return 10;
    case OK_LOGICAL_OR:
        return 11;
    case OK_ASSIGN:
    case OK_TILDE:
        return 12;
    }
    return 0;
}

// Get the number of sub-expressions.
size_t Expr_binary::get_sub_expression_count() const
{
    return 2;
}

// Get the i'th subexpression.
Expr *Expr_binary::get_sub_expression(size_t i)
{
    switch (i) {
    case 0: return m_lhs;
    case 1: return m_rhs;
    default: return NULL;
    }
}

// Get the i'th subexpression.
Expr const *Expr_binary::get_sub_expression(size_t i) const
{
    switch (i) {
    case 0: return m_lhs;
    case 1: return m_rhs;
    default: return NULL;
    }
}

void Expr_binary::pp(pp::Pretty_print &p) const {

    bool paren_left =  m_lhs->get_prio() > 0 && m_lhs->get_prio() > get_prio();
    bool paren_right = m_rhs->get_prio() > 0 && m_rhs->get_prio() >= get_prio();

    if (m_op == OK_ARRAY_SUBSCRIPT) {
        m_lhs->pp(p);
        p.lbracket();
        m_rhs->pp(p);
        p.rbracket();
        return;
    }

    if (paren_left)
        p.lparen();
    m_lhs->pp(p);
    if (paren_left)
        p.rparen();

    switch (m_op) {
    case OK_ARRAY_SUBSCRIPT:
        break;
    case OK_SELECT:
        p.string(binary_token(m_op));
        break;

    case OK_MULTIPLY:
    case OK_DIVIDE:
    case OK_MODULO:
    case OK_PLUS:
    case OK_MINUS:
    case OK_SHIFT_LEFT:
    case OK_SHIFT_RIGHT:
    case OK_SHIFT_RIGHT_ARITH:
    case OK_LESS:
    case OK_LESS_OR_EQUAL:
    case OK_GREATER_OR_EQUAL:
    case OK_GREATER:
    case OK_EQUAL:
    case OK_NOT_EQUAL:
    case OK_BITWISE_AND:
    case OK_BITWISE_OR:
    case OK_BITWISE_XOR:
    case OK_LOGICAL_AND:
    case OK_LOGICAL_OR:
    case OK_ASSIGN:
    case OK_TILDE:
        p.space();
        p.string(binary_token(m_op));
        p.space();
        break;
    }
    if (paren_right)
        p.lparen();
    m_rhs->pp(p);
    if (paren_right)
        p.rparen();
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

// Get the priority of the expression (for pretty-printing).
int Expr_conditional::get_prio() const
{
    return 0;
}

// Get the number of sub-expressions.
size_t Expr_conditional::get_sub_expression_count() const
{
    return 3;
}

// Get the i'th subexpression.
Expr *Expr_conditional::get_sub_expression(size_t i)
{
    switch (i) {
    case 0: return m_cond;
    case 1: return m_true_expr;
    case 2: return m_false_expr;
    default: return NULL;
    }
}

// Get the i'th subexpression.
Expr const *Expr_conditional::get_sub_expression(size_t i) const
{
    switch (i) {
    case 0: return m_cond;
    case 1: return m_true_expr;
    case 2: return m_false_expr;
    default: return NULL;
    }
}

void Expr_conditional::pp(pp::Pretty_print &p) const {
    m_cond->pp(p);
    p.space();
    p.string("?");
    p.space();
    m_true_expr->pp(p);
    p.space();
    p.string(":");
    p.space();
    m_false_expr->pp(p);
}

// ------------------------------ Argument -------------------------------

// Constructor.
Argument::Argument(Expr *expr)
    : m_expr(expr)
{
}

Expr *Argument::get_expr() {
    return m_expr;
}

Expr const *Argument::get_expr() const {
    return m_expr;
}

// ------------------------------ Expr_call ------------------------------

// Constructor.
Expr_call::Expr_call(
    mi::mdl::Memory_arena            *arena,
    Type *type,
    Expr                    *callee)
    : Base(callee->get_location(), type)
    , m_callee(callee)
    , m_arguments()
    , m_argument_count(0)
{
}

// Get the kind of expression.
Expr::Kind Expr_call::get_kind() const
{
    return s_kind;
}

// Get the priority of the expression (for pretty-printing).
int Expr_call::get_prio() const
{
    return 0;
}

// Get the number of sub-expressions.
size_t Expr_call::get_sub_expression_count() const
{
    return 1 + m_argument_count;
}

// Get the i'th subexpression.
Expr *Expr_call::get_sub_expression(size_t i)
{
    if (i == 0)
        return m_callee;
    if (i <= m_argument_count)
        return get_argument(i - 1);
    return NULL;
}

// Get the i'th subexpression.
Expr const *Expr_call::get_sub_expression(size_t i) const
{
    if (i == 0)
        return m_callee;
    if (i <= m_argument_count)
        return get_argument(i - 1);
    return NULL;
}

// Get the argument at index.
Expr *Expr_call::get_argument(size_t index)
{
    mi::mdl::Ast_list<Argument>::iterator it(m_arguments.begin()),
        end(m_arguments.end());
    for (; it != end && index > 0; --index, ++it) {
    }
    if (it != end && index == 0)
        return it->get_expr();
    return NULL;
}

// Get the argument at index.
Expr const *Expr_call::get_argument(size_t index) const
{
    Expr_call *c = const_cast<Expr_call *>(this);
    return c->get_argument(index);
}

Argument_list const &Expr_call::get_argument_list() const
{
    return m_arguments;
}

void Expr_call::add_argument(Argument *arg)
{
    m_arguments.push(arg);
    m_argument_count++;
}

void Expr_call::pp(pp::Pretty_print &p) const {
    bool first = true;
    m_callee->pp(p);
    p.lparen();
    for (mi::mdl::Ast_list<Argument>::const_iterator it(m_arguments.begin()), end(m_arguments.end());
         it != end;
         ++it) {
        if (first)
            first = false;
        else {
            p.comma();
            p.space();
        }
        it->get_expr()->pp(p);
    }
    p.rparen();
    if (p.flags() & pp::Pretty_print::Flags::PRINT_RETURN_TYPES) {
        p.string(":");
        get_type()->pp(p);
    }
}

// ------------------------------- Expr_attribute -------------------------------

// Constructor.
Expr_attribute::Expr_attribute(
    Location const &loc,
    Type *type,
    Expr *arg,
    Expr_attribute::Expr_attribute_vector attributes,
    char const *node_name)
    : Base(loc, type)
    , m_arg(arg)
    , m_attributes(attributes)
    , m_node_name(node_name)
{
}

// Get the kind of expression.
Expr::Kind Expr_attribute::get_kind() const
{
    return s_kind;
}

// Get the priority of the expression (for pretty-printing).
int Expr_attribute::get_prio() const
{
    return 0;
}

// Get the number of sub-expressions.
size_t Expr_attribute::get_sub_expression_count() const
{
    return 1;
}

// Get the i'th subexpression.
Expr *Expr_attribute::get_sub_expression(size_t i)
{
    return i == 0 ? m_arg : NULL;
}

// Get the i'th subexpression.
Expr const *Expr_attribute::get_sub_expression(size_t i) const
{
    return i == 0 ? m_arg : NULL;
}

void Expr_attribute::pp(pp::Pretty_print &p) const {
    m_arg->pp(p);
    p.space();
    p.string("[[");
    p.space();
    for (size_t i = 0; i < m_attributes.size(); i++) {
        Expr_attribute_entry const &attr = m_attributes[i];

        if (i > 0) {
            p.comma();
            p.space();
        }
        if (p.flags() & pp::Pretty_print::Flags::PRINT_ATTRIBUTE_TYPES) {
            m_attributes[i].type->pp(p);
            p.space();
        }
        p.string(attr.name->get_name());
        if (attr.expr) {
            p.space();
            if (attr.is_pattern) {
                p.string("~");
            } else {
                p.string("=");
            }
            p.space();
            attr.expr->pp(p);
        }
    }
    p.space();
    p.string("]]");
}

char const *Expr_attribute::get_node_name() const {
    return m_node_name;
}

// -------------------------------------- Expr_factory --------------------------------------

// Constructs a new expression factory.
Expr_factory::Expr_factory(
    mi::mdl::Memory_arena  &arena,
    Compilation_unit *compilation_unit,
    Value_factory &vf)
    : Base()
    , m_arena(arena)
    , m_builder(arena)
    , m_compilation_unit(compilation_unit)
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
Expr_ref *Expr_factory::create_reference(
    Location const &loc,
    Type *type,
    Symbol *symbol)
{
    MDL_ASSERT(symbol != NULL);

    return m_builder.create<Expr_ref>(loc, type, symbol);
}

// Create a new unary expression.
Expr_type_annotation *Expr_factory::create_type_annotation(
    Location const &loc,
    Type *type,
    Expr *arg,
    Symbol *type_name)
{
    MDL_ASSERT(arg != NULL);
    return m_builder.create<Expr_type_annotation>(loc, type, arg, type_name);
}

// Create a new unary expression.
Expr_unary *Expr_factory::create_unary(
    Location const &loc,
    Type *type,
    Expr_unary::Operator op,
    Expr *arg)
{
    MDL_ASSERT(arg != NULL);
    return m_builder.create<Expr_unary>(loc, type, op, arg);
}

// Create a new binary expression.
Expr_binary *Expr_factory::create_binary(
    Location const &loc,
    Type *type,
    Expr_binary::Operator op,
    Expr *left,
    Expr *right)
{
    MDL_ASSERT(left != NULL && right != NULL);

    return m_builder.create<Expr_binary>(loc, type, op, left, right);
}

// Create a new conditional expression.
Expr_conditional *Expr_factory::create_conditional(
    Expr           *cond,
    Expr           *true_expr,
    Expr           *false_expr)
{
    MDL_ASSERT(cond != NULL && true_expr != NULL && false_expr != NULL);
    return m_builder.create<Expr_conditional>(cond, true_expr, false_expr);
}

// Create a new call argument
Argument *Expr_factory::create_argument(Expr *expr)
{
    MDL_ASSERT(expr != NULL);
    return m_builder.create<Argument>(expr);
}

// Create a new call expression.
Expr_call *Expr_factory::create_call(
    Type *type,
    Expr *callee)
{
    MDL_ASSERT(callee != NULL);
    return m_builder.create<Expr_call>(m_builder.get_arena(), type, callee);
}

// Create a new attribute expression.
Expr_attribute *Expr_factory::create_attribute(
    Location const &loc,
    Type *type,
    Expr *arg,
    Expr_attribute::Expr_attribute_vector attributes,
    char const *node_name)
{
    MDL_ASSERT(arg != NULL);
    return m_builder.create<Expr_attribute>(loc, type, arg, attributes,
                                            Arena_strdup(m_arena, node_name));
}
