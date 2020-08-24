/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/mdl/mdl_declarations.h>
#include <mi/mdl/mdl_expressions.h>
#include <mi/mdl/mdl_modules.h>
#include <mi/mdl/mdl_names.h>
#include <mi/mdl/mdl_statements.h>

#include "compilercore_visitor.h"
#include "compilercore_tools.h"

namespace mi {
namespace mdl {

Module_visitor::Module_visitor()
{
}

void Module_visitor::visit(
    IModule const *module)
{
    for (size_t i = 0, n = module->get_declaration_count(); i < n; ++i) {
        IDeclaration const *decl = module->get_declaration(i);

        do_declaration(decl);
    }
}

void Module_visitor::visit(
    IStatement const *stmt)
{
    do_statement(stmt);
}

IExpression const *Module_visitor::visit(
    IExpression const *expr)
{
    return do_expression(expr);
}

void Module_visitor::visit(
    IDeclaration const *decl)
{
    do_declaration(decl);
}

void Module_visitor::visit(
    IArgument const *arg)
{
    do_argument(arg);
}

void Module_visitor::visit(
    IParameter const *param)
{
    do_parameter(param);
}

void Module_visitor::visit(
    IType_name const *tname)
{
    do_type_name(tname);
}

void Module_visitor::visit(
    IAnnotation_block const *block)
{
    do_annotations(block);
}

void Module_visitor::visit(
    IQualified_name const *qual_name)
{
    do_qualified_name(qual_name);
}

void Module_visitor::visit(
    ISimple_name const *sname)
{
    do_simple_name(sname);
}

// ----------------------- declarations -----------------------
bool Module_visitor::pre_visit(IDeclaration *decl) { return true; }
void Module_visitor::post_visit(IDeclaration *decl) {}

bool Module_visitor::pre_visit(IDeclaration_invalid *decl) {
    return pre_visit(static_cast<IDeclaration *>(decl));
}
void Module_visitor::post_visit(IDeclaration_invalid *decl) {
    post_visit(static_cast<IDeclaration *>(decl));
}

bool Module_visitor::pre_visit(IDeclaration_import *decl) {
    return pre_visit(static_cast<IDeclaration *>(decl));
}
void Module_visitor::post_visit(IDeclaration_import *decl) {
    post_visit(static_cast<IDeclaration *>(decl));
}

bool Module_visitor::pre_visit(IDeclaration_annotation *anno) {
    return pre_visit(static_cast<IDeclaration *>(anno));
}
void Module_visitor::post_visit(IDeclaration_annotation *anno) {
    post_visit(static_cast<IDeclaration *>(anno));
}

bool Module_visitor::pre_visit(IDeclaration_constant *con) {
    return pre_visit(static_cast<IDeclaration *>(con));
}
void Module_visitor::post_visit(IDeclaration_constant *con) {
    post_visit(static_cast<IDeclaration *>(con));
}

bool Module_visitor::pre_visit(ISimple_name *name) { return true; }
void Module_visitor::post_visit(ISimple_name *name) {}

bool Module_visitor::pre_visit(IQualified_name *name) { return true; }
void Module_visitor::post_visit(IQualified_name *name) {}

bool Module_visitor::pre_visit(IParameter *param) { return true; }
void Module_visitor::post_visit(IParameter *param) {}

bool Module_visitor::pre_visit(IAnnotation_block *block) { return true; }
void Module_visitor::post_visit(IAnnotation_block *block) {}

bool Module_visitor::pre_visit(IAnnotation *anno) { return true; }
void Module_visitor::post_visit(IAnnotation *anno) {}

bool Module_visitor::pre_visit(IArgument_named *arg) { return true; }
void Module_visitor::post_visit(IArgument_named *arg) {}

bool Module_visitor::pre_visit(IArgument_positional *arg) { return true; }
void Module_visitor::post_visit(IArgument_positional *arg) {}

bool Module_visitor::pre_visit(IType_name *tname) { return true; }
void Module_visitor::post_visit(IType_name *tname) {}

bool Module_visitor::pre_visit(IDeclaration_type_alias *type_alias) { return true; }
void Module_visitor::post_visit(IDeclaration_type_alias *type_alias) {}

bool Module_visitor::pre_visit(IDeclaration_type_struct *type_struct) { return true; }
void Module_visitor::post_visit(IDeclaration_type_struct *type_struct) {}

bool Module_visitor::pre_visit(IDeclaration_type_enum *type_enum) { return true; }
void Module_visitor::post_visit(IDeclaration_type_enum *type_enum) {}

bool Module_visitor::pre_visit(IDeclaration_variable *var_decl) { return true; }
void Module_visitor::post_visit(IDeclaration_variable *var_decl) {}

bool Module_visitor::pre_visit(IDeclaration_function *fkt_decl) { return true; }
void Module_visitor::post_visit(IDeclaration_function *fkt_decl) {}

bool Module_visitor::pre_visit(IDeclaration_module *mod_decl) { return true; }
void Module_visitor::post_visit(IDeclaration_module *mod_decl) {}

bool Module_visitor::pre_visit(IDeclaration_namespace_alias *alias_decl) { return true; }
void Module_visitor::post_visit(IDeclaration_namespace_alias *alias_decl) {}

// ----------------------- expressions -----------------------

bool Module_visitor::pre_visit(IExpression *expr) { return true; }
IExpression *Module_visitor::post_visit(IExpression *expr) { return expr; }

bool Module_visitor::pre_visit(IExpression_invalid *expr) {
    return pre_visit(static_cast<IExpression *>(expr));
}
IExpression *Module_visitor::post_visit(IExpression_invalid *expr) {
    return post_visit(static_cast<IExpression *>(expr));
}

bool Module_visitor::pre_visit(IExpression_literal *expr) {
    return pre_visit(static_cast<IExpression *>(expr));
}
IExpression *Module_visitor::post_visit(IExpression_literal *expr) {
    return post_visit(static_cast<IExpression *>(expr));
}

bool Module_visitor::pre_visit(IExpression_reference *expr) {
    return pre_visit(static_cast<IExpression *>(expr));
}
IExpression *Module_visitor::post_visit(IExpression_reference *expr) {
    return post_visit(static_cast<IExpression *>(expr));
}

bool Module_visitor::pre_visit(IExpression_unary *expr) {
    return pre_visit(static_cast<IExpression *>(expr));
}
IExpression *Module_visitor::post_visit(IExpression_unary *expr) {
    return post_visit(static_cast<IExpression *>(expr));
}

bool Module_visitor::pre_visit(IExpression_binary *expr) {
    return pre_visit(static_cast<IExpression *>(expr));
}
IExpression *Module_visitor::post_visit(IExpression_binary *expr) {
    return post_visit(static_cast<IExpression *>(expr));
}

bool Module_visitor::pre_visit(IExpression_conditional *expr) {
    return pre_visit(static_cast<IExpression *>(expr));
}
IExpression *Module_visitor::post_visit(IExpression_conditional *expr) {
    return post_visit(static_cast<IExpression *>(expr));
}

bool Module_visitor::pre_visit(IExpression_call *expr) {
    return pre_visit(static_cast<IExpression *>(expr));
}
IExpression *Module_visitor::post_visit(IExpression_call *expr) {
    return post_visit(static_cast<IExpression *>(expr));
}

bool Module_visitor::pre_visit(IExpression_let *expr) {
    return pre_visit(static_cast<IExpression *>(expr));
}
IExpression *Module_visitor::post_visit(IExpression_let *expr) {
    return post_visit(static_cast<IExpression *>(expr));
}

// ----------------------- statements -----------------------

bool Module_visitor::pre_visit(IStatement *stmt) { return true; }
void Module_visitor::post_visit(IStatement *stmt) {}

bool Module_visitor::pre_visit(IStatement_invalid *stmt) {
    return pre_visit(static_cast<IStatement *>(stmt));
}
void Module_visitor::post_visit(IStatement_invalid *stmt) {
    post_visit(static_cast<IStatement *>(stmt));
}

bool Module_visitor::pre_visit(IStatement_compound *stmt) {
    return pre_visit(static_cast<IStatement *>(stmt));
}
void Module_visitor::post_visit(IStatement_compound *stmt) {
    post_visit(static_cast<IStatement *>(stmt));
}

bool Module_visitor::pre_visit(IStatement_declaration *stmt) {
    return pre_visit(static_cast<IStatement *>(stmt));
}
void Module_visitor::post_visit(IStatement_declaration *stmt) {
    post_visit(static_cast<IStatement *>(stmt));
}

bool Module_visitor::pre_visit(IStatement_expression *stmt) {
    return pre_visit(static_cast<IStatement *>(stmt));
}
void Module_visitor::post_visit(IStatement_expression *stmt) {
    post_visit(static_cast<IStatement *>(stmt));
}

bool Module_visitor::pre_visit(IStatement_if *stmt) {
    return pre_visit(static_cast<IStatement *>(stmt));
}
void Module_visitor::post_visit(IStatement_if *stmt) {
    post_visit(static_cast<IStatement *>(stmt));
}

bool Module_visitor::pre_visit(IStatement_case *stmt) {
    return pre_visit(static_cast<IStatement *>(stmt));
}
void Module_visitor::post_visit(IStatement_case *stmt) {
    post_visit(static_cast<IStatement *>(stmt));
}

bool Module_visitor::pre_visit(IStatement_switch *stmt) {
    return pre_visit(static_cast<IStatement *>(stmt));
}
void Module_visitor::post_visit(IStatement_switch *stmt) {
    post_visit(static_cast<IStatement *>(stmt));
}

bool Module_visitor::pre_visit(IStatement_while *stmt) {
    return pre_visit(static_cast<IStatement *>(stmt));
}
void Module_visitor::post_visit(IStatement_while *stmt) {
    post_visit(static_cast<IStatement *>(stmt));
}

bool Module_visitor::pre_visit(IStatement_do_while *stmt) {
    return pre_visit(static_cast<IStatement *>(stmt));
}
void Module_visitor::post_visit(IStatement_do_while *stmt) {
    post_visit(static_cast<IStatement *>(stmt));
}

bool Module_visitor::pre_visit(IStatement_for *stmt) {
    return pre_visit(static_cast<IStatement *>(stmt));
}
void Module_visitor::post_visit(IStatement_for *stmt) {
    post_visit(static_cast<IStatement *>(stmt));
}

bool Module_visitor::pre_visit(IStatement_break *stmt) {
    return pre_visit(static_cast<IStatement *>(stmt));
}
void Module_visitor::post_visit(IStatement_break *stmt) {
    post_visit(static_cast<IStatement *>(stmt));
}

bool Module_visitor::pre_visit(IStatement_continue *stmt) {
    return pre_visit(static_cast<IStatement *>(stmt));
}
void Module_visitor::post_visit(IStatement_continue *stmt) {
    post_visit(static_cast<IStatement *>(stmt));
}

bool Module_visitor::pre_visit(IStatement_return *stmt) {
    return pre_visit(static_cast<IStatement *>(stmt));
}
void Module_visitor::post_visit(IStatement_return *stmt) {
    post_visit(static_cast<IStatement *>(stmt));
}

// ----------------------- internal -----------------------

void Module_visitor::do_simple_name(ISimple_name const *name)
{
    ISimple_name *d = const_cast<ISimple_name *>(name);

    pre_visit(d);
    post_visit(d);
}

void Module_visitor::do_qualified_name(IQualified_name const *name) {
    IQualified_name *d = const_cast<IQualified_name *>(name);

    if (pre_visit(d)) {
        for (size_t i = 0, n = d->get_component_count(); i < n; ++i) {
            ISimple_name const *name = d->get_component(i);

            do_simple_name(name);
        }
    }
    post_visit(d);
}

IExpression const *Module_visitor::do_invalid_expression(
    IExpression_invalid const *expr)
{
    IExpression_invalid *e = const_cast<IExpression_invalid *>(expr);

    pre_visit(e);
    return post_visit(e);
}

IExpression const *Module_visitor::do_literal_expression(
    IExpression_literal const *expr)
{
    IExpression_literal *e = const_cast<IExpression_literal *>(expr);

    pre_visit(e);
    return post_visit(e);
}

IExpression const *Module_visitor::do_reference_expression(
    IExpression_reference const *expr)
{
    IExpression_reference *e = const_cast<IExpression_reference *>(expr);

    if (pre_visit(e)) {
        // visit the referenced object
        IType_name const *rname = e->get_name();
        do_type_name(rname);
    }
    return post_visit(e);
}

IExpression const *Module_visitor::do_unary_expression(
    IExpression_unary const *expr)
{
    IExpression_unary *e = const_cast<IExpression_unary *>(expr);

    if (pre_visit(e)) {
        IExpression const *op   = e->get_argument();
        IExpression const *n_op = do_expression(op);
        if (n_op != op) {
            e->set_argument(n_op);
        }

        if (IType_name const *tn = e->get_type_name()) {
            do_type_name(tn);
        }
    }
    return post_visit(e);
}

IExpression const *Module_visitor::do_binary_expression(
    IExpression_binary const *expr)
{
    IExpression_binary *e = const_cast<IExpression_binary *>(expr);

    if (pre_visit(e)) {
        IExpression const *l   = e->get_left_argument();
        IExpression const *n_l = do_expression(l);
        if (n_l != l) {
            e->set_left_argument(n_l);
        }

        IExpression const *r   = e->get_right_argument();
        IExpression const *n_r = do_expression(r);
        if (n_r != r) {
            e->set_right_argument(n_r);
        }
    }
    return post_visit(e);
}

IExpression const *Module_visitor::do_conditional_expression(
    IExpression_conditional const *expr)
{
    IExpression_conditional *e = const_cast<IExpression_conditional *>(expr);

    if (pre_visit(e)) {
        IExpression const *c   = e->get_condition();
        IExpression const *n_c = do_expression(c);
        if (n_c != c) {
            e->set_condition(n_c);
        }

        IExpression const *t   = e->get_true();
        IExpression const *n_t = do_expression(t);
        if (n_t != t) {
            e->set_true(n_t);
        }

        IExpression const *f   = e->get_false();
        IExpression const *n_f = do_expression(f);
        if (n_f != f) {
            e->set_false(n_f);
        }
    }
    return post_visit(e);
}

IExpression const *Module_visitor::do_call_expression(
    IExpression_call const *expr)
{
    IExpression_call *e = const_cast<IExpression_call *>(expr);

    if (pre_visit(e)) {
        IExpression const *ref   = e->get_reference();
        IExpression const *n_ref = do_expression(ref);
        if (n_ref != ref) {
            e->set_reference(n_ref);
        }

        for (size_t i = 0, n = e->get_argument_count(); i < n; ++i) {
            IArgument const *arg = e->get_argument(i);
            do_argument(arg);
        }
    }
    return post_visit(e);
}

IExpression const *Module_visitor::do_let_expression(
    IExpression_let const *expr)
{
    IExpression_let *e = const_cast<IExpression_let *>(expr);

    if (pre_visit(e)) {
        for (size_t i = 0, n = e->get_declaration_count(); i < n; ++i) {
            IDeclaration const *decl = e->get_declaration(i);
            do_declaration(decl);
        }

        IExpression const *ex   = e->get_expression();
        IExpression const *n_ex = do_expression(ex);
        if (n_ex != ex) {
            e->set_expression(n_ex);
        }
    }
    return post_visit(e);
}

IExpression const *Module_visitor::do_expression(
    IExpression const *expr)
{
    switch (expr->get_kind()) {
    case IExpression::EK_INVALID:
        {
            IExpression_invalid const *e = cast<IExpression_invalid>(expr);
            return do_invalid_expression(e);
        }
    case IExpression::EK_LITERAL:
        {
            IExpression_literal const *e = cast<IExpression_literal>(expr);
            return do_literal_expression(e);
        }
    case IExpression::EK_REFERENCE:
        {
            IExpression_reference const *e = cast<IExpression_reference>(expr);
            return do_reference_expression(e);
        }
    case IExpression::EK_UNARY:
        {
            IExpression_unary const *e = cast<IExpression_unary>(expr);
            return do_unary_expression(e);
        }
    case IExpression::EK_BINARY:
        {
            IExpression_binary const *e = cast<IExpression_binary>(expr);
            return do_binary_expression(e);
        }
    case IExpression::EK_CONDITIONAL:
        {
            IExpression_conditional const *e = cast<IExpression_conditional>(expr);
            return do_conditional_expression(e);
        }
    case IExpression::EK_CALL:
        {
            IExpression_call const *e = cast<IExpression_call>(expr);
            return do_call_expression(e);
        }
    case IExpression::EK_LET:
        {
            IExpression_let const *e = cast<IExpression_let>(expr);
            return do_let_expression(e);
        }
    }
    MDL_ASSERT("!unhandled expression kind");
    return expr;
}

void Module_visitor::do_invalid_statement(IStatement_invalid const *stmt)
{
    IStatement_invalid *s = const_cast<IStatement_invalid *>(stmt);

    pre_visit(s);
    post_visit(s);
}

void Module_visitor::do_compound_statement(
    IStatement_compound const *stmt)
{
    IStatement_compound *s = const_cast<IStatement_compound *>(stmt);

    if (pre_visit(s)) {
        for (size_t i = 0, n = s->get_statement_count(); i < n; ++i) {
            IStatement const *st = s->get_statement(i);
            do_statement(st);
        }
    }
    post_visit(s);
}

void Module_visitor::do_declaration_statement(
    IStatement_declaration const *stmt)
{
    IStatement_declaration *s = const_cast<IStatement_declaration *>(stmt);

    if (pre_visit(s)) {
        IDeclaration const *decl = s->get_declaration();
        do_declaration(decl);
    }
    post_visit(s);
}

void Module_visitor::do_expression_statement(
    IStatement_expression const *stmt)
{
    IStatement_expression *s = const_cast<IStatement_expression *>(stmt);

    if (pre_visit(s)) {
        if (IExpression const *expr = s->get_expression()) {
            IExpression const *n_expr = do_expression(expr);
            if (n_expr != expr) {
                s->set_expression(n_expr);
            }
        }
    }
    post_visit(s);
}

void Module_visitor::do_if_statement(
    IStatement_if const *stmt)
{
    IStatement_if *s = const_cast<IStatement_if *>(stmt);

    if (pre_visit(s)) {
        IExpression const *cond   = s->get_condition();
        IExpression const *n_cond = do_expression(cond);
        if (n_cond != cond) {
            s->set_condition(n_cond);
        }

        IStatement const *t = s->get_then_statement();
        do_statement(t);

        if (IStatement const *e = s->get_else_statement()) {
            do_statement(e);
        }
    }
    post_visit(s);
}

void Module_visitor::do_case_statement(
    IStatement_case const *stmt)
{
    IStatement_case *s = const_cast<IStatement_case *>(stmt);

    if (pre_visit(s)) {
        if (IExpression const *label = s->get_label()) {
            IExpression const *n_label = do_expression(label);
            if (n_label != label) {
                s->set_label(n_label);
            }
        }

        for (size_t i = 0, n = s->get_statement_count(); i < n; ++i) {
            IStatement const *st = s->get_statement(i);
            do_statement(st);
        }
    }
    post_visit(s);
}

void Module_visitor::do_switch_statement(
    IStatement_switch const *stmt)
{
    IStatement_switch *s = const_cast<IStatement_switch *>(stmt);

    if (pre_visit(s)) {
        IExpression const *cond   = s->get_condition();
        IExpression const *n_cond = do_expression(cond);
        if (n_cond != cond) {
            s->set_condition(n_cond);
        }

        for (size_t i = 0, n = s->get_case_count(); i < n; ++i) {
            IStatement const *st = s->get_case(i);
            do_statement(st);
        }
    }
    post_visit(s);
}

void Module_visitor::do_while_statement(
    IStatement_while const *stmt)
{
    IStatement_while *s = const_cast<IStatement_while *>(stmt);

    if (pre_visit(s)) {
        IExpression const *cond   = s->get_condition();
        IExpression const *n_cond = do_expression(cond);
        if (n_cond != cond) {
            s->set_condition(n_cond);
        }

        IStatement const *body = s->get_body();
        do_statement(body);
    }
    post_visit(s);
}

void Module_visitor::do_do_while_statement(
    IStatement_do_while const *stmt)
{
    IStatement_do_while *s = const_cast<IStatement_do_while *>(stmt);

    if (pre_visit(s)) {
        IStatement const *body = s->get_body();
        do_statement(body);

        IExpression const *cond   = s->get_condition();
        IExpression const *n_cond = do_expression(cond);
        if (n_cond != cond) {
            s->set_condition(n_cond);
        }
    }
    post_visit(s);
}

void Module_visitor::do_for_statement(
    IStatement_for const *stmt)
{
    IStatement_for *s = const_cast<IStatement_for *>(stmt);

    if (pre_visit(s)) {
        if (IStatement const *init = s->get_init()) {
            do_statement(init);
        }

        if (IExpression const *cond = s->get_condition()) {
            IExpression const *n_cond = do_expression(cond);
            if (n_cond != cond) {
                s->set_condition(n_cond);
            }
        }

        if (IExpression const *upd = s->get_update()) {
            IExpression const *n_upd = do_expression(upd);
            if (n_upd != upd) {
                s->set_update(n_upd);
            }
        }

        IStatement const *body = s->get_body();
        do_statement(body);
    }
    post_visit(s);
}

void Module_visitor::do_break_statement(
    IStatement_break const *stmt)
{
    IStatement_break *s = const_cast<IStatement_break *>(stmt);

    pre_visit(s);
    post_visit(s);
}

void Module_visitor::do_continue_statement(
    IStatement_continue const *stmt)
{
    IStatement_continue *s = const_cast<IStatement_continue *>(stmt);

    pre_visit(s);
    post_visit(s);
}

void Module_visitor::do_return_statement(
    IStatement_return const *stmt)
{
    IStatement_return *s = const_cast<IStatement_return *>(stmt);

    if (pre_visit(s)) {
        if (IExpression const *expr = s->get_expression()) {
            IExpression const *n_expr = do_expression(expr);
            if (n_expr != expr) {
                s->set_expression(n_expr);
            }
        }
    }
    post_visit(s);
}

void Module_visitor::do_statement(
    IStatement const *stmt)
{
    switch (stmt->get_kind()) {
    case IStatement::SK_INVALID:
        {
            IStatement_invalid const *s = static_cast<IStatement_invalid const *>(stmt);
            do_invalid_statement(s);
            break;
        }
    case IStatement::SK_COMPOUND:
        {
            IStatement_compound const *s = static_cast<IStatement_compound const *>(stmt);
            do_compound_statement(s);
            break;
        }
    case IStatement::SK_DECLARATION:
        {
            IStatement_declaration const *s = static_cast<IStatement_declaration const *>(stmt);
            do_declaration_statement(s);
            break;
        }
    case IStatement::SK_EXPRESSION:
        {
            IStatement_expression const *s = static_cast<IStatement_expression const *>(stmt);
            do_expression_statement(s);
            break;
        }
    case IStatement::SK_IF:
        {
            IStatement_if const *s = static_cast<IStatement_if const *>(stmt);
            do_if_statement(s);
            break;
        }
    case IStatement::SK_CASE:
        {
            IStatement_case const *s = static_cast<IStatement_case const *>(stmt);
            do_case_statement(s);
            break;
        }
    case IStatement::SK_SWITCH:
        {
            IStatement_switch const *s = static_cast<IStatement_switch const *>(stmt);
            do_switch_statement(s);
            break;
        }
    case IStatement::SK_WHILE:
        {
            IStatement_while const *s = static_cast<IStatement_while const *>(stmt);
            do_while_statement(s);
            break;
        }
    case IStatement::SK_DO_WHILE:
        {
            IStatement_do_while const *s = static_cast<IStatement_do_while const *>(stmt);
            do_do_while_statement(s);
            break;
        }
    case IStatement::SK_FOR:
        {
            IStatement_for const *s = static_cast<IStatement_for const *>(stmt);
            do_for_statement(s);
            break;
        }
    case IStatement::SK_BREAK:
        {
            IStatement_break const *s = static_cast<IStatement_break const *>(stmt);
            do_break_statement(s);
            break;
        }
    case IStatement::SK_CONTINUE:
        {
            IStatement_continue const *s = static_cast<IStatement_continue const *>(stmt);
            do_continue_statement(s);
            break;
        }
    case IStatement::SK_RETURN:
        {
            IStatement_return const *s = static_cast<IStatement_return const *>(stmt);
            do_return_statement(s);
            break;
        }
    }
}

void Module_visitor::do_invalid_import(IDeclaration_invalid const *import)
{
    IDeclaration_invalid *d = const_cast<IDeclaration_invalid *>(import);

    pre_visit(d);
    post_visit(d);
}

void Module_visitor::do_declaration_import(IDeclaration_import const *import)
{
    IDeclaration_import *d = const_cast<IDeclaration_import *>(import);

    if (pre_visit(d)) {
        if (IQualified_name const *using_mod = d->get_module_name())
            do_qualified_name(using_mod);
        for (size_t i = 0, n = d->get_name_count(); i < n; ++i) {
            IQualified_name const *qname = d->get_name(i);

            do_qualified_name(qname);
        }
    }
    post_visit(d);
}

void Module_visitor::do_named_argument(
    IArgument_named const *arg)
{
    IArgument_named *a = const_cast<IArgument_named *>(arg);

    if (pre_visit(a)) {
        ISimple_name const *sname = a->get_parameter_name();
        do_simple_name(sname);

        IExpression const *expr   = a->get_argument_expr();
        IExpression const *n_expr = do_expression(expr);
        if (n_expr != expr) {
            a->set_argument_expr(n_expr);
        }
    }
    post_visit(a);
}

void Module_visitor::do_positional_argument(
    IArgument_positional const *arg)
{
    IArgument_positional *a = const_cast<IArgument_positional *>(arg);

    if (pre_visit(a)) {
        IExpression const *expr   = a->get_argument_expr();
        IExpression const *n_expr = do_expression(expr);
        if (n_expr != expr) {
            a->set_argument_expr(n_expr);
        }
    }
    post_visit(a);
}

void Module_visitor::do_argument(
    IArgument const *arg)
{
    switch (arg->get_kind()) {
    case IArgument::AK_NAMED:
        do_named_argument(static_cast<IArgument_named const *>(arg));
        break;
    case IArgument::AK_POSITIONAL:
        do_positional_argument(static_cast<IArgument_positional const *>(arg));
        break;
    }
}

void Module_visitor::do_annotation(
    IAnnotation const *anno)
{
    IAnnotation *a = const_cast<IAnnotation *>(anno);

    if (pre_visit(a)) {
        IQualified_name const *qname = a->get_name();
        do_qualified_name(qname);

        for (size_t i = 0, n = a->get_argument_count(); i < n; ++i) {
            IArgument const *arg = a->get_argument(i);

            do_argument(arg);
        }

        if (IAnnotation_enable_if const *ei = as<IAnnotation_enable_if>(anno)) {
            IAnnotation_enable_if *e = const_cast<IAnnotation_enable_if *>(ei);

            if (IExpression const *expr = e->get_expression()) {
                IExpression const *n_expr = do_expression(expr);
                if (n_expr != expr) {
                    e->set_expression(n_expr);
                }
            }
        }
    }
    post_visit(a);
}

void Module_visitor::do_annotations(
    IAnnotation_block const *block)
{
    IAnnotation_block *b = const_cast<IAnnotation_block *>(block);

    if (pre_visit(b)) {
        for (size_t i = 0, n = b->get_annotation_count(); i < n; ++i) {
            IAnnotation const *a = b->get_annotation(i);
            do_annotation(a);
        }
    }
    post_visit(b);
}

void Module_visitor::do_type_name(
    IType_name const *tn)
{
    IType_name *t = const_cast<IType_name *>(tn);

    if (pre_visit(t)) {
        IQualified_name *qname = t->get_qualified_name();
        do_qualified_name(qname);

        if (IExpression const *size = t->get_array_size()) {
            IExpression const *n_size = do_expression(size);
            if (n_size != size) {
                t->set_array_size(n_size);
            }
        }
    }
    post_visit(t);
}

void Module_visitor::do_parameter(
    IParameter const *param)
{
    IParameter *p = const_cast<IParameter *>(param);

    if (pre_visit(p)) {
        IType_name const *tn = p->get_type_name();
        do_type_name(tn);

        ISimple_name const *sname = p->get_name();
        do_simple_name(sname);

        /// Get the initializing expression.
        if (IExpression const *init = p->get_init_expr()) {
            IExpression const *n_init = do_expression(init);
            if (n_init != init) {
                p->set_init_expr(n_init);
            }
        }

        /// Get the annotation block.
        if (IAnnotation_block const *anno = p->get_annotations()) {
            do_annotations(anno);
        }
    }
    post_visit(p);
}

void Module_visitor::do_declaration_annotation(
    IDeclaration_annotation const *anno)
{
    IDeclaration_annotation *d = const_cast<IDeclaration_annotation *>(anno);

    if (pre_visit(d)) {
        do_simple_name(d->get_name());
        for (size_t i = 0, n = d->get_parameter_count(); i < n; ++i) {
            IParameter const *p = d->get_parameter(i);
            do_parameter(p);
        }
        if (IAnnotation_block const *anno = d->get_annotations()) {
            do_annotations(anno);
        }
    }
    post_visit(d);
}

void Module_visitor::do_declaration_constant(
    IDeclaration_constant const *con)
{
    IDeclaration_constant *d = const_cast<IDeclaration_constant *>(con);

    if (pre_visit(d)) {
        do_type_name(d->get_type_name());
        for (size_t i = 0, n = d->get_constant_count(); i < n; ++i) {
            ISimple_name const *cname = d->get_constant_name(i);
            do_simple_name(cname);

            IExpression const *init   = d->get_constant_exp(i);
            IExpression const *n_init = do_expression(init);
            if (n_init != init) {
                d->set_variable_init(i, n_init);
            }

            if (IAnnotation_block const *anno = d->get_annotations(i)) {
                do_annotations(anno);
            }
        }
    }
    post_visit(d);
}

void Module_visitor::do_type_alias(
    IDeclaration_type_alias const *ta)
{
    IDeclaration_type_alias *t = const_cast<IDeclaration_type_alias *>(ta);

    if (pre_visit(t)) {
        IType_name const *tn = t->get_type_name();
        do_type_name(tn);

        ISimple_name const *sname = t->get_alias_name();
        do_simple_name(sname);
    }
    post_visit(t);
}

void Module_visitor::do_type_struct(
    IDeclaration_type_struct const *ts)
{
    IDeclaration_type_struct *t = const_cast<IDeclaration_type_struct *>(ts);

    if (pre_visit(t)) {
        ISimple_name const *sname = t->get_name();
        do_simple_name(sname);

        if (IAnnotation_block const *anno = ts->get_annotations()) {
            do_annotations(anno);
        }

        for (size_t i = 0, n = t->get_field_count(); i < n; ++i) {
            IType_name const *tname = t->get_field_type_name(i);
            do_type_name(tname);

            ISimple_name const *fname = t->get_field_name(i);
            do_simple_name(fname);

            if (IExpression const *init = t->get_field_init(i)) {
                IExpression const *n_init = do_expression(init);
                if (n_init != init) {
                    t->set_field_init(i, n_init);
                }
            }

            if (IAnnotation_block const *anno = t->get_annotations(i)) {
                do_annotations(anno);
            }
        }
    }
    post_visit(t);
}

void Module_visitor::do_type_enum(
    IDeclaration_type_enum const *te)
{
    IDeclaration_type_enum *t = const_cast<IDeclaration_type_enum *>(te);

    if (pre_visit(t)) {
        ISimple_name const *sname = t->get_name();
        do_simple_name(sname);

        if (IAnnotation_block const *anno = t->get_annotations()) {
            do_annotations(anno);
        }

        for (size_t i = 0, n = t->get_value_count(); i < n; ++i) {
            ISimple_name const *vname = t->get_value_name(i);
            do_simple_name(vname);

            if (IExpression const *init = t->get_value_init(i)) {
                IExpression const *n_init = do_expression(init);
                if (n_init != init) {
                    t->set_value_init(i, n_init);
                }
            }

            if (IAnnotation_block const *anno = t->get_annotations(i)) {
                do_annotations(anno);
            }
        }
    }
    post_visit(t);
}

void Module_visitor::do_variable_decl(
    IDeclaration_variable const *var_decl)
{
    IDeclaration_variable *d = const_cast<IDeclaration_variable *>(var_decl);

    if (pre_visit(d)) {
        IType_name const *tname = d->get_type_name();
        do_type_name(tname);

        for (size_t i = 0, n = d->get_variable_count(); i < n; ++i) {
            ISimple_name const *vname = d->get_variable_name(i);
            do_simple_name(vname);

            if (IExpression const *init = d->get_variable_init(i)) {
                IExpression const *n_init = do_expression(init);
                if (n_init != init) {
                    d->set_variable_init(i, n_init);
                }
            }

            if (IAnnotation_block const *anno = d->get_annotations(i)) {
                do_annotations(anno);
            }
        }
    }
    post_visit(d);
}

void Module_visitor::do_function_decl(
    IDeclaration_function const *fkt)
{
    IDeclaration_function *d = const_cast<IDeclaration_function *>(fkt);

    if (pre_visit(d)) {
        IType_name const *rname = d->get_return_type_name();
        do_type_name(rname);

        if (IAnnotation_block const *ret_anno = d->get_return_annotations())
            do_annotations(ret_anno);

        ISimple_name const *sname = d->get_name();
        do_simple_name(sname);

        for (size_t i = 0, n = d->get_parameter_count(); i < n; ++i) {
            IParameter const *param = d->get_parameter(i);
            do_parameter(param);
        }

        if (IStatement const *body = d->get_body()) {
            do_statement(body);
        }

        if (IAnnotation_block const *anno = d->get_annotations()) {
            do_annotations(anno);
        }
    }
    post_visit(d);
}

void Module_visitor::do_module_decl(
    IDeclaration_module const *mod)
{
    IDeclaration_module *d = const_cast<IDeclaration_module *>(mod);

    if (pre_visit(d)) {
        if (IAnnotation_block const *anno = d->get_annotations()) {
            do_annotations(anno);
        }
    }
    post_visit(d);
}

void Module_visitor::do_namespace_alias(
    IDeclaration_namespace_alias const *alias_decl)
{
    IDeclaration_namespace_alias *d = const_cast<IDeclaration_namespace_alias *>(alias_decl);

    if (pre_visit(d)) {
        do_simple_name(d->get_alias());
        do_qualified_name(d->get_namespace());
    }
    post_visit(d);
}

void Module_visitor::do_declaration(
    IDeclaration const *decl)
{
    switch (decl->get_kind()) {
    case IDeclaration::DK_INVALID:
        {
            IDeclaration_invalid const *import = static_cast<IDeclaration_invalid const *>(decl);
            do_invalid_import(import);
            break;
        }
    case IDeclaration::DK_IMPORT:
        {
            IDeclaration_import const *import = static_cast<IDeclaration_import const *>(decl);
            do_declaration_import(import);
            break;
        }
    case IDeclaration::DK_ANNOTATION:
        {
            IDeclaration_annotation const *anno =
                static_cast<IDeclaration_annotation const *>(decl);
            do_declaration_annotation(anno);
            break;
        }
    case IDeclaration::DK_CONSTANT:
        {
            IDeclaration_constant const *c = static_cast<IDeclaration_constant const *>(decl);
            do_declaration_constant(c);
            break;
        }
    case IDeclaration::DK_TYPE_ALIAS:
        {
            IDeclaration_type_alias const *ta = static_cast<IDeclaration_type_alias const *>(decl);
            do_type_alias(ta);
            break;
        }
    case IDeclaration::DK_TYPE_STRUCT:
        {
            IDeclaration_type_struct const *ts =
                static_cast<IDeclaration_type_struct const *>(decl);
            do_type_struct(ts);
            break;
        }
    case IDeclaration::DK_TYPE_ENUM:
        {
            IDeclaration_type_enum const *te = static_cast<IDeclaration_type_enum const *>(decl);
            do_type_enum(te);
            break;
        }
    case IDeclaration::DK_VARIABLE:
        {
            IDeclaration_variable const *v = static_cast<IDeclaration_variable const *>(decl);
            do_variable_decl(v);
            break;
        }
    case IDeclaration::DK_FUNCTION:
        {
            IDeclaration_function const *f = static_cast<IDeclaration_function const *>(decl);
            do_function_decl(f);
            break;
        }
    case IDeclaration::DK_MODULE:
        {
            IDeclaration_module const *m = static_cast<IDeclaration_module const *>(decl);
            do_module_decl(m);
            break;
        }
    case IDeclaration::DK_NAMESPACE_ALIAS:
        {
            IDeclaration_namespace_alias const *a =
                static_cast<IDeclaration_namespace_alias const *>(decl);
            do_namespace_alias(a);
            break;
        }
    }
}

}  // mdl
}  // mi
