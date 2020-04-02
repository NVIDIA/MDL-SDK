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

#include "compiler_hlsl_compilation_unit.h"
#include "compiler_hlsl_declarations.h"
#include "compiler_hlsl_exprs.h"
#include "compiler_hlsl_stmts.h"
#include "compiler_hlsl_visitor.h"

namespace mi {
namespace mdl {
namespace hlsl {

CUnit_visitor::CUnit_visitor()
{
}

// ----------------------- entry points  -----------------------

void CUnit_visitor::visit(Compilation_unit *unit)
{
    for (Compilation_unit::iterator it(unit->decl_begin()), end(unit->decl_end());
         it != end;
         ++it)
    {
        Declaration *decl = it;

        do_declaration(decl);
    }
}

void CUnit_visitor::visit(Stmt *stmt)
{
    do_statement(stmt);
}

void CUnit_visitor::visit(Expr *expr)
{
    do_expression(expr);
}

void CUnit_visitor::visit(Declaration *decl)
{
    do_declaration(decl);
}

void CUnit_visitor::visit(Type_name *tname)
{
    do_type_name(tname);
}

void CUnit_visitor::visit(Name *name)
{
    do_name(name);
}

void CUnit_visitor::visit(Array_specifier *spec)
{
    do_array_specifier(spec);
}

// ----------------------- declarations -----------------------

bool CUnit_visitor::pre_visit(Name *name) { return true; }
void CUnit_visitor::post_visit(Name *name) {}

bool CUnit_visitor::pre_visit(Layout_qualifier_id *id) { return true; }
void CUnit_visitor::post_visit(Layout_qualifier_id *id) {}

bool CUnit_visitor::pre_visit(Type_qualifier *tq) { return true; }
void CUnit_visitor::post_visit(Type_qualifier *tq) {}

bool CUnit_visitor::pre_visit(Array_specifier *as) { return true; }
void CUnit_visitor::post_visit(Array_specifier *as) {}

bool CUnit_visitor::pre_visit(Type_name *tname) { return true; }
void CUnit_visitor::post_visit(Type_name *tname) {}

bool CUnit_visitor::pre_visit(Declaration *decl) { return true; }
void CUnit_visitor::post_visit(Declaration *decl) {}

bool CUnit_visitor::pre_visit(Declaration_invalid *decl) {
    return pre_visit(static_cast<Declaration *>(decl));
}
void CUnit_visitor::post_visit(Declaration_invalid *decl) {
    post_visit(static_cast<Declaration *>(decl));
}

bool CUnit_visitor::pre_visit(Init_declarator *init) { return true; }
void CUnit_visitor::post_visit(Init_declarator *init) {}

bool CUnit_visitor::pre_visit(Declaration_variable *decl) {
    return pre_visit(static_cast<Declaration *>(decl));
}
void CUnit_visitor::post_visit(Declaration_variable *decl) {
    post_visit(static_cast<Declaration *>(decl));
}

bool CUnit_visitor::pre_visit(Declaration_param *decl) {
    return pre_visit(static_cast<Declaration *>(decl));
}
void CUnit_visitor::post_visit(Declaration_param *decl) {
    post_visit(static_cast<Declaration *>(decl));
}

bool CUnit_visitor::pre_visit(Declaration_function *decl) {
    return pre_visit(static_cast<Declaration *>(decl));
}
void CUnit_visitor::post_visit(Declaration_function *decl) {
    post_visit(static_cast<Declaration *>(decl));
}

bool CUnit_visitor::pre_visit(Field_declarator *field) { return true; }
void CUnit_visitor::post_visit(Field_declarator *field) {}

bool CUnit_visitor::pre_visit(Declaration_field *decl) {
    return pre_visit(static_cast<Declaration *>(decl));
}
void CUnit_visitor::post_visit(Declaration_field *decl) {
    post_visit(static_cast<Declaration *>(decl));
}

bool CUnit_visitor::pre_visit(Declaration_struct *decl) {
    return pre_visit(static_cast<Declaration *>(decl));
}
void CUnit_visitor::post_visit(Declaration_struct *decl) {
    post_visit(static_cast<Declaration *>(decl));
}

bool CUnit_visitor::pre_visit(Declaration_interface *decl) {
    return pre_visit(static_cast<Declaration *>(decl));
}
void CUnit_visitor::post_visit(Declaration_interface *decl) {
    post_visit(static_cast<Declaration *>(decl));
}

bool CUnit_visitor::pre_visit(Instance_name *name) { return true; }
void CUnit_visitor::post_visit(Instance_name *name) {}

bool CUnit_visitor::pre_visit(Declaration_qualified *decl) {
    return pre_visit(static_cast<Declaration *>(decl));
}
void CUnit_visitor::post_visit(Declaration_qualified *decl) {
    post_visit(static_cast<Declaration *>(decl));
}

// ----------------------- expressions -----------------------

bool CUnit_visitor::pre_visit(Expr *expr) { return true; }
void CUnit_visitor::post_visit(Expr *expr) {}

bool CUnit_visitor::pre_visit(Expr_invalid *expr) {
    return pre_visit(static_cast<Expr *>(expr));
}
void CUnit_visitor::post_visit(Expr_invalid *expr) {
    post_visit(static_cast<Expr *>(expr));
}

bool CUnit_visitor::pre_visit(Expr_literal *expr) {
    return pre_visit(static_cast<Expr *>(expr));
}
void CUnit_visitor::post_visit(Expr_literal *expr) {
    post_visit(static_cast<Expr *>(expr));
}

bool CUnit_visitor::pre_visit(Expr_ref *expr) {
    return pre_visit(static_cast<Expr *>(expr));
}
void CUnit_visitor::post_visit(Expr_ref *expr) {
    post_visit(static_cast<Expr *>(expr));
}

bool CUnit_visitor::pre_visit(Expr_unary *expr) {
    return pre_visit(static_cast<Expr *>(expr));
}
void CUnit_visitor::post_visit(Expr_unary *expr) {
    post_visit(static_cast<Expr *>(expr));
}

bool CUnit_visitor::pre_visit(Expr_binary *expr) {
    return pre_visit(static_cast<Expr *>(expr));
}
void CUnit_visitor::post_visit(Expr_binary *expr) {
    post_visit(static_cast<Expr *>(expr));
}

bool CUnit_visitor::pre_visit(Expr_conditional *expr) {
    return pre_visit(static_cast<Expr *>(expr));
}
void CUnit_visitor::post_visit(Expr_conditional *expr) {
    post_visit(static_cast<Expr *>(expr));
}

bool CUnit_visitor::pre_visit(Expr_call *expr) {
    return pre_visit(static_cast<Expr *>(expr));
}
void CUnit_visitor::post_visit(Expr_call *expr) {
    post_visit(static_cast<Expr *>(expr));
}

bool CUnit_visitor::pre_visit(Expr_compound *expr) {
    return pre_visit(static_cast<Expr *>(expr));
}
void CUnit_visitor::post_visit(Expr_compound *expr) {
    post_visit(static_cast<Expr *>(expr));
}

// ----------------------- statements -----------------------

bool CUnit_visitor::pre_visit(Stmt *stmt) { return true; }
void CUnit_visitor::post_visit(Stmt *stmt) {}

bool CUnit_visitor::pre_visit(Stmt_invalid *stmt) {
    return pre_visit(static_cast<Stmt *>(stmt));
}
void CUnit_visitor::post_visit(Stmt_invalid *stmt) {
    post_visit(static_cast<Stmt *>(stmt));
}

bool CUnit_visitor::pre_visit(Stmt_decl *stmt) {
    return pre_visit(static_cast<Stmt *>(stmt));
}
void CUnit_visitor::post_visit(Stmt_decl *stmt) {
    post_visit(static_cast<Stmt *>(stmt));
}

bool CUnit_visitor::pre_visit(Stmt_compound *stmt) {
    return pre_visit(static_cast<Stmt *>(stmt));
}
void CUnit_visitor::post_visit(Stmt_compound *stmt) {
    post_visit(static_cast<Stmt *>(stmt));
}

bool CUnit_visitor::pre_visit(Stmt_expr *stmt) {
    return pre_visit(static_cast<Stmt *>(stmt));
}
void CUnit_visitor::post_visit(Stmt_expr *stmt) {
    post_visit(static_cast<Stmt *>(stmt));
}

bool CUnit_visitor::pre_visit(Stmt_if *stmt) {
    return pre_visit(static_cast<Stmt *>(stmt));
}
void CUnit_visitor::post_visit(Stmt_if *stmt) {
    post_visit(static_cast<Stmt *>(stmt));
}

bool CUnit_visitor::pre_visit(Stmt_case *stmt) {
    return pre_visit(static_cast<Stmt *>(stmt));
}
void CUnit_visitor::post_visit(Stmt_case *stmt) {
    post_visit(static_cast<Stmt *>(stmt));
}

bool CUnit_visitor::pre_visit(Stmt_switch *stmt) {
    return pre_visit(static_cast<Stmt *>(stmt));
}
void CUnit_visitor::post_visit(Stmt_switch *stmt) {
    post_visit(static_cast<Stmt *>(stmt));
}

bool CUnit_visitor::pre_visit(Stmt_while *stmt) {
    return pre_visit(static_cast<Stmt *>(stmt));
}
void CUnit_visitor::post_visit(Stmt_while *stmt) {
    post_visit(static_cast<Stmt *>(stmt));
}

bool CUnit_visitor::pre_visit(Stmt_do_while *stmt) {
    return pre_visit(static_cast<Stmt *>(stmt));
}
void CUnit_visitor::post_visit(Stmt_do_while *stmt) {
    post_visit(static_cast<Stmt *>(stmt));
}

bool CUnit_visitor::pre_visit(Stmt_for *stmt) {
    return pre_visit(static_cast<Stmt *>(stmt));
}
void CUnit_visitor::post_visit(Stmt_for *stmt) {
    post_visit(static_cast<Stmt *>(stmt));
}

bool CUnit_visitor::pre_visit(Stmt_break *stmt) {
    return pre_visit(static_cast<Stmt *>(stmt));
}
void CUnit_visitor::post_visit(Stmt_break *stmt) {
    post_visit(static_cast<Stmt *>(stmt));
}

bool CUnit_visitor::pre_visit(Stmt_continue *stmt) {
    return pre_visit(static_cast<Stmt *>(stmt));
}
void CUnit_visitor::post_visit(Stmt_continue *stmt) {
    post_visit(static_cast<Stmt *>(stmt));
}

bool CUnit_visitor::pre_visit(Stmt_discard *stmt) {
    return pre_visit(static_cast<Stmt *>(stmt));
}
void CUnit_visitor::post_visit(Stmt_discard *stmt) {
    post_visit(static_cast<Stmt *>(stmt));
}

bool CUnit_visitor::pre_visit(Stmt_return *stmt) {
    return pre_visit(static_cast<Stmt *>(stmt));
}
void CUnit_visitor::post_visit(Stmt_return *stmt) {
    post_visit(static_cast<Stmt *>(stmt));
}

// ----------------------- declarations -----------------------

void CUnit_visitor::do_declaration(Declaration *decl)
{
    switch (decl->get_kind()) {
    case Declaration::DK_INVALID:
        {
            Declaration_invalid *idecl = cast<Declaration_invalid>(decl);
            do_invalid(idecl);
        }
        return;
    case Declaration::DK_VARIABLE:
        {
            Declaration_variable *vdecl = cast<Declaration_variable>(decl);
            do_variable_decl(vdecl);
        }
        return;
    case Declaration::DK_PARAM:
        {
            Declaration_param *pdecl = cast<Declaration_param>(decl);
            do_parameter(pdecl);
        }
        return;
    case Declaration::DK_FUNCTION:
        {
            Declaration_function *fdecl = cast<Declaration_function>(decl);
            do_function_decl(fdecl);
        }
        return;
    case Declaration::DK_FIELD:
        {
            Declaration_field *fdecl = cast<Declaration_field>(decl);
            do_field_decl(fdecl);
        }
        return;
    case Declaration::DK_STRUCT:
        {
            Declaration_struct *sdecl = cast<Declaration_struct>(decl);
            do_type_struct(sdecl);
        }
        return;
    case Declaration::DK_INTERFACE:
        {
            Declaration_interface *idecl = cast<Declaration_interface>(decl);
            do_interface(idecl);
        }
        return;
    case Declaration::DK_QUALIFIER:
        {
            Declaration_qualified *qdecl = cast<Declaration_qualified>(decl);
            do_qualified(qdecl);
        }
        return;
    }
    HLSL_ASSERT(!"Unsupported declaration kind");
}

void CUnit_visitor::do_invalid(Declaration_invalid *decl)
{
    pre_visit(decl);
    post_visit(decl);
}

void CUnit_visitor::do_variable_decl(Declaration_variable *decl)
{
    if (pre_visit(decl)) {
        Type_name *tname = decl->get_type_name();
        do_type_name(tname);

        for (Declaration_variable::iterator it(decl->begin()), end(decl->end()); it != end; ++it) {
            Init_declarator *init = it;
            do_init(init);
        }
    }
    post_visit(decl);
}

void CUnit_visitor::do_init(Init_declarator *init)
{
    if (pre_visit(init)) {
        Name *name = init->get_name();
        visit(name);

        Array_specifiers &as = init->get_array_specifiers();
        for (Array_specifiers::iterator it(as.begin()), end(as.end()); it != end; ++it) {
            Array_specifier *spec = it;

            do_array_specifier(spec);
        }

        if (Expr *expr = init->get_initializer())
            visit(expr);
    }
    post_visit(init);
}

void CUnit_visitor::do_parameter(Declaration_param *decl)
{
    if (pre_visit(decl)) {
        Type_name *tn = decl->get_type_name();
        do_type_name(tn);

        if (Name *name = decl->get_name())
            do_name(name);

        Array_specifiers &as = decl->get_array_specifiers();
        for (Array_specifiers::iterator it(as.begin()), end(as.end()); it != end; ++it) {
            Array_specifier *spec = it;

            do_array_specifier(spec);
        }

        if (Expr *init = decl->get_default_argument())
            do_expression(init);
    }
    post_visit(decl);
}

void CUnit_visitor::do_function_decl(Declaration_function *decl)
{
    if (pre_visit(decl)) {
        Type_name *rname = decl->get_ret_type();
        do_type_name(rname);

        Name *name = decl->get_identifier();
        do_name(name);

        for (Declaration_function::iterator it(decl->begin()), end(decl->end()); it != end; ++it) {
            Declaration *param = it;
            do_declaration(param);
        }

        if (Stmt *body = decl->get_body())
            do_statement(body);
    }
    post_visit(decl);
}

void CUnit_visitor::do_field_decl(Declaration_field *decl)
{
    if (pre_visit(decl)) {
        Type_name *tn = decl->get_type_name();
        do_type_name(tn);

        for (Declaration_field::iterator it(decl->begin()), end(decl->end()); it != end; ++it) {
            Field_declarator *field = it;
            do_declarator(field);
        }
    }
    post_visit(decl);
}

void CUnit_visitor::do_declarator(Field_declarator *field)
{
    if (pre_visit(field)) {
        Name *name = field->get_name();
        do_name(name);

        Array_specifiers &as = field->get_array_specifiers();
        for (Array_specifiers::iterator it(as.begin()), end(as.end()); it != end; ++it) {
            Array_specifier *spec = it;

            do_array_specifier(spec);
        }
    }
    post_visit(field);
}

void CUnit_visitor::do_type_struct(Declaration_struct *decl)
{
    if (pre_visit(decl)) {
        if (Name *name = decl->get_name())
            do_name(name);

        for (Declaration_struct::iterator it(decl->begin()), end(decl->end()); it != end; ++it) {
            Declaration *decl = it;
            do_declaration(decl);
        }
    }
    post_visit(decl);
}

void CUnit_visitor::do_interface(Declaration_interface *decl)
{
    if (pre_visit(decl)) {
        Type_qualifier *tq = &decl->get_qualifier();
        do_type_qualifier(tq);

        for (Declaration_interface::iterator it(decl->begin()), end(decl->end()); it != end; ++it)
        {
            Declaration *decl = it;
            do_declaration(decl);
        }

        if (Name *name = decl->get_identifier())
            do_name(name);

        Array_specifiers &as = decl->get_array_specifiers();
        for (Array_specifiers::iterator it(as.begin()), end(as.end()); it != end; ++it) {
            Array_specifier *spec = it;

            do_array_specifier(spec);
        }
    }
    post_visit(decl);
}

void CUnit_visitor::do_qualified(Declaration_qualified *decl)
{
    if (pre_visit(decl)) {
        Type_qualifier *tq = &decl->get_qualifier();
        do_type_qualifier(tq);

        for (Declaration_qualified::iterator it(decl->begin()), end(decl->end()); it != end; ++it)
        {
            Instance_name *name = it;
            do_instance_name(name);
        }
    }
    post_visit(decl);
}

void CUnit_visitor::do_name(Name *name)
{
    pre_visit(name);
    post_visit(name);
}

void CUnit_visitor::do_type_name(Type_name *tn)
{
    if (pre_visit(tn)) {
        Type_qualifier *tq = &tn->get_qualifier();
        do_type_qualifier(tq);

        if (Name *name = tn->get_name()) {
            do_name(name);
        } else if (Declaration *sdecl = tn->get_struct_decl()) {
            do_declaration(sdecl);
        }
    }
    post_visit(tn);
}

void CUnit_visitor::do_type_qualifier(Type_qualifier *tq)
{
    pre_visit(tq);
    post_visit(tq);
}

void CUnit_visitor::do_array_specifier(Array_specifier *spec)
{
    if (pre_visit(spec)) {
        if (Expr *expr = spec->get_size())
            do_expression(expr);
    }
    post_visit(spec);
}

void CUnit_visitor::do_instance_name(Instance_name *inst)
{
    if (pre_visit(inst)) {
        Name *name = inst->get_name();
        do_name(name);
    }
    post_visit(inst);

}

// ----------------------- expressions -----------------------

void CUnit_visitor::do_expression(Expr *expr)
{
    switch (expr->get_kind()) {
    case Expr::EK_INVALID:
        {
            Expr_invalid *e = cast<Expr_invalid>(expr);
            do_invalid_expression(e);
        }
        return;
    case Expr::EK_LITERAL:
        {
            Expr_literal *e = cast<Expr_literal>(expr);
            do_literal_expression(e);
        }
        return;
    case Expr::EK_REFERENCE:
        {
            Expr_ref *e = cast<Expr_ref>(expr);
            do_reference_expression(e);
        }
        return;
    case Expr::EK_UNARY:
        {
            Expr_unary *e = cast<Expr_unary>(expr);
            do_unary_expression(e);
        }
        return;
    case Expr::EK_BINARY:
        {
            Expr_binary *e = cast<Expr_binary>(expr);
            do_binary_expression(e);
        }
        return;
    case Expr::EK_CONDITIONAL:
        {
            Expr_conditional *e = cast<Expr_conditional>(expr);
            do_conditional_expression(e);
        }
        return;
    case Expr::EK_CALL:
        {
            Expr_call *e = cast<Expr_call>(expr);
            do_call_expression(e);
        }
        return;
    case Expr::EK_COMPOUND:
        {
            Expr_compound *e = cast<Expr_compound>(expr);
            do_compound_expression(e);
        }
        return;
    }
    HLSL_ASSERT(!"Unsupported expression kind");
}

void CUnit_visitor::do_invalid_expression(Expr_invalid *expr)
{
    pre_visit(expr);
    post_visit(expr);
}

void CUnit_visitor::do_literal_expression(Expr_literal *expr)
{
    pre_visit(expr);
    post_visit(expr);
}

void CUnit_visitor::do_reference_expression(Expr_ref *expr)
{
    if (pre_visit(expr)) {
        Type_name *name = expr->get_name();
        do_type_name(name);
    }
    post_visit(expr);
}

void CUnit_visitor::do_unary_expression(Expr_unary *expr)
{
    if (pre_visit(expr)) {
        Expr *op = expr->get_argument();
        do_expression(op);
    }
    post_visit(expr);
}

void CUnit_visitor::do_binary_expression(Expr_binary *expr)
{
    if (pre_visit(expr)) {
        Expr *l = expr->get_left_argument();
        do_expression(l);
        Expr *r = expr->get_right_argument();
        do_expression(r);
    }
    post_visit(expr);
}

void CUnit_visitor::do_conditional_expression(Expr_conditional *expr)
{
    if (pre_visit(expr)) {
        Expr *c = expr->get_condition();
        do_expression(c);
        Expr *t = expr->get_true();
        do_expression(t);
        Expr *f = expr->get_false();
        do_expression(f);
    }
    post_visit(expr);
}

void CUnit_visitor::do_call_expression(Expr_call *expr)
{
    if (pre_visit(expr)) {
        Expr *callee = expr->get_callee();
        do_expression(callee);

        for (size_t i = 0, n = expr->get_argument_count(); i < n; ++i) {
            Expr *arg = expr->get_argument(i);
            do_expression(arg);
        }
    }
    post_visit(expr);
}

void CUnit_visitor::do_compound_expression(Expr_compound *expr)
{
    if (pre_visit(expr)) {
        for (size_t i = 0, n = expr->get_element_count(); i < n; ++i) {
            Expr *elem = expr->get_element(i);
            do_expression(elem);
        }
    }
    post_visit(expr);
}

// ----------------------- statements -----------------------

void CUnit_visitor::do_statement(Stmt *stmt)
{
    switch (stmt->get_kind()) {
    case Stmt::SK_INVALID:
        {
            Stmt_invalid *s = cast<Stmt_invalid>(stmt);
            do_invalid_statement(s);
        }
        break;
    case Stmt::SK_COMPOUND:
        {
            Stmt_compound *s = cast<Stmt_compound>(stmt);
            do_compound_statement(s);
        }
        break;
    case Stmt::SK_DECLARATION:
        {
            Stmt_decl *s = cast<Stmt_decl>(stmt);
            do_declaration_statement(s);
        }
        break;
    case Stmt::SK_EXPRESSION:
        {
            Stmt_expr *s = cast<Stmt_expr>(stmt);
            do_expression_statement(s);
        }
        break;
    case Stmt::SK_IF:
        {
            Stmt_if *s = cast<Stmt_if>(stmt);
            do_if_statement(s);
        }
        break;
    case Stmt::SK_CASE:
        {
            Stmt_case *s = cast<Stmt_case>(stmt);
            do_case_statement(s);
        }
        break;
    case Stmt::SK_SWITCH:
        {
            Stmt_switch *s = cast<Stmt_switch>(stmt);
            do_switch_statement(s);
        }
        break;
    case Stmt::SK_WHILE:
        {
            Stmt_while *s = cast<Stmt_while>(stmt);
            do_while_statement(s);
        }
        break;
    case Stmt::SK_DO_WHILE:
        {
            Stmt_do_while *s = cast<Stmt_do_while>(stmt);
            do_do_while_statement(s);
        }
        break;
    case Stmt::SK_FOR:
        {
            Stmt_for *s = cast<Stmt_for>(stmt);
            do_for_statement(s);
        }
        break;
    case Stmt::SK_BREAK:
        {
            Stmt_break *s = cast<Stmt_break>(stmt);
            do_break_statement(s);
        }
        break;
    case Stmt::SK_CONTINUE:
        {
            Stmt_continue *s = cast<Stmt_continue>(stmt);
            do_continue_statement(s);
        }
        break;
    case Stmt::SK_DISCARD:
        {
            Stmt_discard *s = cast<Stmt_discard>(stmt);
            do_discard_statement(s);
        }
        break;
    case Stmt::SK_RETURN:
        {
            Stmt_return *s = cast<Stmt_return>(stmt);
            do_return_statement(s);
        }
        break;
    }
}

void CUnit_visitor::do_invalid_statement(Stmt_invalid *stmt)
{
    pre_visit(stmt);
    post_visit(stmt);
}

void CUnit_visitor::do_compound_statement(Stmt_compound *stmt)
{
    if (pre_visit(stmt)) {
        for (Stmt_compound::iterator it(stmt->begin()), end(stmt->end()); it != end; ++it) {
            Stmt *st = it;
            do_statement(st);
        }
    }
    post_visit(stmt);
}

void CUnit_visitor::do_declaration_statement(Stmt_decl *stmt)
{
    if (pre_visit(stmt)) {
        Declaration *decl = stmt->get_declaration();
        do_declaration(decl);
    }
    post_visit(stmt);
}

void CUnit_visitor::do_expression_statement(Stmt_expr *stmt)
{
    if (pre_visit(stmt)) {
        if (Expr *expr = stmt->get_expression())
            do_expression(expr);
    }
    post_visit(stmt);
}

void CUnit_visitor::do_if_statement(Stmt_if *stmt)
{
    if (pre_visit(stmt)) {
        Expr *cond = stmt->get_condition();
        do_expression(cond);

        Stmt *t = stmt->get_then_statement();
        do_statement(t);

        if (Stmt *e = stmt->get_else_statement())
            do_statement(e);
    }
    post_visit(stmt);
}

void CUnit_visitor::do_case_statement(Stmt_case *stmt)
{
    if (pre_visit(stmt)) {
        if (Expr *label = stmt->get_label()) {
            do_expression(label);
        }
    }
    post_visit(stmt);
}

void CUnit_visitor::do_switch_statement(Stmt_switch *stmt)
{
    Stmt_switch *s = const_cast<Stmt_switch *>(stmt);

    if (pre_visit(s)) {
        Expr *cond = s->get_condition();
        do_expression(cond);
    }
    post_visit(s);
}

void CUnit_visitor::do_while_statement(Stmt_while *stmt)
{
    if (pre_visit(stmt)) {
        Stmt *cond = stmt->get_condition();
        do_statement(cond);

        Stmt *body = stmt->get_body();
        do_statement(body);
    }
    post_visit(stmt);
}

void CUnit_visitor::do_do_while_statement(Stmt_do_while *stmt)
{
    if (pre_visit(stmt)) {
        Stmt *body = stmt->get_body();
        do_statement(body);

        Expr *cond = stmt->get_condition();
        do_expression(cond);
    }
    post_visit(stmt);
}

void CUnit_visitor::do_for_statement(Stmt_for *stmt)
{
    if (pre_visit(stmt)) {
        if (Stmt *init = stmt->get_init())
            do_statement(init);

        if (Stmt *cond = stmt->get_condition())
            do_statement(cond);

        if (Expr *upd = stmt->get_update())
            do_expression(upd);

        Stmt *body = stmt->get_body();
        do_statement(body);
    }
    post_visit(stmt);
}

void CUnit_visitor::do_break_statement(Stmt_break *stmt)
{
    pre_visit(stmt);
    post_visit(stmt);
}

void CUnit_visitor::do_continue_statement(Stmt_continue *stmt)
{
    pre_visit(stmt);
    post_visit(stmt);
}

void CUnit_visitor::do_discard_statement(Stmt_discard *stmt)
{
    pre_visit(stmt);
    post_visit(stmt);
}

void CUnit_visitor::do_return_statement(Stmt_return *stmt)
{
    if (pre_visit(stmt)) {
        if (Expr *expr = stmt->get_expression())
            do_expression(expr);
    }
    post_visit(stmt);
}

}  // hlsl
}  // mdl
}  // mi
