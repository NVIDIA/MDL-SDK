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

#include "compiler_hlsl_compilation_unit.h"
#include "compiler_hlsl_optimizer.h"

namespace mi {
namespace mdl {
namespace hlsl {

// Constructor.
Optimizer::Optimizer(
    IAllocator       *alloc,
    Compiler         &compiler,
    Compilation_unit &unit,
    int              opt_level)
: m_alloc(alloc)
, m_compiler(compiler)
, m_unit(unit)
, m_sf(unit.get_statement_factory())
, m_ef(unit.get_expression_factory())
, m_df(unit.get_declaration_factory())
, m_value_factory(unit.get_value_factory())
, m_opt_level(opt_level)
{
}

// Run the optimizer on this compilation unit.
void Optimizer::run(
    IAllocator       *alloc,
    Compiler         &compiler,
    Compilation_unit &unit,
    int              opt_level)
{
    if (unit.get_messages().get_error_message_count() > 0)
        return;

    if (opt_level == 0) {
        // all optimizations are switched off
        return;
    }

    Optimizer opt(alloc, compiler, unit, opt_level);

    opt.local_opt();
}

// Checks if two given expressions are semantically the same.
bool Optimizer::same_expr(Expr *a, Expr *b) const
{
    Expr::Kind k = a->get_kind();

    if (k != b->get_kind())
        return false;

    switch (k) {
    case Expr::EK_INVALID:
        return true;
    case Expr::EK_LITERAL:
        {
            Expr_literal *ca = cast<Expr_literal>(a);
            Expr_literal *cb = cast<Expr_literal>(b);

            return ca->get_value() == cb->get_value();
        }
    case Expr::EK_REFERENCE:
        {
            Expr_ref *ra = cast<Expr_ref>(a);
            Expr_ref *rb = cast<Expr_ref>(b);

            return ra->get_definition()== rb->get_definition();
        }
    case Expr::EK_UNARY:
        {
            Expr_unary *ua = cast<Expr_unary>(a);
            Expr_unary *ub = cast<Expr_unary>(b);

            if (ua->get_operator() != ub->get_operator())
                return false;
            return same_expr(ua->get_argument(), ub->get_argument());
        }
    case Expr::EK_BINARY:
        {
            Expr_binary *ba = cast<Expr_binary>(a);
            Expr_binary *bb = cast<Expr_binary>(b);

            if (ba->get_operator() != bb->get_operator())
                return false;
            if (!same_expr(ba->get_left_argument(), bb->get_left_argument()))
                return false;
            return same_expr(ba->get_right_argument(), bb->get_right_argument());
        }
    case Expr::EK_CONDITIONAL:
        {
            Expr_conditional *ca = cast<Expr_conditional>(a);
            Expr_conditional *cb = cast<Expr_conditional>(b);

            if (!same_expr(ca->get_condition(), cb->get_condition()))
                return false;
            if (!same_expr(ca->get_true(), cb->get_true()))
                return false;
            return same_expr(ca->get_false(), cb->get_false());
        }
    case Expr::EK_CALL:
        // NYI
        return false;
    case Expr::EK_COMPOUND:
        // NYI
        return false;
    }
    HLSL_ASSERT(!"unsupported expression kind");
    return false;
}

// Optimize vector constructors.
Expr *Optimizer::optimize_vector_constructor(Expr_call *constr)
{
    Expr_binary *arg0 = as<Expr_binary>(constr->get_argument(0));
    if (arg0 == NULL) {
        return NULL;
    }

    if (arg0->get_operator() != Expr_binary::OK_SELECT) {
        return NULL;
    }

    size_t n_args = constr->get_argument_count();

    Small_VLA<Symbol *, 4> symbols(m_alloc, n_args);

    Expr *base = arg0->get_left_argument();
    symbols[0] = cast<Expr_ref>(arg0->get_right_argument())->get_name()->get_name()->get_symbol();

    Type_vector *v_type = as<Type_vector>(base->get_type());
    if (v_type == NULL) {
        return NULL;
    }

    for (size_t i = 1; i < n_args; ++i) {
        Expr_binary *arg = as<Expr_binary>(constr->get_argument(i));

        if (arg == NULL) {
            return NULL;
        }

        if (arg0->get_operator() != Expr_binary::OK_SELECT) {
            return NULL;
        }

        Expr *left = arg->get_left_argument();

        if (!same_expr(base, left))
            return NULL;

        symbols[i] =
            cast<Expr_ref>(arg->get_right_argument())->get_name()->get_name()->get_symbol();
    }

    string swizzle(m_alloc);
    for (size_t i = 0; i < n_args; ++i) {
        swizzle.append(symbols[i]->get_name());
    }

    bool need_swizzle = true;

    size_t l = v_type->get_size();
    if (l <= 4 && l == swizzle.length() && strncmp(swizzle.c_str(), "xyzw", l) == 0) {
        need_swizzle = false;
    }

    if (need_swizzle) {
        Symbol *sym_swizzle = m_unit.get_symbol_table().get_symbol(swizzle.c_str());

        Location const &loc = constr->get_location();

        Type_name *tn = m_df.create_type_name(loc);
        tn->set_name(m_df.create_name(loc, sym_swizzle));

        Expr *r_swizzle = m_ef.create_reference(tn);

        Expr *res = m_ef.create_binary(Expr_binary::OK_SELECT, base, r_swizzle);
        res->set_type(constr->get_type());

        return res;
    } else {
        // swizzle not needed
        return base;
    }
}

// Optimize calls.
Expr *Optimizer::optimize_call(Expr_call *call)
{
    Expr_ref *callee = as<Expr_ref>(call->get_callee());
    if (callee == NULL)
        return NULL;

    Definition *def = callee->get_definition();
    if (def == NULL) {
        // bad, there should be one
        return NULL;
    }

    Def_function *fdef = as<Def_function>(def);
    if (fdef == NULL)
        return NULL;

    if (fdef->get_semantics() == Def_function::DS_ELEM_CONSTRUCTOR) {
        Type *res_tp = call->get_type();

        if (is<Type_vector>(res_tp->skip_type_alias())) {
            // vector(a.x, a.y, ...) ==> a.xy...
            if (Expr *res = optimize_vector_constructor(call))
                return res;
        }
    }
    return NULL;
}


// Execute a function on the bodies of all HLSL functions.
void Optimizer::run_on_function(
    Stmt *(Optimizer::* body_func)(Stmt *),
    Expr *(Optimizer::* expr_func)(Expr *))
{
    for (Compilation_unit::iterator it(m_unit.decl_begin()), end(m_unit.decl_end());
         it != end;
         ++it)
    {
        Declaration *decl = it;

        if (Declaration_function *fdecl = as<Declaration_function>(decl)) {
            // optimize the function body
            if (Stmt *body = fdecl->get_body()) {
                Stmt *n_body = (this->*body_func)(body);
                if (n_body != NULL) {
                    fdecl->set_body(n_body);
                } else {
                    // The body would be empty, but this is not supported.
                    // Note that the body is a compound statement for functions
                    // and it is cleared in that case, so do nothing here.
                }
            }
        }
    }
}

// Creates an unary expression.
Expr *Optimizer::create_unary(
    Expr_unary::Operator op,
    Expr                *arg,
    Location const      &pos)
{
    Expr *res = m_ef.create_unary(pos, op, arg);

    // FIXME: find the operator to check for the result type
    res->set_type(arg->get_type());

    return res;
}

// Run local optimizations.
Declaration *Optimizer::local_opt(Declaration *decl)
{
    if (Declaration_variable *vdecl = as<Declaration_variable>(decl)) {
        for (hlsl::Init_declarator &idecl : *vdecl) {
            if (Expr *expr = idecl.get_initializer()) {
                Expr *n_expr = local_opt(expr);
                if (expr != n_expr) {
                    idecl.set_initializer(n_expr);
                }
            }
        }
        return decl;
    }

    return NULL;
}

// Run local optimizations.
Stmt *Optimizer::local_opt(Stmt *stmt)
{
    Stmt *res = stmt;

    switch (stmt->get_kind()) {
    case Stmt::SK_INVALID:
        // no optimization possible, but should not occur in a valid compilation unit
        HLSL_ASSERT(!"invalid statement occurred in a valid compilation unit");
        break;

    case Stmt::SK_COMPOUND:
        {
            Stmt_compound *c_smtm = cast<Stmt_compound>(stmt);

            if (c_smtm->size() == 0) {
                // drop the empty block
                return NULL;
            }

            for (Stmt_compound::iterator it(c_smtm->begin()), end(c_smtm->end()); it != end;) {
                Stmt *s = it;
                Stmt *n = local_opt(s);

                if (n != s) {
                    // FIXME: replace
                    ++it;
                } else {
                    ++it;
                }
            }
            return c_smtm;
        }

    case Stmt::SK_DECLARATION:
        {
            Stmt_decl    *decl_stmt = cast<Stmt_decl>(stmt);
            Declaration  *decl = decl_stmt->get_declaration();
            Declaration  *n_decl = local_opt(decl);

            if (n_decl == NULL)
                return NULL;
            if (n_decl != decl)
                decl_stmt->set_declaration(n_decl);
            return decl_stmt;
        }

    case Stmt::SK_EXPRESSION:
        {
            Stmt_expr *e_stmt = cast<Stmt_expr>(stmt);
            Expr      *expr   = e_stmt->get_expression();

            if (expr != NULL) {
                Expr *n_expr = local_opt(expr);
                if (n_expr != expr)
                    e_stmt->set_expression(n_expr);
            } else {
                // useless
                return NULL;
            }
            return e_stmt;
        }

    case Stmt::SK_IF:
        {
            Stmt_if *if_stmt   = cast<Stmt_if>(stmt);
            Expr    *cond      = if_stmt->get_condition();
            Stmt    *then_stmt = if_stmt->get_then_statement();
            Stmt    *else_stmt = if_stmt->get_else_statement();

            Expr *n_cond = local_opt(cond);
            if (n_cond != cond)
                if_stmt->set_condition(cond);

            if (Expr_literal *lit = as<Expr_literal>(n_cond)) {
                Value_bool *val = cast<Value_bool>(lit->get_value());

                if (val->get_value())
                    return local_opt(then_stmt);
                else
                    return else_stmt != NULL ? local_opt(else_stmt) : NULL;
            }
            Stmt *n_then = local_opt(then_stmt);
            Stmt *n_else = else_stmt != NULL ? local_opt(else_stmt) : NULL;

            if (n_then == NULL && n_else == NULL) {
                // both branches are empty, preserve just the condition
                Location const &loc = if_stmt->get_location();
                Stmt *n_stmt = m_sf.create_expression(loc, n_cond);

                return local_opt(n_stmt);
            }

            if (n_then == NULL) {
                // no then but else
                Location const &e_loc = n_cond->get_location();
                Expr *neg = create_unary(
                    Expr_unary::OK_LOGICAL_NOT,
                    n_cond,
                    e_loc);

                neg = local_opt(neg);

                if_stmt->set_condition(neg);
                n_then = n_else;
                n_else = NULL;
            }
            if_stmt->set_then_statement(n_then);
            if_stmt->set_else_statement(n_else);

            return if_stmt;
        }

    case Stmt::SK_CASE:
        // do nothing, just a label
        break;

    case Stmt::SK_SWITCH:
        {
            Stmt_switch *s_smtm = cast<Stmt_switch>(stmt);

            for (Stmt_compound::iterator it(s_smtm->begin()), end(s_smtm->end()); it != end;) {
                Stmt *s = it;
                Stmt *n = local_opt(s);

                if (n != s) {
                    // FIXME: replace
                    ++it;
                } else {
                    ++it;
                }
            }
            return s_smtm;
        }

    case Stmt::SK_WHILE:
        {
            Stmt_while *loop_stmt = cast<Stmt_while>(stmt);
            Stmt       *cond      = local_opt(loop_stmt->get_condition());

            loop_stmt->set_condition(cond);

            if (Stmt_expr *cond_expr = as<Stmt_expr>(cond)) {
                if (Expr_literal *lit = as<Expr_literal>(cond_expr->get_expression())) {
                    Value_bool *v = cast<Value_bool>(lit->get_value());

                    if (v->get_value()) {
                        // endless loop
                    } else {
                        // not executed, removed
                        return NULL;
                    }
                }
            }
            Stmt *body   = loop_stmt->get_body();
            Stmt *n_body = local_opt(body);
            if (n_body == NULL) {
                // while body cannot be empty
                Location const &loc = body->get_location();
                n_body = m_sf.create_expression(loc, NULL);
            }
            loop_stmt->set_body(n_body);
            return loop_stmt;
        }

    case Stmt::SK_DO_WHILE:
        {
            Stmt_do_while *loop_stmt = cast<Stmt_do_while>(stmt);
            Expr *cond = local_opt(loop_stmt->get_condition());

            loop_stmt->set_condition(cond);

            if (Expr_literal *lit = as<Expr_literal>(cond)) {
                Value_bool *v = cast<Value_bool>(lit->get_value());

                if (v->get_value()) {
                    // endless loop
                } else {
                    // body executed once, replace the loop by the body
                    return local_opt(loop_stmt->get_body());
                }
            }
            Stmt *body = loop_stmt->get_body();
            Stmt *n_body = local_opt(body);
            if (n_body == NULL) {
                // do-while body cannot be empty
                Location const &loc = body->get_location();
                n_body = m_sf.create_expression(loc, NULL);
            }
            loop_stmt->set_body(n_body);
            return loop_stmt;
        }

    case Stmt::SK_FOR:
        {
            Stmt_for *for_stmt = cast<Stmt_for>(stmt);
            Stmt     *cond     = for_stmt->get_condition();

            if (cond != NULL) {
                cond = local_opt(cond);
                for_stmt->set_condition(cond);

                if (Stmt_expr *cond_expr = as<Stmt_expr>(cond)) {
                    if (Expr_literal *lit = as<Expr_literal>(cond_expr->get_expression())) {
                        Value_bool *v = cast<Value_bool>(lit->get_value());

                        if (v->get_value()) {
                            // endless loop, remove the condition at all
                            for_stmt->set_condition(NULL);
                        } else {
                            // loop is not executed, only the init statement will prevail
                            Stmt *init = for_stmt->get_init();

                            if (is<Stmt_decl>(init)) {
                                // the for statement created a scope around the if, so we need a
                                // block now
                                Location const &loc = for_stmt->get_location();
                                Stmt_compound *n_init = m_sf.create_compound(loc, init);
                                init = n_init;
                            }

                            return init != NULL ? local_opt(init) : NULL;
                        }
                    }
                }
            }

            Stmt *body   = for_stmt->get_body();
            Stmt *n_body = local_opt(body);
            if (n_body == NULL) {
                // for body cannot be empty
                Location const &loc = body->get_location();
                n_body = m_sf.create_expression(loc, NULL);
            }
            for_stmt->set_body(body);

            Expr *next = for_stmt->get_update();
            if (next != NULL) {
                next = local_opt(next);
                for_stmt->set_update(next);
            }
            return for_stmt;

        }
        break;

    case Stmt::SK_BREAK:
    case Stmt::SK_CONTINUE:
    case Stmt::SK_DISCARD:
        // do nothing
        return stmt;

    case Stmt::SK_RETURN:
        {
            Stmt_return *r_stmt = cast<Stmt_return>(stmt);
            Expr        *expr   = r_stmt->get_expression();

            if (expr != NULL) {
                Expr *n_expr = local_opt(expr);
                if (n_expr != expr)
                    r_stmt->set_expression(n_expr);
            }
            return r_stmt;
        }
    }
    return res;
}

// Run local optimizations.
Expr *Optimizer::local_opt(Expr *expr)
{
    Expr *res = expr;

    switch (expr->get_kind()) {
    case Expr::EK_INVALID:
        // no optimization possible, but should not occur in a valid compilation unit
        HLSL_ASSERT(!"invalid expression occurred in a valid compilation unit");
        break;
    case Expr::EK_LITERAL:
    case Expr::EK_REFERENCE:
        // do nothing
        break;
        case Expr::EK_UNARY:
        {
            Expr_unary *unary = cast<Expr_unary>(expr);
            Expr       *arg   = local_opt(unary->get_argument());

            unary->set_argument(arg);
        }
        break;
    case Expr::EK_BINARY:
        {
            Expr_binary *binary = cast<Expr_binary>(expr);

            Expr *lhs = local_opt(binary->get_left_argument());
            Expr *rhs = local_opt(binary->get_right_argument());

            binary->set_left_argument(lhs);
            binary->set_right_argument(rhs);
        }
        break;
    case Expr::EK_CONDITIONAL:
        {
            Expr_conditional *c_expr = cast<Expr_conditional>(expr);
            Expr             *cond = local_opt(c_expr->get_condition());

            if (Expr_literal *lit = as<Expr_literal>(cond)) {
                Value_bool *b = cast<Value_bool>(lit->get_value());

                if (b->get_value())
                    return local_opt(c_expr->get_true());
                else
                    return local_opt(c_expr->get_false());
            }

            Expr *t_ex = local_opt(c_expr->get_true());
            Expr *f_ex = local_opt(c_expr->get_false());

            if (same_expr(t_ex, f_ex)) {
                // cond ? c : c => c
                return t_ex;
            }

            if (is<Type_bool>(c_expr->get_type()->skip_type_alias())) {
                if (Expr_literal *lit_t = as<Expr_literal>(t_ex)) {
                    if (Expr_literal *lit_f = as<Expr_literal>(f_ex)) {
                        Value_bool *v_t = cast<Value_bool>(lit_t->get_value());
                        Value_bool *v_f = cast<Value_bool>(lit_f->get_value());

                        if (v_t->get_value()) {
                            // cond ? true : false => cond
                            return cond;
                        }
                        if (v_f->get_value()) {
                            // cond ? false : true => !cond
                            return create_unary(
                                Expr_unary::OK_LOGICAL_NOT,
                                cond,
                                c_expr->get_location());
                        }
                    }
                }
            }

            c_expr->set_condition(cond);
            c_expr->set_true(t_ex);
            c_expr->set_false(f_ex);
        }
        break;
    case Expr::EK_CALL:
        if (Expr *res = optimize_call(cast<Expr_call>(expr)))
            return res;
        break;
    case Expr::EK_COMPOUND:
        // NYI
        break;
    }
    return res;
}

// Run local optimizations.
void Optimizer::local_opt()
{
    run_on_function(&Optimizer::local_opt, &Optimizer::local_opt);
}

}  // hlsl
}  // mdl
}  // mi
