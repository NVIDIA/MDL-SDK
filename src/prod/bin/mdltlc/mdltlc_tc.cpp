/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include <sstream>

#include "mdltlc_values.h"
#include "mdltlc_analysis.h"
#include "mdltlc_compilation_unit.h"

/// Typecheck the AST of the mdltl file.
void Compilation_unit::type_check(Environment &builtin_env) {
    for (mi::mdl::Ast_list<Ruleset>::iterator it(m_rulesets.begin()),
             end(m_rulesets.end());
         it != end; ++it) {
        type_check_ruleset(*it, builtin_env);
    }
}

/// Typecheck the AST for a ruleset.
void Compilation_unit::type_check_ruleset(Ruleset &ruleset, Environment &env) {

    if (m_comp_options->get_verbosity() >= 3) {
        printf("[info] Checking ruleset %s...\n", ruleset.get_name());
    }

    for (mi::mdl::Ast_list<Rule>::iterator it(ruleset.m_rules.begin()),
             end(ruleset.m_rules.end());
         it != end;
         ++it) {
        type_check_rule(*it, env);
    }

    for (mi::mdl::Ast_list<Rule>::iterator it(ruleset.m_rules.begin()),
             end(ruleset.m_rules.end());
         it != end;
         ++it) {
        check_attr_types_determined(this, it->get_lhs());
        check_attr_types_determined(this, it->get_rhs());
        if (ruleset.get_strategy() == Ruleset::Strategy::STRAT_TOPDOWN)
            check_topdown_attrs(this, it->get_rhs());

    }

    type_check_postcond(ruleset.get_postcond(), env);
}

bool Compilation_unit::is_material_struct(Type const *t) {
    if (Type_struct const *ts = as<Type_struct>(t)) {
        if (ts->get_name() == m_symbol_table->get_symbol("material")) {
            return true;
        }
        return false;
    }
    return false;
}

bool Compilation_unit::is_material_or_bsdf(Type const *t) {
    return is<Type_bsdf>(t)
        || is<Type_vdf>(t)
        || is<Type_edf>(t)
        || is<Type_hair_bsdf>(t)
        || is<Type_material>(t);
}

/// Typecheck the AST for a rule.
void Compilation_unit::type_check_rule(Rule &rule, Environment &global_env) {

    Environment env(m_arena, Environment::Kind::ENV_LOCAL, &global_env);

    Type *t_lhs = deref(type_check_pattern(rule.get_lhs(), env));
    type_check_where(rule.get_bindings(), env);

    Type *t_rhs = deref(type_check_expr(rule.get_rhs(), env));


    type_check_guard(rule.get_guard(), env);

    if (m_comp_options->get_verbosity() >= 4) {
        dump_env(env);
        dump_env(m_attribute_env);
    }

    if (is<Type_error>(t_lhs) || is<Type_error>(t_rhs)) {
        // No need to make further checks if one of the types is an
        // error type.
        return;
    }

    if (is<Type_var>(t_lhs) || is<Type_var>(t_rhs)) {
        error(rule.get_location(),
              "the type of this rule is not fully determined");
        return;
    }

    if (!is_material_or_bsdf(t_lhs)) {
        error(rule.get_location(),
              "the left-hand side of a rule must be a bsdf or a material");
        return;
    }

    if (!is_material_or_bsdf(t_rhs)) {
        error(rule.get_location(),
              "the right-hand side of a rule must be a bsdf or a material");
        return;
    }

    if (!m_type_factory.types_equal(t_lhs, t_rhs)) {
        error(rule.get_location(),
              "both sides of a rule must have the same type");
    }

}

Type *Compilation_unit::type_check_reference(
    Expr *expr,
    Environment &env,
    Pattern_context pattern_ctx)
{
    if (!is<Expr_ref>(expr)) {
        error(expr->get_location(), "reference expected");
        expr->set_type(m_error_type);
        return m_error_type;
    }
    Expr_ref *e = cast<Expr_ref>(expr);
    Symbol const *name = e->get_name();

    // Each wildcard get a fresh type variable to make it match any
    // type.

    if (!strcmp(name->get_name(), "_")) {
        return m_type_factory.create_type_variable();
    }

    Environment *binding_env = nullptr;
    Environment::Type_list *types = env.find(name, &binding_env);

    if (pattern_ctx == Pattern_context::PC_PATTERN) {
        // If there is no binding, or it is empty, or the name is
        // bound in an enclosing environment, then we bind it.

        if (!types || types->size() == 0 || binding_env != &env) {
            Type_var *tv = m_type_factory.create_type_variable();
            env.bind(name, tv);
            expr->set_type(tv);
            return tv;
        }

        // Otherwise, the name is already bound in the topmost
        // environment, which is illegal.

        mi::mdl::string s_msg(m_arena.get_allocator());
        s_msg = "duplicate definition: ";
        s_msg += name->get_name();
        error(expr->get_location(), s_msg.c_str());
        Type *err_type = m_error_type;
        expr->set_type(err_type);
        return err_type;
    }

    // In expression context from here on...

    if (!types) {
        mi::mdl::string s_msg(m_arena.get_allocator());
        s_msg = "unknown name: ";
        s_msg += name->get_name();
        error(expr->get_location(), s_msg.c_str());
        expr->set_type(m_error_type);
        return m_error_type;
    }

    // types guaranteed to be non-null here.

    if (types->size() == 1) {
        Type *ref_type = types->front();

        expr->set_type(ref_type);
        return ref_type;
    } else {
        mi::mdl::string s_msg(m_arena.get_allocator());
        s_msg = "ambiguous type for reference: ";
        s_msg += name->get_name();
        error(expr->get_location(), s_msg.c_str());
        Type *err_type = m_error_type;
        expr->set_type(err_type);
        return err_type;
    }
}

bool Compilation_unit::check_reference_exists(Expr *expr, Environment &env) {
    if (!is<Expr_ref>(expr)) {
        error(expr->get_location(), "reference expected");
        expr->set_type(m_error_type);
        return false;
    }
    Expr_ref *e = cast<Expr_ref>(expr);
    Symbol const *name = e->get_name();

    Environment::Type_list *types = env.find(name);

    if (!types || types->size() == 0) {
        mi::mdl::string s_msg(m_arena.get_allocator());
        s_msg = "unknown name for reference: ";
        s_msg += name->get_name();
        error(expr->get_location(), s_msg.c_str());
        expr->set_type(m_error_type);
        return false;
    }

    Type *t = types->front();
    expr->set_type(t);
    return true;
}

void Compilation_unit::generate_overload_hints(Expr *expr, Expr_call *call_expr, Environment::Type_list &ts, Mdl_type_vector &arg_types) {
    mi::mdl::string s_msg(get_allocator());
    s_msg = "\tin call: ";
    {
        std::stringstream out;
        pp::Pretty_print p(m_arena, out);
        call_expr->pp(p);
        s_msg += out.str().c_str();
    }

    hint(expr->get_location(), s_msg.c_str());

    s_msg = "\twith argument types: ";

    {
        std::stringstream out;
        pp::Pretty_print p(m_arena, out);
        expr->pp(p);
        s_msg += out.str().c_str();
    }
    s_msg += "(";

    bool first = true;
    for (Mdl_type_vector::iterator it(arg_types.begin()), end(arg_types.end());
         it != end; ++it) {
        Type *arg_type = *it;

        if (first)
            first = false;
        else
            s_msg += ", ";
        {
            std::stringstream out;
            pp::Pretty_print p(m_arena, out);
            arg_type->pp(p);
            s_msg += out.str().c_str();
        }
    }
    s_msg += ")";
    hint(expr->get_location(), s_msg.c_str());

    hint(expr->get_location(), "the following overloads were considered:");
    for (Environment::Type_list::iterator it(ts.begin()), end(ts.end());
         it != end; ++it) {
        s_msg = "\t";
        {
            std::stringstream out;
            pp::Pretty_print p(m_arena, out);
            (*it)->pp(p);
            s_msg += out.str().c_str();
        }

        hint(expr->get_location(), s_msg.c_str());
    }
}

/// Resolve the type of the reference `name` that appears in
/// expression `expr.  The type list `ts` contains all the types that
/// have been found for `name` in the environment. `arg_types` is a
/// list of the actual argument types used in the call to `name`, and
/// are used to filter the types in `ts`.
///
/// Returns a type set if there are multiple matches for the actual
/// arguments, a non-set type if the match is unique or the error type
/// if no matching overload was found.
Type *Compilation_unit::resolve_overload(Symbol const *name, Expr *expr, Expr_call *call_expr, Environment::Type_list *ts, Mdl_type_vector &arg_types) {
    Type *result = nullptr;

    // We collect all types that match the argument list to generate
    // hints to the user if overload resolution results in
    // ambiguities.
    Environment::Type_list matched_types(get_allocator());

    // We also collect all function types for the given identifier, to
    // generate hints in case no matching function is found at all.
    Environment::Type_list function_types(get_allocator());

    for (Environment::Type_list::iterator it(ts->begin()), end(ts->end());
         it != end; ++it) {

        Type *t = *it;

        if (Type_function *tf = as<Type_function>(t)) {
            function_types.push_back(t);
            if (tf->get_parameter_count() == arg_types.size()) {
                bool mismatch = false;
                int i = 0;

                for (Mdl_type_vector::iterator it(arg_types.begin()), end(arg_types.end());
                     it != end; ++it, ++i) {
                    Type *arg_type = *it;
                    Type *param_type = tf->get_parameter_type(i);

                    if (!m_type_factory.types_match(arg_type, param_type)) {
                        mismatch = true;
                        break;
                    }
                }
                if (!mismatch) {
                    matched_types.push_back(t);

                    if (!result) {
                        result = t;
                    }
                }
            }
        }
    }

    // We have multiple matches. That means that no unique overload
    // was found.
    if (matched_types.size() > 1) {
        mi::mdl::string s_msg(m_arena.get_allocator());
        s_msg = "no unique definition found for called reference: ";
        s_msg += name->get_name();
        error(expr->get_location(), s_msg.c_str());

        generate_overload_hints(expr, call_expr, matched_types, arg_types);

        return m_type_factory.create_type_variable();
    }

    // We don't have any match. This means that the identifier either
    // is not defined, or it is not the name of a function.
    if (!result) {
        mi::mdl::string s_msg(m_arena.get_allocator());
        s_msg = "no matching overload found for called reference: ";
        s_msg += name->get_name();
        error(expr->get_location(), s_msg.c_str());

        if (function_types.size() > 0)
            generate_overload_hints(expr, call_expr, function_types, arg_types);

        return m_error_type;
    }

    return result;
}

/// Type check a reference expression that is in call position.
///
/// The vector `arg_types` holds the inferred types of the actual call
/// arguments and are used in overload resolution.
Type *Compilation_unit::type_check_called_reference(Expr *expr, Expr_call *call_expr, Mdl_type_vector &arg_types, Environment &env) {
    if (!is<Expr_ref>(expr)) {
        error(expr->get_location(), "reference expected");
        expr->set_type(m_error_type);
        return m_error_type;
    }
    Expr_ref *e = cast<Expr_ref>(expr);
    Symbol const *name = e->get_name();

    Environment::Type_list *types = env.find(name);

    if (!types || types->size() == 0) {
        mi::mdl::string s_msg(m_arena.get_allocator());
        s_msg = "unknown name for called reference: ";
        s_msg += name->get_name();
        error(expr->get_location(), s_msg.c_str());
        expr->set_type(m_error_type);
        return m_error_type;
    } else {
        Type *candidate = resolve_overload(name, expr, call_expr, types, arg_types);

        expr->set_type(candidate);
        return candidate;
    }
}

Type *Compilation_unit::type_check_attribute(Expr *expr, Expr_attribute *e, Environment &env,
                                             Pattern_context pattern_ctx) {

    Type *t_arg = deref(pattern_ctx == Pattern_context::PC_EXPRESSION
                        ? type_check_expr(e->get_argument(), env)
                        : type_check_pattern(e->get_argument(), env));

    e->get_argument()->set_type(t_arg);

    Var_set defined_attrs(m_arena.get_allocator());

    Expr_attribute::Expr_attribute_vector &attrs = e->get_attributes();
    for (size_t i = 0; i < attrs.size(); i++) {
        Expr_attribute::Expr_attribute_entry &p = attrs[i];

        // Remember best error location here, might be used below
        // in multiple places.
        Location const &loc = p.expr ? p.expr->get_location() : expr->get_location();

        // Check that each attribute name only appears once per attribute set.
        if (defined_attrs.find(p.name) != defined_attrs.end()) {
            mi::mdl::string s_msg(m_arena.get_allocator());
            s_msg = "duplicate attribute in attribute set: ";
            s_msg += p.name->get_name();
            error(loc, s_msg.c_str());
        }
        defined_attrs.insert(p.name);

        if (p.is_pattern != (pattern_ctx == Pattern_context::PC_PATTERN)) {
            if (pattern_ctx == Pattern_context::PC_PATTERN)
                error(loc, "attributes in patterns must use the `~` operator");
            else
                error(loc, "attributes in expressions must use the `=` operator");
        }

        Type *t = p.expr
            ? deref(pattern_ctx == Pattern_context::PC_EXPRESSION
                    ? type_check_expr(p.expr, env)
                    : type_check_pattern(p.expr, env))
            : p.type;

        if (is<Type_error>(t)) {
            t_arg = m_error_type;
        }

        t = deref(t);

        // Check whether the attribute name is already defined as
        // a regular pattern variable.
        Environment *binding_env = nullptr;
        Environment::Type_list *env_types = env.find(p.name, &binding_env);
        Environment::Type_list *attr_types = m_attribute_env.find(p.name);

        if (env_types && (!attr_types || attr_types->size() == 0) && binding_env == &env) {
            mi::mdl::string s_msg(m_arena.get_allocator());
            s_msg = "variable `";
            s_msg += p.name->get_name();
            s_msg += "` is already bound to a pattern variable";
            error(loc, s_msg.c_str());
            t_arg = m_error_type;
            t = m_error_type;
        } else {
            if (!attr_types || attr_types->size() == 0) {
                m_attribute_env.bind(p.name, t);
            } else {
                Type *other_t = deref(attr_types->front());
                if (Type_var *t_tv = as<Type_var>(t)) {
                    if (!is<Type_var>(other_t)) {
                        t_tv->assign_type(other_t, m_type_factory);
                    }
                } else if (Type_var *tv = as<Type_var>(other_t)) {
                    tv->assign_type(t, m_type_factory);
                } else if (is<Type_error>(other_t) || is<Type_error>(t)) {
                    // Error was already emitted.
                } else if (!m_type_factory.types_equal(t, other_t)) {
                    mi::mdl::string s_msg(m_arena.get_allocator());
                    s_msg = "type mismatch for attribute: ";
                    s_msg += p.name->get_name();
                    error(loc, s_msg.c_str());
                    t_arg = m_error_type;
                    t = m_error_type;
                }
            }
            if (!env_types || env_types->size() == 0)
                env.bind(p.name, t);
        }
        p.type = t;
        if (p.expr)
            p.expr->set_type(t);
    }
    return t_arg;
}

/// Type check a pattern. This is used for checking the LHS of an
/// expression, and also for `match` expressions.
///
/// The most important difference between `type_check_pattern` and
/// `type_check_expr` is that patterns are syntactically restricted
/// expressions, and that patterns can bind variables by extending the
/// environment passed in as `env`.
Type *Compilation_unit::type_check_pattern(Expr *expr, Environment &env) {
    switch (expr->get_kind()) {

    case Expr::Kind::EK_INVALID:
    case Expr::Kind::EK_UNARY:
    case Expr::Kind::EK_BINARY:
    {
        Expr_binary *e = cast<Expr_binary>(expr);
        if (e->get_operator() == Expr_binary::Operator::OK_TILDE) {
            Type *ref_t = deref(type_check_reference(e->get_left_argument(), env, Pattern_context::PC_PATTERN));
            Type *t = deref(type_check_pattern(e->get_right_argument(), env));

            if (Type_var *tv = as<Type_var>(ref_t)) {
                if (is<Type_var>(t)) {
                    error(e->get_left_argument()->get_location(),
                          "the type of this variable is not fully determined");
                    expr->set_type(m_error_type);
                    return m_error_type;
                }
                tv->assign_type(t, m_type_factory);
                e->get_left_argument()->set_type(t);
            }
            e->set_type(t);
            return t;
        }

        error(expr->get_location(), "[BUG] invalid expression kind in pattern");
        MDL_ASSERT(!"[BUG] invalid expression kind in pattern");
        expr->set_type(m_error_type);
        return m_error_type;
    }

    case Expr::Kind::EK_TYPE_ANNOTATION:
    {
        Expr_type_annotation *e = cast<Expr_type_annotation>(expr);
        Type *t = builtin_type_for(e->get_type_name()->get_name());
        if (!t) {
            error(expr->get_location(), "invalid type in type annotation. Only basic builtin types are allowed.");
            t = m_error_type;
            expr->set_type(t);
            e->get_argument()->set_type(t);
            return t;
        }
        Type *t_arg = deref(type_check_pattern(e->get_argument(), env));

        if (is<Type_error>(t_arg)) {
            return t_arg;
        }
        if (Type_var *tv = as<Type_var>(t_arg)) {
            tv->assign_type(t, m_type_factory);
        } else {
            if (!m_type_factory.types_equal(t, t_arg)) {
                error(expr->get_location(), "type annotation does not match type of annotated expression");
                expr->set_type(m_error_type);
                e->get_argument()->set_type(m_error_type);
                return m_error_type;
            }
        }
        e->get_argument()->set_type(t);
        expr->set_type(t);
        return t;
    }

    case Expr::Kind::EK_LITERAL:
    {
        error(expr->get_location(), "literals are not allowed in patterns");
        expr->set_type(m_error_type);
        return m_error_type;
    }

    case Expr::Kind::EK_REFERENCE:
    {
        return type_check_reference(expr, env, Pattern_context::PC_PATTERN);
    }

    case Expr::Kind::EK_CONDITIONAL:
    {
        Expr_conditional * e = cast<Expr_conditional>(expr);

        /*Type *type_cond =*/ type_check_pattern(e->get_condition(), env);

        Type *type_true = deref(type_check_pattern(e->get_true(), env));
        Type *type_false = deref(type_check_pattern(e->get_false(), env));


        if (is<Type_var>(type_true) || is<Type_var>(type_false)) {
        } else {
            if (!m_type_factory.types_equal(type_true, type_false)) {
                error(expr->get_location(), "type mismatch in branches of conditional expression");
                return m_error_type;
            }
        }
        return type_true;
    }

    case Expr::Kind::EK_CALL:
    {
        Type *t = type_check_call(expr, env, Pattern_context::PC_PATTERN);
        if (!is<Type_var>(t)
            && !is<Type_error>(t)
            && !is<Type_material_surface>(t)
            && !is<Type_material_volume>(t)
            && !is<Type_material_emission>(t)
            && !is<Type_material_geometry>(t)
            && !is_material_or_bsdf(t)) {
            error(expr->get_location(), "call in pattern must have bsdf or material return type");
            expr->set_type(m_error_type);
            return m_error_type;
        }
        return t;
    }

    case Expr::Kind::EK_ATTRIBUTE:
    {
        Expr_attribute *e = cast<Expr_attribute>(expr);

        Type *t_arg = type_check_attribute(expr, e, env, Pattern_context::PC_PATTERN);

        expr->set_type(t_arg);
        return t_arg;
    }
    }

    error(expr->get_location(), "internal error: end of type_check_pattern() reached");
    expr->set_type(m_error_type);
    return m_error_type;
}

/// Type check a guard expression.
void Compilation_unit::type_check_guard(Expr *guard, Environment &env) {
    if (!guard)
        return;

    // Note that the type of guard is examined in `type_check_expr`
    // and also the return type of the guard expression is checked
    // there. So we can ignore the returned type here.
    type_check_expr(guard, env);
}

/// Type check a where clause.
///
/// The bindings are added to environment `env`. It is an error to
/// redefine an variable.
void Compilation_unit::type_check_where(Argument_list &list, Environment &env) {
    Argument_list l;

    // We construct a reversed list here, because where bindings are
    // handled bottom up.
    for (Argument_list::iterator it(list.begin()), end(list.end());
         it != end; ++it) {
        Argument *a = m_expr_factory.create_argument(it->get_expr());
        l.push_front(a);
    }

    for (Argument_list::iterator it(l.begin()), end(l.end());
         it != end; ++it) {
        // The following casts and asserts check the invariant that
        // the parser only generates assignment expressions with
        // references as the LHS within a where clause.

        Expr_binary *e = cast<Expr_binary>(it->get_expr());
        MDL_ASSERT(e->get_operator() == Expr_binary::Operator::OK_ASSIGN);

        Expr_ref *r = cast<Expr_ref>(e->get_left_argument());
        Symbol const *name = r->get_name();

        Type *t = deref(type_check_expr(e->get_right_argument(), env));

        Environment *binding_env = nullptr;
        Environment::Type_list *types = env.find(name, &binding_env);

        if (types && types->size() > 0) {
            mi::mdl::string s_msg(m_arena.get_allocator());
            s_msg = "variable `";
            s_msg += name->get_name();
            s_msg += "` cannot be redefined in where clause";
            if (binding_env->get_kind() == Environment::Kind::ENV_ATTRIBUTE)
                s_msg += " (was already defined as an attribute)";

            error(e->get_location(), s_msg.c_str());
            t = m_error_type;
        } else {
            env.bind(name, t);
        }
        r->set_type(t);
        e->set_type(t);
    }
}

/// Type check a postcondition expression.
void Compilation_unit::type_check_postcond(Postcond &postcond, Environment &env) {
    if (postcond.is_empty())
        return;

    Expr *expr = postcond.get_expr();
    Type *t_cond = deref(type_check_expr(expr, env));

    if (is<Type_error>(t_cond))
        return; // Error already reported.

    if (!is<Type_bool>(t_cond)) {
        error(expr->get_location(),
              "a post condition expression must be a boolean expression");
    }
}

/// Type check a call expression.
///
/// If `pattern_ctx` is PC_PATTERN, we are currently type checking the
/// LHS of a rule, otherwise we are checking an RHS expression, where
/// clause, guard or postcondition.
Type *Compilation_unit::type_check_call(Expr *expr, Environment &env,
                                        Pattern_context pattern_ctx) {
    Expr_call * e = cast<Expr_call>(expr);
    Mdl_type_vector arg_types(m_arena.get_allocator());

    bool has_errors = false;
    for (int i = 0; i < e->get_argument_count(); i++) {
        Expr *arg = e->get_argument(i);

        Type *arg_type = deref(pattern_ctx == Pattern_context::PC_EXPRESSION
                               ? type_check_expr(arg, env)
                               : type_check_pattern(arg, env));

        if (is<Type_error>(arg_type)) {
            // We want to type check as much of the expression as
            // possibly, therefore we remember that we encountered an
            // issue and continue to check the remaining parameters.
            expr->set_type(m_error_type);
            has_errors = true;
            continue;
        }

        arg_types.push_back(arg_type);
    }

    if (has_errors) {
        expr->set_type(m_error_type);
        return m_error_type;
    }

    Expr *callee = e->get_callee();
    Type *callee_type = deref(type_check_called_reference(callee,
                                                          e,
                                                          arg_types,
                                                          env));

    if (Type_function *function_type = as<Type_function>(callee_type)) {
        expr->set_type(function_type->get_return_type());

        for (int i = 0; i < e->get_argument_count(); i++) {
            Expr *arg = e->get_argument(i);
            Type *param_type = function_type->get_parameter_type(i);

            if (Type_var *tv = as<Type_var>(arg->get_type())) {
                tv->assign_type(param_type, m_type_factory);
            }
        }

        if (pattern_ctx == Pattern_context::PC_PATTERN && function_type->get_semantics() == mi::mdl::IDefinition::Semantics::DS_UNKNOWN) {
            mi::mdl::string s(m_arena.get_allocator());
            s = "called reference in pattern must have an assigned semantics: ";
            {
                std::stringstream out;
                pp::Pretty_print p(m_arena, out);
                callee->pp(p);
                s += out.str().c_str();
        }
            error(callee->get_location(), s.c_str());
        }
        return function_type->get_return_type();
    } else if (is<Type_var>(callee_type)) {
        Type *t_v = m_type_factory.create_type_variable();
        expr->set_type(t_v);
        return t_v;
    } else {
        if (!is<Type_error>(callee_type)) {
            error(expr->get_location(), "callee is not a function");
        }
        expr->set_type(m_error_type);
        return m_error_type;
    }
}

Type *Compilation_unit::type_check_expr(Expr *expr, Environment &env) {
    switch (expr->get_kind()) {

    case Expr::Kind::EK_INVALID:
        error(expr->get_location(), "[BUG] encountered invalid expression node");
        MDL_ASSERT(!"[BUG] encountered invalid expression node");
        return m_error_type;

    case Expr::Kind::EK_TYPE_ANNOTATION:
    {
        Expr_type_annotation *e = cast<Expr_type_annotation>(expr);
        Type *t = builtin_type_for(e->get_type_name()->get_name());
        if (!t) {
            error(expr->get_location(), "invalid type in type annotation. Only basic builtin types are allowed");
            expr->set_type(m_error_type);
            e->get_argument()->set_type(m_error_type);
            return m_error_type;
        }
        Type *t_arg = deref(type_check_expr(e->get_argument(), env));

        if (is<Type_error>(t_arg)) {
            return t_arg;
        }
        if (Type_var *tv = as<Type_var>(t_arg)) {
            tv->assign_type(t, m_type_factory);
        } else {
            if (!m_type_factory.types_equal(t, t_arg)) {
                error(expr->get_location(), "type annotation does not match type of annotated expression");
                expr->set_type(m_error_type);
                e->get_argument()->set_type(m_error_type);
                return m_error_type;
            }
        }
        e->get_argument()->set_type(t);
        expr->set_type(t);
        return t;
    }

    case Expr::Kind::EK_UNARY:
    {
        Expr_unary *e = cast<Expr_unary>(expr);

        switch (e->get_operator()) {

        case Expr_unary::Operator::OK_IF_GUARD:
        case Expr_unary::Operator::OK_MAYBE_GUARD:
        {
            Type *t_arg = deref(type_check_expr(e->get_argument(), env));
            if (is<Type_error>(t_arg)) {
                return t_arg;
            }
            if (!is<Type_bool>(t_arg)) {
                error(expr->get_location(),
                      "a rule guard must be a boolean expression");
                return m_error_type;
            }
            return t_arg;
        }

        case Expr_unary::Operator::OK_NONODE:
        {
            if (!check_reference_exists(e->get_argument(),
                                        env)) {
                expr->set_type(m_error_type);
                return m_error_type;
            }
            expr->set_type(m_type_factory.get_bool());
            return m_type_factory.get_bool();
        }

        case Expr_unary::Operator::OK_MATCH:
        {
            Type *t_arg = deref(type_check_pattern(e->get_argument(), env));

            (void) t_arg; // Nothing we can check here.

            return m_type_factory.get_bool();
        }

        case Expr_unary::Operator::OK_OPTION:
        {
            // We do not want to bind identifiers in option() calls,
            // as they are not mdltl variables.
            Environment dummy_env(m_arena, Environment::Kind::ENV_LOCAL, nullptr);

            // Type check the argument, because it might be a type annotation.
            Type *t_arg = deref(type_check_pattern(e->get_argument(), dummy_env));

            // If there is no type annotation, we derive the type from
            // the name of the option.

            if (Expr_ref *ref = as<Expr_ref>(e->get_argument())) {
                if (Type_var *t_ref = as<Type_var>(e->get_argument()->get_type())) {
                    Symbol const *option_name = ref->get_name();

                    if (option_name == m_symbol_table->get_symbol("top_layer_weight")) {
                        t_arg = m_type_factory.get_float();
                    } else if (option_name == m_symbol_table->get_symbol("global_ior")) {
                        t_arg = m_type_factory.get_float();
                    } else if (option_name == m_symbol_table->get_symbol("global_float_ior")) {
                        t_arg = m_type_factory.get_float();
                    } else if (option_name == m_symbol_table->get_symbol("merge_metal_and_base_color")) {
                        t_arg = m_type_factory.get_bool();
                    } else if (option_name == m_symbol_table->get_symbol("merge_transmission_and_base_color")) {
                        t_arg = m_type_factory.get_bool();
                    } else {
                        error(expr->get_location(),
                              "unsupported option name");
                        t_arg = m_error_type;
                    }
                    t_ref->assign_type(t_arg, m_type_factory);
                }
            }
            expr->set_type(t_arg);
            return t_arg;
        }

        default:
        {
            Type *t_arg = deref(type_check_expr(e->get_argument(), env));

            return t_arg;
        }
        }
    }

    case Expr::Kind::EK_BINARY:
    {
        Expr_binary *e = cast<Expr_binary>(expr);

        Type *t_lhs = deref(type_check_expr(e->get_left_argument(), env));
        if (is<Type_error>(t_lhs)) {
            expr->set_type(t_lhs);
            return t_lhs;
        }

        switch (e->get_operator()) {
        case Expr_binary::Operator::OK_SELECT:
        {
            if (Type_struct *ts = as<Type_struct>(t_lhs)) {
                Expr_ref *tr = cast<Expr_ref>(expr->get_sub_expression(1));

                Type *field_t = ts->get_field_type(tr->get_name());
                if (field_t) {
                    expr->set_type(field_t);
                    return field_t;
                } else {
                    mi::mdl::string s_msg(m_arena.get_allocator());
                    s_msg = "unknown field name: ";
                    s_msg += tr->get_name()->get_name();
                    error(expr->get_sub_expression(1)->get_location(), s_msg.c_str());
                    expr->set_type(m_error_type);
                    return m_error_type;
                }
            } else {
                error(expr->get_sub_expression(0)->get_location(),
                      "struct expression expected");
                expr->set_type(m_error_type);
                return m_error_type;
            }
        }
        case Expr_binary::Operator::OK_ARRAY_SUBSCRIPT:
        {
            Type *t_rhs = deref(type_check_expr(e->get_right_argument(), env));

            if (is<Type_error>(t_rhs)) {
                expr->set_type(t_rhs);
                return t_rhs;
            }

            if (Type_array *ta = as<Type_array>(t_lhs)) {
                if (is<Type_int>(t_rhs)) {
                    return ta->get_element_type();
                } else {
                    error(expr->get_sub_expression(1)->get_location(),
                          "array index must be of type int");
                    expr->set_type(m_error_type);
                    return m_error_type;
                }
            } else {
                error(expr->get_sub_expression(0)->get_location(),
                      "array expression expected");
                expr->set_type(m_error_type);
                return m_error_type;
            }
        }
        default:
        {
            Type *t_rhs = deref(type_check_expr(e->get_right_argument(), env));

            if (Type_var *tv = as<Type_var>(t_lhs)) {
                if (!is<Type_var>(t_rhs)) {
                    tv->assign_type(t_rhs, m_type_factory);
                    t_lhs = t_rhs;
                }
            }
            if (is<Type_error>(t_rhs)) {
                expr->set_type(t_rhs);
                return t_rhs;
            }

            if (Type_var *tv = as<Type_var>(t_rhs)) {
                if (!is<Type_var>(t_lhs)) {
                    tv->assign_type(t_lhs, m_type_factory);
                    t_rhs = t_lhs;
                }
            }

            Type *common_type = types_compatible(expr, e->get_operator(), t_lhs, t_rhs);
            if (is<Type_error>(common_type)) {
                return common_type;
            }
            expr->set_type(common_type);
            return common_type;
        }
        }
    }

    case Expr::Kind::EK_LITERAL:
    {
        Expr_literal *expr_lit = cast<Expr_literal>(expr);
        Type *type_lit = expr_lit->get_value()->get_type();
        expr->set_type(type_lit);
        return type_lit;
    }

    case Expr::Kind::EK_REFERENCE:
        return deref(type_check_reference(expr, env, Pattern_context::PC_EXPRESSION));

    case Expr::Kind::EK_CONDITIONAL:
    {
        Expr_conditional * e = cast<Expr_conditional>(expr);

        Type *type_cond = deref(type_check_expr(e->get_condition(), env));
        Type *type_true = deref(type_check_expr(e->get_true(), env));
        Type *type_false = deref(type_check_expr(e->get_false(), env));

        if (is<Type_error>(type_cond)) {
            expr->set_type(type_cond);
            return type_cond;
        }

        if (is<Type_error>(type_true)) {
            expr->set_type(type_true);
            return type_true;
        }

        if (is<Type_error>(type_false)) {
            expr->set_type(type_false);
            return type_false;
        }

        if (!m_type_factory.types_equal(type_cond, m_type_factory.get_bool())) {
            error(expr->get_location(),
                  "condition in conditional expression must be of type bool");
            return m_error_type;
        }

        if (!m_type_factory.types_equal(type_true, type_false)) {
            error(expr->get_location(),
                  "type mismatch in branches of conditional expressions");
            return m_error_type;
        }

        expr->set_type(type_true);
        return type_true;
    }

    case Expr::Kind::EK_CALL:
        return type_check_call(expr, env, Pattern_context::PC_EXPRESSION);

    case Expr::Kind::EK_ATTRIBUTE:
    {
        Expr_attribute *e = cast<Expr_attribute>(expr);

        Type *t_arg = type_check_attribute(expr, e, env, Pattern_context::PC_EXPRESSION);

        expr->set_type(t_arg);
        return t_arg;
    }

    }

    error(expr->get_location(),
          "internal error: end of type_check_expr() reached");
    return m_error_type;
}

Type *Compilation_unit::types_compatible_arith(Expr *expr, Expr_binary::Operator op, Type *type1, Type *type2) {
    if (m_type_factory.types_equal(type1, type2))
        return type1;

    if (is_color(type1) && is_scalar(type2))
        return type1;
    if (is_scalar(type1) && is_color(type2))
        return type2;
    if (is_color(type1) && is_color(type2))
        return type1;

    if (is_vector(type1) && is_scalar(type2))
        return type1;
    if (is_scalar(type1) && is_vector(type2))
        return type2;
    if (is_vector(type1) && is_vector(type2)) {
        Type_vector *v1 = cast<Type_vector>(type1);
        Type_vector *v2 = cast<Type_vector>(type2);
        if (!types_compatible_arith(expr, op, v1->get_element_type(), v2->get_element_type()) ||
            v1->get_size() != v2->get_size()) {
            error(expr->get_location(), "type mismatch in binary vector operation");
            expr->set_type(m_error_type);
            return m_error_type;
        }
        return type1;
    }

    if (is_matrix(type1) && is_scalar(type2))
        return type1;
    if (is_scalar(type1) && is_matrix(type2))
        return type2;

    error(expr->get_location(), "type mismatch in arithmetic operation");
    expr->set_type(m_error_type);
    return m_error_type;
}

Type *Compilation_unit::types_compatible_cmp(Expr *expr, Expr_binary::Operator op, Type *type1, Type *type2) {
    Type *t_bool = m_type_factory.get_bool();

    switch (op) {
    case Expr_binary::Operator::OK_EQUAL:
    case Expr_binary::Operator::OK_NOT_EQUAL:
    {
        if (is_scalar(type1) && is_scalar(type2))
            return t_bool;

        if (is<Type_bool>(type1) && is<Type_bool>(type2))
            return t_bool;

        if (is_color(type1) && is_color(type2))
            return t_bool;

        if (is<Type_string>(type1) && is<Type_string>(type2))
            return t_bool;

        if (is<Type_enum>(type1) && m_type_factory.types_equal(type1, type2))
            return t_bool;

        if (is_vector(type1) && is_vector(type2)) {
            Type_vector* v1 = cast<Type_vector>(type1);
            Type_vector* v2 = cast<Type_vector>(type2);
            if (!types_compatible_cmp(expr, op, v1->get_element_type(), v2->get_element_type()) ||
                v1->get_size() != v2->get_size()) {
                error(expr->get_location(), "type mismatch in vector comparison");
                expr->set_type(m_error_type);
                return m_error_type;
            }
            return t_bool;
        }

        if (is_vector(type1)) {
            Type_vector* v1 = cast<Type_vector>(type1);
            if (m_type_factory.types_equal(v1->get_element_type(), type2))
                return t_bool;
        }

        if (is_vector(type2)) {
            Type_vector* v2 = cast<Type_vector>(type2);
            if (m_type_factory.types_equal(type1, v2->get_element_type()))
                return t_bool;
        }

        break;
    }

    default:
    {
        if (is_scalar(type1) && m_type_factory.types_equal(type1, type2))
            return t_bool;
    }
    }

    error(expr->get_location(), "type mismatch in comparison operation");
    expr->set_type(m_error_type);
    return m_error_type;
}

Type *Compilation_unit::types_compatible(Expr *expr, Expr_binary::Operator op, Type *type1, Type *type2) {
    switch (op) {
    case Expr_binary::Operator::OK_MULTIPLY:
        return types_compatible_arith(expr, op, type1, type2);

    case Expr_binary::Operator::OK_DIVIDE:
        return types_compatible_arith(expr, op, type1, type2);

    case Expr_binary::Operator::OK_MODULO:
        if (is_scalar(type1) && is_scalar(type2))
            return promoted_type(type1, type2);
        error(expr->get_location(), "type mismatch in arithmetic operation");
        expr->set_type(m_error_type);
        return m_error_type;

    case Expr_binary::Operator::OK_PLUS:
        return types_compatible_arith(expr, op, type1, type2);

    case Expr_binary::Operator::OK_MINUS:
        return types_compatible_arith(expr, op, type1, type2);

    case Expr_binary::Operator::OK_SHIFT_LEFT:
    case Expr_binary::Operator::OK_SHIFT_RIGHT:
    case Expr_binary::Operator::OK_SHIFT_RIGHT_ARITH:
        if (!is<Type_int>(type1) || !is<Type_int>(type2)) {
            error(expr->get_location(), "shift operators require int arguments");
            expr->set_type(m_error_type);
            return m_error_type;
        }
        return type1;

    case Expr_binary::Operator::OK_LESS:
    case Expr_binary::Operator::OK_LESS_OR_EQUAL:
    case Expr_binary::Operator::OK_GREATER_OR_EQUAL:
    case Expr_binary::Operator::OK_GREATER:
    case Expr_binary::Operator::OK_EQUAL:
    case Expr_binary::Operator::OK_NOT_EQUAL:
        return types_compatible_cmp(expr, op, type1, type2);

    case Expr_binary::Operator::OK_BITWISE_AND:
    case Expr_binary::Operator::OK_BITWISE_OR:
    case Expr_binary::Operator::OK_BITWISE_XOR:
        if (!is<Type_int>(type1) || !is<Type_int>(type2)) {
            error(expr->get_location(), "type mismatch in bitwise operation");
            expr->set_type(m_error_type);
            return m_error_type;
        }
        return type1;

    case Expr_binary::Operator::OK_LOGICAL_AND:
    case Expr_binary::Operator::OK_LOGICAL_OR:
        if (!is<Type_bool>(type1) || !is<Type_bool>(type2)) {
            error(expr->get_location(), "type mismatch in logical operation");
            expr->set_type(m_error_type);
            return m_error_type;
        }
        return m_type_factory.get_bool();

    case Expr_binary::Operator::OK_ASSIGN:
        return type2;

    default:
        error(expr->get_location(), "[BUG] unhandled binary operator");
        MDL_ASSERT(!"[BUG] unhandled binary operator");
        expr->set_type(m_error_type);
        return m_error_type;

    }
}

