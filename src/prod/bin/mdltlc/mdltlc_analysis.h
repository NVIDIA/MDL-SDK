/******************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDLTLC_ANALYSIS_H
#define MDLTLC_ANALYSIS_H 1

#include "mdltlc_types.h"
#include "mdltlc_exprs.h"
#include "mdltlc_expr_walker.h"

// A visitor to calculate variables used in a pattern.
class Defined_vars_visitor : public Const_expr_visitor {
public:
    Defined_vars_visitor(Var_set &defined_vars)
        : m_defined_vars(defined_vars) {}

    virtual void visit(Expr_ref const *e) {
        m_defined_vars.insert(e->get_name());
    }

    virtual void visit(Expr_attribute const *e) {
        Expr_attribute::Expr_attribute_vector const &attrs = e->get_attributes();
        for (size_t i = 0; i < attrs.size(); i++) {
            m_defined_vars.insert(attrs[i].name);
        }
    }

private:
    Var_set &m_defined_vars;
};

/// Add all variables declared in the given pattern to the variable
/// set.
void defined_vars(Expr const *expr, Var_set &def_vars);

// A visitor to calculate variables used in an expression.
class Used_vars_visitor : public Const_expr_visitor {
public:
    Used_vars_visitor(Var_set &used_vars)
        : m_used_vars(used_vars) {}

    virtual void visit(Expr_ref const *e) {
        m_used_vars.insert(e->get_name());
    }

private:
    Var_set &m_used_vars;
};

/// Add all variables used in the given expression to the variable
/// set.
void used_vars(Expr const *expr, Var_set &u_vars);

// A visitor to calculate variables used in a pattern. This is only
// used for attributes, which are considered used in patterns because
// their name is required for matching.
class Lhs_used_vars_visitor : public Const_expr_visitor {
public:
    Lhs_used_vars_visitor(Var_set &used_vars)
        : m_used_vars(used_vars) {}

    virtual void visit(Expr_attribute const *expr) {
        Expr_attribute::Expr_attribute_vector const &attrs = expr->get_attributes();
        for (size_t i = 0; i < attrs.size(); i++) {
            Expr_attribute::Expr_attribute_entry const &attr = attrs[i];
            m_used_vars.insert(attr.name);
        }
    }
private:
    Var_set &m_used_vars;
};

/// Add all variables used in the given pattern (that is, attribute
/// names) to the variable set.
void lhs_used_vars(Expr const *expr, Var_set &u_vars);

// A visitor to check for unbound type variables in attribute types.
class Check_attr_types_visitor : public Const_expr_visitor {
public:
    Check_attr_types_visitor(Compilation_unit *unit)
        : m_unit(unit) {}

    virtual void visit(Expr_attribute const *expr);

private:
    Compilation_unit *m_unit;

};

/// Check that the types of all attributes appearing in the given
/// expression are determined (that is, are not unbound type
/// variables.
void check_attr_types_determined(Compilation_unit *unit, Expr const *expr);

class Check_topdown_attr_visitor : public Const_expr_visitor {
public:
    Check_topdown_attr_visitor(Compilation_unit *unit)
        : m_unit(unit) {}

    virtual void visit(Expr_attribute const *expr);

private:
    Compilation_unit *m_unit;
};

// Check that an RHS expression in a top-down rule set does not create
// new attributes.
void check_topdown_attrs(Compilation_unit *unit, Expr const *expr);

// A visitor to calculate variables used in calls to target materials.
class Used_target_material_visitor : public Const_expr_visitor {
public:
    Used_target_material_visitor(mi::mdl::Memory_arena &arena,
                                 Var_set &used_vars)
        : m_arena(arena)
        , m_used_vars(used_vars) {}

    virtual void visit(Expr_call const *e) {
        Expr_ref const * ref = cast<Expr_ref>(e->get_callee());
        Type_function const *t = cast<Type_function>(ref->get_type());
        Type const *ret_t = t->get_return_type();

        if (t->get_semantics() == mi::mdl::IDefinition::Semantics::DS_UNKNOWN) {
            if (is<Type_bsdf>(ret_t)
                || is<Type_vdf>(ret_t)
                || is<Type_edf>(ret_t)
                || is<Type_hair_bsdf>(ret_t)
                || is<Type_material>(ret_t)) {
                m_used_vars.insert(ref->get_name());
            }
            if (Type_struct const *ts = as<Type_struct>(ret_t)) {
                if (!strcmp(ts->get_name()->get_name(), "material")) {
                    m_used_vars.insert(ref->get_name());
                }
            }
        }
    }

private:
    mi::mdl::Memory_arena &m_arena;
    Var_set &m_used_vars;
};

/// Add all function names with unknown semantics to the variable
/// set. These are custom target material functions.
void used_target_materials(mi::mdl::Memory_arena &arena,
                           Expr const *expr, Var_set &u_vars);

#endif
