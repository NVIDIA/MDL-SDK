/******************************************************************************
 * Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDLTLC_RULES_H
#define MDLTLC_RULES_H 1

#include <mi/mdl/mdl_iowned.h>
#include <mdl/compiler/compilercore/compilercore_memory_arena.h>
#include <mdl/compiler/compilercore/compilercore_ast_list.h>

#include "mdltlc_pprint.h"

#include "mdltlc_locations.h"
#include "mdltlc_exprs.h"
#include "mdltlc_symbols.h"

/// Import statement.
class Import : public mi::mdl::Ast_list_element<Import>
{
    typedef mi::mdl::Ast_list_element<Import> Base;

    friend class mi::mdl::Arena_builder;

public:

    Symbol *get_symbol() const;
    const char *get_name() const;

    Location const &get_location() const { return m_loc; }

protected:
    /// Constructor.
    explicit Import(Location const &loc,
                    Symbol *symbol);

private:
    // non copyable
    Import(Import const &) = delete;
    Import &operator=(Import const &) = delete;

protected:
    Location m_loc;
    Symbol *m_symbol;
};

typedef mi::mdl::Ast_list<Import> Import_list;

class Debug_out : public mi::mdl::Ast_list_element<Debug_out>
{
    typedef mi::mdl::Ast_list_element<Import> Base;

    friend class mi::mdl::Arena_builder;

public:

    Symbol *get_symbol() const;
    const char *get_name() const;

    Location const &get_location() const { return m_loc; }

protected:
    /// Constructor.
    explicit Debug_out(Location const &loc,
                       Symbol *symbol);

private:
    // non copyable
    Debug_out(Debug_out const &) = delete;
    Debug_out &operator=(Debug_out const &) = delete;

protected:
    Location m_loc;
    Symbol *m_symbol;
};

typedef mi::mdl::Ast_list<Debug_out> Debug_out_list;

/// Rule statement.
class Rule : public mi::mdl::Ast_list_element<Rule>
{
    typedef mi::mdl::Ast_list_element<Rule> Base;

    friend class mi::mdl::Arena_builder;

public:

    /// The possible kinds of rule result codes.
    enum Result_code {
        RC_NO_RESULT_CODE,                ///< No result code.
        RC_SKIP_RECURSION,                ///< Skip repeated application of current rule set.
        RC_REPEAT_RULES                   ///< Apply rules again after application.
    };

    /// The possible kinds of rule result codes.
    enum Dead_rule {
        DR_NO_DEAD_RULE,                  ///< No deadrule marker.
        DR_DEAD                           ///< Rule marked as dead.
    };

    /// Return the left-hand side (the pattern) of a rule.
    Expr *get_lhs();
    Expr const *get_lhs() const;

    /// Set the left-hand side of a rule.
   void set_lhs(Expr *lhs);

    /// Return the right-hand side (the expression) of a rule.
    Expr *get_rhs();
    Expr const *get_rhs() const;

    /// Return the name of the rule (if any), or nullptr.
    char const* get_rule_name() const {
        if (m_rule_name != nullptr) {
            return m_rule_name->get_name();
        } else {
            return nullptr;
        }
    }

    /// Return the result code of a rule).
    Result_code get_result_code() const;

    /// Return the guard expression of a rule or NULL if absent.
    Expr *get_guard();
    Expr const *get_guard() const;

    /// Return the "deadrule" flag of a rule.
    Dead_rule get_dead_rule() const;


    /// Add a binding clause ("where") to a rule.
    void set_bindings(Argument_list &binding);

    /// Return a reference to the binding ("where" clause) list of a
    /// rule.
    Argument_list &get_bindings();
    Argument_list const &get_bindings() const;

    Debug_out_list const &get_debug_out() const;
    void set_debug_out(Debug_out_list &deb_outs);
    /// Pretty-print the rule using the given pretty-printer.
    void pp(pp::Pretty_print &p) const;

    /// Return the rule's identifier (a hash).
    unsigned get_uid() const { return m_uid; }

    /// Calculate a hash value for the rule and return it. Also stores
    /// the value as the rules uid.
    unsigned calc_hash(mi::mdl::IAllocator *alloc, char const *ruleset_name);

    /// Get the Location.
    Location const &get_location() const { return m_loc; }

protected:
    /// Constructor.
    explicit Rule(mi::mdl::Memory_arena *arena,
                  Location const &loc,
                  Symbol const *rule_name,
                  Expr *expr_left,
                  Expr *expr_right,
                  Result_code result_code,
                  Expr *guard,
                  Dead_rule dead_rule);

private:
    // non copyable
    Rule(Rule const &) = delete;
    Rule &operator=(Rule const &) = delete;

protected:
    mi::mdl::Memory_arena *m_arena;

    /// The location of this rule
    Location const m_loc;

    Symbol const *m_rule_name;
    Expr* m_expr_left;
    Expr* m_expr_right;
    Result_code m_result_code;
    Expr* m_guard;
    Argument_list m_bindings;
    Debug_out_list m_debug_out;
    Dead_rule m_dead_rule;
    unsigned m_uid;
};

typedef mi::mdl::Ast_list<Rule> Rule_list;

/// Postcond statement.
class Postcond : public mi::mdl::Interface_owned
{
    typedef mi::mdl::Interface_owned Base;

    friend class mi::mdl::Arena_builder;
    friend class Ruleset;

public:

    void set_expr(Expr *expr);
    Expr *get_expr();
    Expr const *get_expr() const;

    bool is_empty() const;

protected:
    /// Constructor.
    explicit Postcond();
    explicit Postcond(Expr *expr);

private:
    // non copyable
    Postcond(Postcond const &) = delete;
    Postcond &operator=(Postcond const &) = delete;

protected:
    bool m_empty;
    Expr *m_expr;
};

/// mdltl rule.
class Ruleset : public mi::mdl::Ast_list_element<Ruleset>
{
    typedef mi::mdl::Ast_list_element<Ruleset> Base;

    friend class mi::mdl::Arena_builder;
    friend class Compilation_unit;

public:

    /// The possible kinds of expressions.
    enum Strategy {
        STRAT_TOPDOWN,                          ///< Topdown ruleset.
        STRAT_BOTTOMUP,                         ///< Bottomup ruleset.
    };

    char const *get_name() const;
    Strategy get_strategy() const;
    Import_list const &get_imports() const;
    Rule_list const &get_rules() const;
    Rule_list &get_rules();
    Postcond &get_postcond();

    void set_postcond_expr(Expr *expr);
    void add_import(Import *import);
    void add_rule(Rule *rule);

    /// Get the Location.
    Location const &get_location() const { return m_loc; }

    void pp(pp::Pretty_print &p) const;

protected:
    /// Constructor.
    explicit Ruleset(Location const &location,
                     Symbol *name,
                     Strategy strategy);

private:
    // non copyable
    Ruleset(Ruleset const &) = delete;
    Ruleset &operator=(Ruleset const &) = delete;

protected:
    /// The location of this rule
    Location const m_loc;
    Symbol *m_name;
    Strategy m_strategy;
    Import_list m_imports;
    Rule_list m_rules;
    Postcond m_postcond;
};

typedef mi::mdl::Ast_list<Ruleset> Ruleset_list;

/// Create rules and related entities.
class Rule_factory : public mi::mdl::Interface_owned
{
    typedef mi::mdl::Interface_owned Base;
    friend class Compilation_unit;
    friend class mi::mdl::Arena_builder;
public:

    /// Create an import.
    Import *create_import(Location const &loc, Symbol *sym);

    Debug_out *create_debug_out(Location const &loc, Symbol *sym);

    /// Create a rule.
    Rule *create_rule(Location const &location,
                      Symbol const *rule_name,
                      Expr *expr_left,
                      Expr *expr_right,
                      Rule::Result_code result_code,
                      Expr *guard,
                      Rule::Dead_rule dead_rule);

    /// Create a postcond.
    Postcond *create_postcond();

    /// Create a new ruleset.
    Ruleset *create_ruleset(Location const &location,
                            Symbol *symbol,
                            Ruleset::Strategy strategy);

private:
    /// Constructor.

    /// \param arena  the memory arena to allocate on.
    Rule_factory(
        mi::mdl::Memory_arena &arena);

private:
    /// The Arena used;
    mi::mdl::Memory_arena &m_arena;

    /// The Arena builder used;
    mi::mdl::Arena_builder m_builder;
};

#endif
