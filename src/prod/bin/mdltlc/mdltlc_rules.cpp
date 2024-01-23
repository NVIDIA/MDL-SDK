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

#include <sstream>

#include <mdl/compiler/compilercore/compilercore_hash.h>

#include "mdltlc_types.h"
#include "mdltlc_rules.h"

// ------------------------------- Import --------------------------------

// Constructor.
Import::Import(Location const &loc, Symbol *symbol)
    : m_loc(loc)
    , m_symbol(symbol)
{
}

Symbol *Import::get_symbol() const
{
  return m_symbol;
}

const char *Import::get_name() const
{
  return m_symbol->get_name();
}

// -------------------------------- Rule ---------------------------------

// ------------------------------ Debug_out ------------------------------

// Constructor.
Debug_out::Debug_out(Location const &loc, Symbol *symbol)
    : m_loc(loc)
    , m_symbol(symbol)
{
}

Symbol *Debug_out::get_symbol() const
{
  return m_symbol;
}

const char *Debug_out::get_name() const
{
  return m_symbol->get_name();
}

// -------------------------------- Rule ---------------------------------

// Constructor.
Rule::Rule(mi::mdl::Memory_arena *arena,
           Location const &location,
           Symbol const *rule_name,
           Expr *expr_left,
           Expr *expr_right,
           Result_code result_code,
           Expr *guard,
           Dead_rule dead_rule)
    : m_arena(arena)
    , m_loc(location)
    , m_rule_name(rule_name)
    , m_expr_left(expr_left)
    , m_expr_right(expr_right)
    , m_result_code(result_code)
    , m_guard(guard)
    , m_bindings()
    , m_debug_out()
    , m_dead_rule(dead_rule)
    , m_uid(0)
{
}

Expr *Rule::get_lhs() {
    return m_expr_left;
}

Expr const *Rule::get_lhs() const{
    return m_expr_left;
}

void Rule::set_lhs(Expr *lhs) {
    m_expr_left = lhs;
}

Expr *Rule::get_rhs() {
    return m_expr_right;
}

Expr const *Rule::get_rhs() const {
    return m_expr_right;
}

Rule::Result_code Rule::get_result_code() const {
    return m_result_code;
}

Expr *Rule::get_guard() {
    return m_guard;
}

Expr const *Rule::get_guard() const {
    return m_guard;
}

Rule::Dead_rule Rule::get_dead_rule() const {
    return m_dead_rule;
}

/// Add all bindings in the list to the rule. The list `bindings` will
/// be empty afterwards.
void Rule::set_bindings(Argument_list &bindings)
{
  Argument *arg = bindings.front();
  while (arg) {
    bindings.pop_front();
    m_bindings.push(arg);
    arg = bindings.front();
  }
}

Argument_list &Rule::get_bindings() {
    return m_bindings;
}

Argument_list const &Rule::get_bindings() const {
    return m_bindings;
}

Debug_out_list const &Rule::get_debug_out() const {
  return m_debug_out;
}

void Rule::set_debug_out(Debug_out_list &deb_outs)
{
  Debug_out *deb_out = deb_outs.front();
  while (deb_out) {
    deb_outs.pop_front();
    m_debug_out.push(deb_out);
    deb_out = deb_outs.front();
  }
}

void Rule::pp(pp::Pretty_print &p) const {
    if (m_rule_name != nullptr) {
        p.string("rule");
        p.space();
        p.string(m_rule_name->get_name());
        p.space();
    }
    m_expr_left->pp(p);
    p.nl();
    p.string("-->");
    p.nl();
    m_expr_right->pp(p);
    switch (m_result_code) {
    case Rule::Result_code::RC_NO_RESULT_CODE:
        break;
    case Rule::Result_code::RC_SKIP_RECURSION:
        p.space();
        p.string("skip_recursion");
        break;
    case Rule::Result_code::RC_REPEAT_RULES:
        p.space();
        p.string("repeat_rules");
        break;
    }
    if (m_guard) {
        p.nl();
        m_guard->pp(p);
    }
    if (!m_bindings.empty()) {
        p.nl();
        p.string("where");
        p.space();
        for (mi::mdl::Ast_list<Argument>::const_iterator it(m_bindings.begin()),
                 end(m_bindings.end());
             it != end;
             ++it) {
            it->get_expr()->pp(p);
            p.space();
        }
    }
    switch (m_dead_rule) {
    case Rule::Dead_rule::DR_NO_DEAD_RULE:
        break;
    case Rule::Dead_rule::DR_DEAD:
        p.string("dead_rule");
        break;
    }
    if (!m_debug_out.empty()) {
        p.nl();
        p.string("debug_out(");
        int l = 0;
        for (mi::mdl::Ast_list<Debug_out>::const_iterator it(m_debug_out.begin()),
                 end(m_debug_out.end());
             it != end;
             ++it) {
            if (l > 0) {
                p.comma();
                p.space();
            }
            p.string(it->get_name());
            l++;
        }
        p.string(")");
    }
    p.string(";");
}

unsigned Rule::calc_hash(mi::mdl::IAllocator *alloc, char const *ruleset_name) {

    std::stringstream out;

    out << ruleset_name << ":";

    pp::Pretty_print p(*m_arena, out);
    m_expr_left->pp(p);

    if (m_guard) {
        p.string(" if ");
        m_guard->pp(p);
    }

    if (!m_bindings.empty()) {
        p.string(" where ");
        bool first = true;
        for (mi::mdl::Ast_list<Argument>::const_iterator it(m_bindings.begin()),
                 end(m_bindings.end());
             it != end; ++it) {
            if (first)
                first = false;
            else
                p.string(",");
            it->get_expr()->pp(p);
        }
    }

    mi::mdl::MD5_hasher hasher;
    unsigned char result[16];
    unsigned uid;

    hasher.update(out.str().c_str());
    hasher.final(result);
    uid = (result[0] << 24 | result[1] << 16 | result[2] << 8 | result[3]) ^
        (result[4] << 24 | result[5] << 16 | result[6] << 8 | result[7]) ^
        (result[8] << 24 | result[9] << 16 | result[10] << 8 | result[11]) ^
        (result[12] << 24 | result[13] << 16 | result[14] << 8 | result[15]);
    uid = uid % 999983;

    m_uid = uid;
    return uid;
}

// ------------------------------ Postcond -------------------------------

// Constructor.
Postcond::Postcond()
  : m_empty(true)
  , m_expr()
{
}

// Constructor.
Postcond::Postcond(Expr *expr)
  : m_empty(false)
  , m_expr(expr)
{
}

void Postcond::set_expr(Expr *expr)
{
  m_empty = false;
  m_expr = expr;
}

Expr *Postcond::get_expr() {
  MDL_ASSERT(!m_empty && "attempt to access expression of empty Postcond");
  return m_expr;
}

Expr const *Postcond::get_expr() const {
  MDL_ASSERT(!m_empty && "attempt to access expression of empty Postcond");
  return m_expr;
}

bool Postcond::is_empty() const {
  return m_empty;
}

// ------------------------------- Ruleset -------------------------------

// Constructor.
Ruleset::Ruleset(Location const &location,
                 Symbol *name,
                 Strategy strategy)
    : m_loc(location)
    , m_name(name)
    , m_strategy(strategy)
    , m_imports()
    , m_rules()
    , m_postcond()
{
}

// Get the strategy.
const char *Ruleset::get_name() const {
  return m_name->get_name();
}

// Get the strategy.
Ruleset::Strategy Ruleset::get_strategy() const {
  return m_strategy;
}

Import_list const &Ruleset::get_imports() const {
  return m_imports;
}

Rule_list &Ruleset::get_rules() {
  return m_rules;
}

Rule_list const &Ruleset::get_rules() const {
  return m_rules;
}

Postcond &Ruleset::get_postcond() {
  return m_postcond;
}

void Ruleset::set_postcond_expr(Expr *expr)
{
  m_postcond.set_expr(expr);
}

void Ruleset::add_import(Import *import)
{
  m_imports.push(import);
}

void Ruleset::add_rule(Rule *rule)
{
  m_rules.push(rule);
}

void Ruleset::pp(pp::Pretty_print &p) const {
    p.string("rules");
    p.space();
    p.string(m_name->get_name());
    switch (m_strategy) {
    case Ruleset::STRAT_BOTTOMUP:
        p.space();
        p.string("bottomup");
        break;
    case Ruleset::STRAT_TOPDOWN:
        p.space();
        p.string("topdown");
        break;
    }
    p.space();
    p.lbrace();
    ++p;
    p.nl();
    for (mi::mdl::Ast_list<Import>::const_iterator it(m_imports.begin()),
             end(m_imports.end());
         it != end;
         ++it) {
        p.string("import");
        p.space();
        p.string(it->get_name());
        p.string(";");
        p.nl();
    }
    for (mi::mdl::Ast_list<Rule>::const_iterator it(m_rules.begin()),
             end(m_rules.end());
         it != end;
         ++it) {
        it->pp(p);
        p.nl();
    }
    if (!m_postcond.is_empty()) {
        p.string("postcond");
        p.space();
        m_postcond.get_expr()->pp(p);
    }
    --p;
    p.nl();
    p.rbrace();
    p.nl();
}

// ----------------------------- Rule_factory ----------------------------

// Constructs a new ruleset factory.
Rule_factory::Rule_factory(
    mi::mdl::Memory_arena  &arena)
    : Base()
    , m_arena(arena)
    , m_builder(arena)
{
}

Import *Rule_factory::create_import(Location const &loc, Symbol *symbol)
{
    return m_builder.create<Import>(loc, symbol);
}

Debug_out *Rule_factory::create_debug_out(Location const &loc, Symbol *symbol)
{
    return m_builder.create<Debug_out>(loc, symbol);
}

Rule *Rule_factory::create_rule(Location const &location,
                                Symbol const *rule_name,
                                Expr *expr_left,
                                Expr *expr_right,
                                Rule::Result_code result_code,
                                Expr *guard,
                                Rule::Dead_rule dead_rule)
{
    return m_builder.create<Rule>(&m_arena,
                                  location,
                                  rule_name,
                                  expr_left,
                                  expr_right,
                                  result_code, guard, dead_rule);
}

// Create an postcond.
Postcond *Rule_factory::create_postcond()
{
  return m_builder.create<Postcond>();
}

// Create a ruleset.
Ruleset *Rule_factory::create_ruleset(Location const &location,
                                      Symbol *symbol,
                                      Ruleset::Strategy strategy)
{
    return m_builder.create<Ruleset>(location, symbol, strategy);
}

