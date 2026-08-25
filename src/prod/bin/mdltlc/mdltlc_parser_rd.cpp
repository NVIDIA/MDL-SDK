/******************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "mdltlc_compilation_unit.h"
#include "mdltlc_parser_rd.h"

namespace {

enum Token_kind {
    TK_EOF,
    TK_IDENT,
    TK_INTEGER_LITERAL,
    TK_FLOATING_LITERAL,
    TK_STRING_LITERAL,
    TK_IMPORT,
    TK_TRUE,
    TK_FALSE,
    TK_INC_OP,
    TK_DEC_OP,
    TK_MAPSTO,
    TK_AT_OP,
    TK_ATTR_BEGIN,
    TK_ATTR_END,
    TK_BOTTOMUP,
    TK_TOPDOWN,
    TK_REPEAT_RULES,
    TK_SKIP_RECURSION,
    TK_RULES,
    TK_DEADRULE,
    TK_POSTCOND,
    TK_IF_KW,
    TK_MAYBE,
    TK_WHERE,
    TK_OPTION,
    TK_NONODE,
    TK_MATCH,
    TK_DEBUG_NAME,
    TK_DEBUG_PRINT,
    TK_LBRACE,
    TK_RBRACE,
    TK_LPAREN,
    TK_RPAREN,
    TK_ASSIGN,
    TK_TILDE,
    TK_COMMA,
    TK_COLON,
    TK_SEMICOLON,
    TK_QUESTION,
    TK_LBRACKET,
    TK_RBRACKET,
    TK_DOT,
    TK_PLUS,
    TK_MINUS,
    TK_BANG,
    TK_STAR,
    TK_SLASH,
    TK_PERCENT,
    TK_SHIFT_LEFT,
    TK_SHIFT_RIGHT,
    TK_SHIFT_RIGHT_ARITH,
    TK_LESS_EQUAL,
    TK_GREATER_EQUAL,
    TK_LESS,
    TK_GREATER,
    TK_EQUAL,
    TK_NOT_EQUAL,
    TK_AMP,
    TK_LOGICAL_AND,
    TK_PIPE,
    TK_LOGICAL_OR,
    TK_CARET,
    TK_ERROR
};

struct Token {
    Token_kind  kind;
    std::string text;
    unsigned    line;
    unsigned    column;

    Token()
        : kind(TK_EOF)
        , text()
        , line(1)
        , column(1)
    {
    }
};

static bool is_letter(char c)
{
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static bool is_digit(char c)
{
    unsigned char uc = static_cast<unsigned char>(c);
    return std::isdigit(uc) != 0;
}

static bool is_hex_digit(char c)
{
    unsigned char uc = static_cast<unsigned char>(c);
    return std::isxdigit(uc) != 0;
}

static bool is_oct_digit(char c)
{
    return c >= '0' && c <= '7';
}

static Token_kind keyword_kind(std::string const &text)
{
    if (text == "import") {
        return TK_IMPORT;
    }
    if (text == "true") {
        return TK_TRUE;
    }
    if (text == "false") {
        return TK_FALSE;
    }
    if (text == "bottomup") {
        return TK_BOTTOMUP;
    }
    if (text == "topdown") {
        return TK_TOPDOWN;
    }
    if (text == "repeat_rules") {
        return TK_REPEAT_RULES;
    }
    if (text == "skip_recursion") {
        return TK_SKIP_RECURSION;
    }
    if (text == "rules") {
        return TK_RULES;
    }
    if (text == "deadrule") {
        return TK_DEADRULE;
    }
    if (text == "postcond") {
        return TK_POSTCOND;
    }
    if (text == "if") {
        return TK_IF_KW;
    }
    if (text == "maybe") {
        return TK_MAYBE;
    }
    if (text == "where") {
        return TK_WHERE;
    }
    if (text == "option") {
        return TK_OPTION;
    }
    if (text == "nonode") {
        return TK_NONODE;
    }
    if (text == "match") {
        return TK_MATCH;
    }
    if (text == "debug_name") {
        return TK_DEBUG_NAME;
    }
    if (text == "debug_print") {
        return TK_DEBUG_PRINT;
    }
    return TK_IDENT;
}

class Scanner {
public:
    Scanner(char const *source, size_t length)
        : m_source(source)
        , m_length(length)
        , m_pos(0)
        , m_line(1)
        , m_column(1)
        , m_error(false)
        , m_error_message()
        , m_error_line(1)
        , m_error_column(1)
    {
    }

    Token next()
    {
        skip_space_and_comments();

        Token token;
        token.line = m_line;
        token.column = m_column;

        if (m_error) {
            token.kind = TK_ERROR;
            return token;
        }

        if (eof()) {
            token.kind = TK_EOF;
            return token;
        }

        size_t start = m_pos;
        char c = peek();

        if (is_letter(c) || c == '_' || (c == ':' && peek(1) == ':')) {
            advance_char();
            while (!eof()) {
                char n = peek();
                if (is_letter(n) || is_digit(n) || n == '_' || n == ':') {
                    advance_char();
                } else {
                    break;
                }
            }
            token.text.assign(m_source + start, m_pos - start);
            token.kind = keyword_kind(token.text);
            return token;
        }

        if (is_digit(c) || (c == '.' && is_digit(peek(1)))) {
            scan_number(token, start);
            return token;
        }

        if (c == '"') {
            scan_string(token, start);
            return token;
        }

        token.kind = scan_operator();
        token.text.assign(m_source + start, m_pos - start);
        if (token.kind != TK_ERROR) {
            return token;
        }

        set_error("unexpected character", token.line, token.column);
        return token;
    }

    bool has_error() const { return m_error; }
    std::string const &error_message() const { return m_error_message; }
    unsigned error_line() const { return m_error_line; }
    unsigned error_column() const { return m_error_column; }

private:
    bool eof(size_t offset = 0) const
    {
        return m_pos + offset >= m_length;
    }

    char peek(size_t offset = 0) const
    {
        return eof(offset) ? '\0' : m_source[m_pos + offset];
    }

    bool starts_with(char const *s) const
    {
        size_t len = std::strlen(s);
        return m_pos + len <= m_length && std::strncmp(m_source + m_pos, s, len) == 0;
    }

    void advance_char()
    {
        if (eof()) {
            return;
        }

        char c = m_source[m_pos++];
        if (c == '\r') {
            if (!eof() && peek() == '\n') {
                ++m_pos;
            }
            ++m_line;
            m_column = 1;
        } else if (c == '\n') {
            ++m_line;
            m_column = 1;
        } else {
            ++m_column;
        }
    }

    void advance_bytes(size_t count)
    {
        for (size_t i = 0; i < count; ++i) {
            advance_char();
        }
    }

    void skip_space_and_comments()
    {
        for (;;) {
            while (!eof()) {
                char c = peek();
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                    advance_char();
                } else {
                    break;
                }
            }

            if (starts_with("//")) {
                advance_bytes(2);
                while (!eof() && peek() != '\n' && peek() != '\r') {
                    advance_char();
                }
                if (!eof()) {
                    advance_char();
                }
                continue;
            }

            if (starts_with("/*")) {
                unsigned start_line = m_line;
                unsigned start_column = m_column;
                advance_bytes(2);
                int level = 1;
                while (level > 0) {
                    if (eof()) {
                        set_error("unterminated block comment", start_line, start_column);
                        return;
                    }
                    if (starts_with("/*")) {
                        advance_bytes(2);
                        ++level;
                    } else if (starts_with("*/")) {
                        advance_bytes(2);
                        --level;
                    } else {
                        advance_char();
                    }
                }
                continue;
            }

            return;
        }
    }

    void scan_number(Token &token, size_t start)
    {
        bool is_float = false;

        if (peek() == '.') {
            is_float = true;
            advance_char();
            while (is_digit(peek())) {
                advance_char();
            }
            scan_exponent_if_present(is_float);
            scan_float_suffix_if_present();
            token.kind = TK_FLOATING_LITERAL;
            token.text.assign(m_source + start, m_pos - start);
            return;
        }

        if (peek() == '0' && (peek(1) == 'x' || peek(1) == 'X') && is_hex_digit(peek(2))) {
            advance_bytes(2);
            while (is_hex_digit(peek())) {
                advance_char();
            }
            token.kind = TK_INTEGER_LITERAL;
            token.text.assign(m_source + start, m_pos - start);
            return;
        }

        if (peek() == '0') {
            size_t digit_count = 0;
            while (is_digit(peek(digit_count))) {
                ++digit_count;
            }

            if (peek(digit_count) == '.' || valid_exponent_starts_at(digit_count)) {
                advance_bytes(digit_count);
                if (peek() == '.') {
                    is_float = true;
                    advance_char();
                    while (is_digit(peek())) {
                        advance_char();
                    }
                }
                scan_exponent_if_present(is_float);
                scan_float_suffix_if_present();
                token.kind = TK_FLOATING_LITERAL;
                token.text.assign(m_source + start, m_pos - start);
                return;
            }

            advance_char();
            while (is_oct_digit(peek())) {
                advance_char();
            }
            token.kind = TK_INTEGER_LITERAL;
            token.text.assign(m_source + start, m_pos - start);
            return;
        }

        while (is_digit(peek())) {
            advance_char();
        }

        if (peek() == '.') {
            is_float = true;
            advance_char();
            while (is_digit(peek())) {
                advance_char();
            }
        }

        scan_exponent_if_present(is_float);
        if (is_float) {
            scan_float_suffix_if_present();
        }

        token.kind = is_float ? TK_FLOATING_LITERAL : TK_INTEGER_LITERAL;
        token.text.assign(m_source + start, m_pos - start);
    }

    bool valid_exponent_starts_at(size_t offset) const
    {
        if (peek(offset) != 'e' && peek(offset) != 'E') {
            return false;
        }

        ++offset;
        if (peek(offset) == '+' || peek(offset) == '-') {
            ++offset;
        }
        return is_digit(peek(offset));
    }

    void scan_exponent_if_present(bool &is_float)
    {
        if (!valid_exponent_starts_at(0)) {
            return;
        }

        is_float = true;
        advance_char();
        if (peek() == '+' || peek() == '-') {
            advance_char();
        }
        while (is_digit(peek())) {
            advance_char();
        }
    }

    void scan_float_suffix_if_present()
    {
        char c = peek();
        if (c == 'f' || c == 'F' || c == 'd' || c == 'D') {
            advance_char();
        }
    }

    void scan_string(Token &token, size_t start)
    {
        advance_char();
        while (!eof()) {
            if (peek() == '"') {
                advance_char();
                token.kind = TK_STRING_LITERAL;
                token.text.assign(m_source + start, m_pos - start);
                return;
            }
            if (peek() == '\\') {
                advance_char();
                if (!eof()) {
                    advance_char();
                }
            } else {
                advance_char();
            }
        }

        token.kind = TK_ERROR;
        token.text.assign(m_source + start, m_pos - start);
        set_error("unterminated string literal", token.line, token.column);
    }

    Token_kind scan_operator()
    {
        if (starts_with("-->")) {
            advance_bytes(3);
            return TK_MAPSTO;
        }
        if (starts_with("++")) {
            advance_bytes(2);
            return TK_INC_OP;
        }
        if (starts_with("--")) {
            advance_bytes(2);
            return TK_DEC_OP;
        }
        if (starts_with("[[")) {
            advance_bytes(2);
            return TK_ATTR_BEGIN;
        }
        if (starts_with("]]")) {
            advance_bytes(2);
            return TK_ATTR_END;
        }
        if (starts_with(">>>")) {
            advance_bytes(3);
            return TK_SHIFT_RIGHT_ARITH;
        }
        if (starts_with("<<")) {
            advance_bytes(2);
            return TK_SHIFT_LEFT;
        }
        if (starts_with(">>")) {
            advance_bytes(2);
            return TK_SHIFT_RIGHT;
        }
        if (starts_with("<=")) {
            advance_bytes(2);
            return TK_LESS_EQUAL;
        }
        if (starts_with(">=")) {
            advance_bytes(2);
            return TK_GREATER_EQUAL;
        }
        if (starts_with("==")) {
            advance_bytes(2);
            return TK_EQUAL;
        }
        if (starts_with("!=")) {
            advance_bytes(2);
            return TK_NOT_EQUAL;
        }
        if (starts_with("&&")) {
            advance_bytes(2);
            return TK_LOGICAL_AND;
        }
        if (starts_with("||")) {
            advance_bytes(2);
            return TK_LOGICAL_OR;
        }

        char c = peek();
        advance_char();
        switch (c) {
        case '@': return TK_AT_OP;
        case '{': return TK_LBRACE;
        case '}': return TK_RBRACE;
        case '(': return TK_LPAREN;
        case ')': return TK_RPAREN;
        case '=': return TK_ASSIGN;
        case '~': return TK_TILDE;
        case ',': return TK_COMMA;
        case ':': return TK_COLON;
        case ';': return TK_SEMICOLON;
        case '?': return TK_QUESTION;
        case '[': return TK_LBRACKET;
        case ']': return TK_RBRACKET;
        case '.': return TK_DOT;
        case '+': return TK_PLUS;
        case '-': return TK_MINUS;
        case '!': return TK_BANG;
        case '*': return TK_STAR;
        case '/': return TK_SLASH;
        case '%': return TK_PERCENT;
        case '<': return TK_LESS;
        case '>': return TK_GREATER;
        case '&': return TK_AMP;
        case '|': return TK_PIPE;
        case '^': return TK_CARET;
        default:  return TK_ERROR;
        }
    }

    void set_error(char const *message, unsigned line, unsigned column)
    {
        if (m_error) {
            return;
        }
        m_error = true;
        m_error_message = message;
        m_error_line = line;
        m_error_column = column;
    }

private:
    char const *m_source;
    size_t      m_length;
    size_t      m_pos;
    unsigned    m_line;
    unsigned    m_column;
    bool        m_error;
    std::string m_error_message;
    unsigned    m_error_line;
    unsigned    m_error_column;
};

class Parser {
public:
    Parser(Compilation_unit &unit, char const *source, size_t length)
        : m_unit(unit)
        , m_scanner(source, length)
        , m_curr()
        , m_la(m_scanner.next())
        , m_error(false)
        , m_error_message()
        , m_error_line(1)
        , m_error_column(1)
        , m_expr_factory(&unit.get_expression_factory())
        , m_type_factory(unit.get_type_factory())
        , m_value_factory(&unit.get_value_factory())
        , m_rule_factory(&unit.get_rule_factory())
        , m_symtab(&unit.get_symbol_table())
        , m_gensym_counter(0)
    {
    }

    bool parse()
    {
        if (!scanner_ok()) {
            return false;
        }

        while (m_la.kind == TK_RULES) {
            if (!parse_rules()) {
                return false;
            }
        }

        return expect(TK_EOF, "end of file");
    }

    std::string const &error_message() const { return m_error_message; }
    unsigned error_line() const { return m_error_line; }
    unsigned error_column() const { return m_error_column; }

private:
    Location curr_loc() const
    {
        return Location(Location::OWNER_FILE_IDX, m_curr.line, m_curr.column);
    }

    Location la_loc() const
    {
        return Location(Location::OWNER_FILE_IDX, m_la.line, m_la.column);
    }

    Type *fresh_type()
    {
        return m_type_factory->create_type_variable();
    }

    bool scanner_ok()
    {
        if (!m_scanner.has_error()) {
            return true;
        }
        set_error(
            m_scanner.error_message().c_str(),
            m_scanner.error_line(),
            m_scanner.error_column());
        return false;
    }

    void advance()
    {
        m_curr = m_la;
        m_la = m_scanner.next();
        scanner_ok();
    }

    bool accept(Token_kind kind)
    {
        if (m_la.kind != kind) {
            return false;
        }
        advance();
        return !m_error;
    }

    bool expect(Token_kind kind, char const *expected)
    {
        if (m_la.kind == kind) {
            advance();
            return !m_error;
        }

        std::string message("expected ");
        message += expected;
        if (m_la.kind != TK_EOF) {
            message += " before '";
            message += m_la.text;
            message += "'";
        }
        set_error(message.c_str(), m_la.line, m_la.column);
        return false;
    }

    void set_error(char const *message, unsigned line, unsigned column)
    {
        if (m_error) {
            return;
        }
        m_error = true;
        m_error_message = message;
        m_error_line = line;
        m_error_column = column;
    }

    bool parse_identifier(Symbol *&symbol)
    {
        if (!expect(TK_IDENT, "identifier")) {
            return false;
        }
        symbol = m_symtab->get_symbol(m_curr.text.c_str());
        return true;
    }

    bool parse_rules()
    {
        if (!expect(TK_RULES, "'rules'")) {
            return false;
        }

        Symbol *sym = nullptr;
        if (!parse_identifier(sym)) {
            return false;
        }

        Ruleset::Strategy strat = Ruleset::STRAT_BOTTOMUP;
        if (!parse_strategy(strat)) {
            return false;
        }

        Ruleset *ruleset = m_rule_factory->create_ruleset(curr_loc(), sym, strat);

        if (!expect(TK_LBRACE, "'{'")) {
            return false;
        }

        while (m_la.kind == TK_IMPORT) {
            if (!parse_import(ruleset)) {
                return false;
            }
        }

        while (m_la.kind == TK_IDENT) {
            if (!parse_rule(ruleset)) {
                return false;
            }
        }

        if (m_la.kind == TK_POSTCOND) {
            Expr *expr = nullptr;
            if (!expect(TK_POSTCOND, "'postcond'") ||
                !parse_postcond_or_expr(expr) ||
                !expect(TK_SEMICOLON, "';'")) {
                return false;
            }
            ruleset->set_postcond_expr(expr);
        }

        if (!expect(TK_RBRACE, "'}'")) {
            return false;
        }

        m_unit.add_ruleset(ruleset);
        return true;
    }

    bool parse_strategy(Ruleset::Strategy &strat)
    {
        if (accept(TK_BOTTOMUP)) {
            strat = Ruleset::STRAT_BOTTOMUP;
            return true;
        }
        if (accept(TK_TOPDOWN)) {
            strat = Ruleset::STRAT_TOPDOWN;
            return true;
        }
        set_error("expected ruleset strategy", m_la.line, m_la.column);
        return false;
    }

    bool parse_import(Ruleset *ruleset)
    {
        if (!expect(TK_IMPORT, "'import'")) {
            return false;
        }
        Location loc = curr_loc();

        Symbol *sym = nullptr;
        if (!parse_identifier(sym)) {
            return false;
        }

        Import *import = m_rule_factory->create_import(loc, sym);
        ruleset->add_import(import);
        return expect(TK_SEMICOLON, "';'");
    }

    bool parse_rule(Ruleset *ruleset)
    {
        Expr *expr_left = nullptr;
        Expr *expr_right = nullptr;
        Expr *expr_guard = nullptr;
        Rule::Result_code res_code = Rule::RC_NO_RESULT_CODE;
        Rule::Dead_rule dead_rule = Rule::DR_NO_DEAD_RULE;
        Argument_list bindings;
        Debug_out_list deb_outs;
        Symbol *rule_name = nullptr;

        if (!parse_node(expr_left) ||
            !expect(TK_MAPSTO, "'-->'")) {
            return false;
        }
        Location rule_loc = curr_loc();

        if (!parse_expression(expr_right)) {
            return false;
        }

        if (m_la.kind == TK_REPEAT_RULES || m_la.kind == TK_SKIP_RECURSION) {
            if (!parse_result_code(res_code)) {
                return false;
            }
        }

        if (m_la.kind == TK_IF_KW || m_la.kind == TK_MAYBE) {
            if (!parse_guard(expr_guard)) {
                return false;
            }
        }

        if (m_la.kind == TK_WHERE) {
            if (!parse_where(bindings)) {
                return false;
            }
        }

        if (accept(TK_DEBUG_NAME)) {
            if (!expect(TK_STRING_LITERAL, "string literal")) {
                return false;
            }
            std::string dname = m_curr.text;
            if (dname.size() >= 2) {
                dname = dname.substr(1, dname.size() - 2);
            }
            rule_name = m_symtab->get_symbol(dname.c_str());
        }

        while (m_la.kind == TK_DEBUG_PRINT) {
            if (!parse_debug_print(deb_outs)) {
                return false;
            }
        }

        if (accept(TK_DEADRULE)) {
            dead_rule = Rule::DR_DEAD;
        }

        if (!expect(TK_SEMICOLON, "';'")) {
            return false;
        }

        Rule *rule = m_rule_factory->create_rule(
            rule_loc,
            rule_name,
            expr_left,
            expr_right,
            res_code,
            expr_guard,
            dead_rule);
        rule->set_bindings(bindings);
        rule->set_debug_out(deb_outs);
        ruleset->add_rule(rule);
        return true;
    }

    bool parse_result_code(Rule::Result_code &res_code)
    {
        if (accept(TK_REPEAT_RULES)) {
            res_code = Rule::RC_REPEAT_RULES;
            return true;
        }
        if (accept(TK_SKIP_RECURSION)) {
            res_code = Rule::RC_SKIP_RECURSION;
            return true;
        }
        set_error("expected rule result code", m_la.line, m_la.column);
        return false;
    }

    bool parse_guard(Expr *&guard_expr)
    {
        Expr *expr = nullptr;
        if (accept(TK_IF_KW)) {
            if (!parse_expression(expr)) {
                return false;
            }
            guard_expr = m_expr_factory->create_unary(
                curr_loc(), m_type_factory->get_bool(), Expr_unary::OK_IF_GUARD, expr);
            return true;
        }
        if (accept(TK_MAYBE)) {
            if (!parse_expression(expr)) {
                return false;
            }
            guard_expr = m_expr_factory->create_unary(
                curr_loc(), m_type_factory->get_bool(), Expr_unary::OK_MAYBE_GUARD, expr);
            return true;
        }
        set_error("expected rule guard", m_la.line, m_la.column);
        return false;
    }

    bool parse_where(Argument_list &bindings)
    {
        return expect(TK_WHERE, "'where'") && parse_binding_list(bindings);
    }

    bool parse_binding_list(Argument_list &bindings)
    {
        Expr *expr = nullptr;
        if (!parse_binding(expr)) {
            return false;
        }
        bindings.push(m_expr_factory->create_argument(expr));

        while (m_la.kind == TK_IDENT) {
            if (!parse_binding(expr)) {
                return false;
            }
            bindings.push(m_expr_factory->create_argument(expr));
        }
        return true;
    }

    bool parse_binding(Expr *&expr)
    {
        Symbol *sym = nullptr;
        if (!parse_identifier(sym)) {
            return false;
        }

        Expr *ref = m_expr_factory->create_reference(curr_loc(), fresh_type(), sym);
        Location loc = curr_loc();

        Expr *expr_right = nullptr;
        if (!expect(TK_ASSIGN, "'='") ||
            !parse_expression(expr_right)) {
            return false;
        }

        expr = m_expr_factory->create_binary(
            loc, fresh_type(), Expr_binary::OK_ASSIGN, ref, expr_right);
        return true;
    }

    bool parse_debug_print(Debug_out_list &deb_outs)
    {
        return expect(TK_DEBUG_PRINT, "'debug_print'") &&
            expect(TK_LPAREN, "'('") &&
            parse_debug_variable_list(deb_outs) &&
            expect(TK_RPAREN, "')'");
    }

    bool parse_debug_variable_list(Debug_out_list &deb_outs)
    {
        Symbol *sym = nullptr;
        Location loc = curr_loc();

        if (m_la.kind == TK_IDENT) {
            if (!parse_identifier(sym)) {
                return false;
            }
            deb_outs.push(m_rule_factory->create_debug_out(loc, sym));

            while (accept(TK_COMMA)) {
                Location comma_loc = curr_loc();
                if (!parse_identifier(sym)) {
                    return false;
                }
                deb_outs.push(m_rule_factory->create_debug_out(comma_loc, sym));
            }
        }
        return true;
    }

    bool parse_node(Expr *&expr)
    {
        Symbol *sym = nullptr;
        if (!parse_identifier(sym)) {
            return false;
        }

        Expr *ref = m_expr_factory->create_reference(curr_loc(), fresh_type(), sym);
        expr = ref;

        if (accept(TK_AT_OP)) {
            Location loc = curr_loc();
            if (!parse_identifier(sym)) {
                return false;
            }
            expr = m_expr_factory->create_type_annotation(loc, fresh_type(), expr, sym);
        } else if (accept(TK_TILDE)) {
            Location loc = curr_loc();
            if (!parse_node(expr)) {
                return false;
            }
            expr = m_expr_factory->create_binary(
                loc, fresh_type(), Expr_binary::OK_TILDE, ref, expr);
        } else if (m_la.kind == TK_QUESTION) {
            if (!parse_ternary_node_continuation(ref, expr)) {
                return false;
            }
        } else if (m_la.kind == TK_LPAREN) {
            if (!parse_call_node_continuation(ref, expr)) {
                return false;
            }
        }

        if (m_la.kind == TK_ATTR_BEGIN) {
            Expr_attribute::Expr_attribute_vector attrs;
            Expr *the_expr = expr;
            std::string node_name;
            if (!expect(TK_ATTR_BEGIN, "'[['") ||
                !parse_attribute_expr(attrs, node_name) ||
                !expect(TK_ATTR_END, "']]'")) {
                return false;
            }
            expr = m_expr_factory->create_attribute(
                curr_loc(), fresh_type(), the_expr, attrs, node_name.c_str());
        }
        return true;
    }

    bool parse_ternary_node_continuation(Expr *expr_cond, Expr *&expr)
    {
        Expr *expr_then = nullptr;
        Expr *expr_else = nullptr;
        if (!expect(TK_QUESTION, "'?'") ||
            !parse_node(expr_then) ||
            !expect(TK_COLON, "':'") ||
            !parse_node(expr_else)) {
            return false;
        }
        expr = m_expr_factory->create_conditional(expr_cond, expr_then, expr_else);
        return true;
    }

    bool parse_call_node_continuation(Expr *expr_callee, Expr *&expr)
    {
        Expr_call *call_expr = m_expr_factory->create_call(fresh_type(), expr_callee);
        if (!expect(TK_LPAREN, "'('")) {
            return false;
        }
        if (m_la.kind != TK_RPAREN) {
            if (!parse_paramlist(call_expr)) {
                return false;
            }
        }
        if (!expect(TK_RPAREN, "')'")) {
            return false;
        }
        expr = call_expr;
        return true;
    }

    bool parse_paramlist(Expr_call *call_expr)
    {
        Expr *expr = nullptr;
        if (!parse_node(expr)) {
            return false;
        }
        call_expr->add_argument(m_expr_factory->create_argument(expr));

        while (accept(TK_COMMA)) {
            if (!parse_node(expr)) {
                return false;
            }
            call_expr->add_argument(m_expr_factory->create_argument(expr));
        }
        return true;
    }

    bool parse_attribute_expr(
        Expr_attribute::Expr_attribute_vector &attrs,
        std::string                          &node_name)
    {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%d", m_gensym_counter++);
        node_name = "node_result_";
        node_name += buf;

        Expr_attribute::Expr_attribute_entry entry;
        if (!parse_attr_entry(entry)) {
            return false;
        }
        attrs.push_back(entry);

        while (accept(TK_COMMA)) {
            if (!parse_attr_entry(entry)) {
                return false;
            }
            attrs.push_back(entry);
        }
        return true;
    }

    bool parse_attr_entry(Expr_attribute::Expr_attribute_entry &entry)
    {
        Symbol *sym = nullptr;
        if (!parse_identifier(sym)) {
            return false;
        }

        entry.name = sym;
        entry.type = fresh_type();
        entry.expr = nullptr;
        entry.is_pattern = true;

        if (accept(TK_ASSIGN)) {
            entry.is_pattern = false;
            return parse_expression(entry.expr);
        }
        if (accept(TK_TILDE)) {
            return parse_expression(entry.expr);
        }
        return true;
    }

    bool parse_expression(Expr *&expr)
    {
        if (!parse_logical_or_expr(expr)) {
            return false;
        }

        if (accept(TK_QUESTION)) {
            Expr *expr_cond = expr;
            Expr *expr_then = nullptr;
            Expr *expr_else = nullptr;
            if (!parse_expression(expr_then) ||
                !expect(TK_COLON, "':'") ||
                !parse_expression(expr_else)) {
                return false;
            }
            expr = m_expr_factory->create_conditional(expr_cond, expr_then, expr_else);
        }
        return true;
    }

    bool parse_logical_or_expr(Expr *&expr)
    {
        if (!parse_logical_and_expr(expr)) {
            return false;
        }
        while (m_la.kind == TK_LOGICAL_OR) {
            Expr *left = expr;
            Expr *right = nullptr;
            Location loc = curr_loc();
            if (!expect(TK_LOGICAL_OR, "'||'") ||
                !parse_logical_and_expr(right)) {
                return false;
            }
            expr = m_expr_factory->create_binary(
                loc, fresh_type(), Expr_binary::OK_LOGICAL_OR, left, right);
        }
        return true;
    }

    bool parse_logical_and_expr(Expr *&expr)
    {
        if (!parse_inclusive_or_expr(expr)) {
            return false;
        }
        while (m_la.kind == TK_LOGICAL_AND) {
            Expr *left = expr;
            Expr *right = nullptr;
            Location loc = curr_loc();
            if (!expect(TK_LOGICAL_AND, "'&&'") ||
                !parse_inclusive_or_expr(right)) {
                return false;
            }
            expr = m_expr_factory->create_binary(
                loc, fresh_type(), Expr_binary::OK_LOGICAL_AND, left, right);
        }
        return true;
    }

    bool parse_inclusive_or_expr(Expr *&expr)
    {
        if (!parse_exclusive_or_expr(expr)) {
            return false;
        }
        while (m_la.kind == TK_PIPE) {
            Expr *left = expr;
            Expr *right = nullptr;
            Location loc = curr_loc();
            if (!expect(TK_PIPE, "'|'") ||
                !parse_exclusive_or_expr(right)) {
                return false;
            }
            expr = m_expr_factory->create_binary(
                loc, fresh_type(), Expr_binary::OK_BITWISE_OR, left, right);
        }
        return true;
    }

    bool parse_exclusive_or_expr(Expr *&expr)
    {
        if (!parse_and_expr(expr)) {
            return false;
        }
        while (m_la.kind == TK_CARET) {
            Expr *left = expr;
            Expr *right = nullptr;
            Location loc = curr_loc();
            if (!expect(TK_CARET, "'^'") ||
                !parse_and_expr(right)) {
                return false;
            }
            expr = m_expr_factory->create_binary(
                loc, fresh_type(), Expr_binary::OK_BITWISE_XOR, left, right);
        }
        return true;
    }

    bool parse_and_expr(Expr *&expr)
    {
        if (!parse_equality_expr(expr)) {
            return false;
        }
        while (m_la.kind == TK_AMP) {
            Expr *left = expr;
            Expr *right = nullptr;
            Location loc = curr_loc();
            if (!expect(TK_AMP, "'&'") ||
                !parse_equality_expr(right)) {
                return false;
            }
            expr = m_expr_factory->create_binary(
                loc, fresh_type(), Expr_binary::OK_BITWISE_AND, left, right);
        }
        return true;
    }

    bool parse_equality_expr(Expr *&expr)
    {
        if (!parse_relational_expr(expr)) {
            return false;
        }
        while (m_la.kind == TK_EQUAL || m_la.kind == TK_NOT_EQUAL) {
            Expr *left = expr;
            Expr *right = nullptr;
            Expr_binary::Operator op = Expr_binary::OK_EQUAL;
            Location loc = curr_loc();
            if (!parse_equality_operator(op) ||
                !parse_relational_expr(right)) {
                return false;
            }
            expr = m_expr_factory->create_binary(loc, fresh_type(), op, left, right);
        }
        return true;
    }

    bool parse_equality_operator(Expr_binary::Operator &op)
    {
        if (accept(TK_EQUAL)) {
            op = Expr_binary::OK_EQUAL;
            return true;
        }
        if (accept(TK_NOT_EQUAL)) {
            op = Expr_binary::OK_NOT_EQUAL;
            return true;
        }
        set_error("expected equality operator", m_la.line, m_la.column);
        return false;
    }

    bool parse_relational_expr(Expr *&expr)
    {
        if (!parse_shift_expr(expr)) {
            return false;
        }
        while (m_la.kind == TK_LESS_EQUAL || m_la.kind == TK_GREATER_EQUAL ||
               m_la.kind == TK_LESS || m_la.kind == TK_GREATER) {
            Expr *left = expr;
            Expr *right = nullptr;
            Expr_binary::Operator op = Expr_binary::OK_LESS;
            Location loc = curr_loc();
            if (!parse_relational_operator(op) ||
                !parse_shift_expr(right)) {
                return false;
            }
            expr = m_expr_factory->create_binary(loc, fresh_type(), op, left, right);
        }
        return true;
    }

    bool parse_relational_operator(Expr_binary::Operator &op)
    {
        if (accept(TK_LESS_EQUAL)) {
            op = Expr_binary::OK_LESS_OR_EQUAL;
            return true;
        }
        if (accept(TK_GREATER_EQUAL)) {
            op = Expr_binary::OK_GREATER_OR_EQUAL;
            return true;
        }
        if (accept(TK_LESS)) {
            op = Expr_binary::OK_LESS;
            return true;
        }
        if (accept(TK_GREATER)) {
            op = Expr_binary::OK_GREATER;
            return true;
        }
        set_error("expected relational operator", m_la.line, m_la.column);
        return false;
    }

    bool parse_shift_expr(Expr *&expr)
    {
        if (!parse_additive_expr(expr)) {
            return false;
        }
        while (m_la.kind == TK_SHIFT_LEFT ||
               m_la.kind == TK_SHIFT_RIGHT ||
               m_la.kind == TK_SHIFT_RIGHT_ARITH) {
            Expr *left = expr;
            Expr *right = nullptr;
            Expr_binary::Operator op = Expr_binary::OK_SHIFT_LEFT;
            Location loc = curr_loc();
            if (!parse_shift_operator(op) ||
                !parse_additive_expr(right)) {
                return false;
            }
            expr = m_expr_factory->create_binary(loc, fresh_type(), op, left, right);
        }
        return true;
    }

    bool parse_shift_operator(Expr_binary::Operator &op)
    {
        if (accept(TK_SHIFT_LEFT)) {
            op = Expr_binary::OK_SHIFT_LEFT;
            return true;
        }
        if (accept(TK_SHIFT_RIGHT)) {
            op = Expr_binary::OK_SHIFT_RIGHT;
            return true;
        }
        if (accept(TK_SHIFT_RIGHT_ARITH)) {
            op = Expr_binary::OK_SHIFT_RIGHT_ARITH;
            return true;
        }
        set_error("expected shift operator", m_la.line, m_la.column);
        return false;
    }

    bool parse_additive_expr(Expr *&expr)
    {
        if (!parse_multiplicative_expr(expr)) {
            return false;
        }
        while (m_la.kind == TK_PLUS || m_la.kind == TK_MINUS) {
            Expr *left = expr;
            Expr *right = nullptr;
            Expr_binary::Operator op = Expr_binary::OK_PLUS;
            Location loc = curr_loc();
            if (!parse_additive_operator(op) ||
                !parse_multiplicative_expr(right)) {
                return false;
            }
            expr = m_expr_factory->create_binary(loc, fresh_type(), op, left, right);
        }
        return true;
    }

    bool parse_additive_operator(Expr_binary::Operator &op)
    {
        if (accept(TK_PLUS)) {
            op = Expr_binary::OK_PLUS;
            return true;
        }
        if (accept(TK_MINUS)) {
            op = Expr_binary::OK_MINUS;
            return true;
        }
        set_error("expected additive operator", m_la.line, m_la.column);
        return false;
    }

    bool parse_multiplicative_expr(Expr *&expr)
    {
        if (!parse_unary_expr(expr)) {
            return false;
        }
        while (m_la.kind == TK_STAR || m_la.kind == TK_SLASH || m_la.kind == TK_PERCENT) {
            Expr *left = expr;
            Expr *right = nullptr;
            Expr_binary::Operator op = Expr_binary::OK_MULTIPLY;
            Location loc = curr_loc();
            if (!parse_multiplicative_operator(op) ||
                !parse_unary_expr(right)) {
                return false;
            }
            expr = m_expr_factory->create_binary(loc, fresh_type(), op, left, right);
        }
        return true;
    }

    bool parse_multiplicative_operator(Expr_binary::Operator &op)
    {
        if (accept(TK_STAR)) {
            op = Expr_binary::OK_MULTIPLY;
            return true;
        }
        if (accept(TK_SLASH)) {
            op = Expr_binary::OK_DIVIDE;
            return true;
        }
        if (accept(TK_PERCENT)) {
            op = Expr_binary::OK_MODULO;
            return true;
        }
        set_error("expected multiplicative operator", m_la.line, m_la.column);
        return false;
    }

    bool parse_unary_expr(Expr *&expr)
    {
        Expr *expr_sub = nullptr;

        if (accept(TK_INC_OP)) {
            if (!parse_unary_expr(expr_sub)) {
                return false;
            }
            expr = m_expr_factory->create_unary(
                curr_loc(), fresh_type(), Expr_unary::OK_PRE_INCREMENT, expr_sub);
            return true;
        }

        if (accept(TK_DEC_OP)) {
            if (!parse_unary_expr(expr_sub)) {
                return false;
            }
            expr = m_expr_factory->create_unary(
                curr_loc(), fresh_type(), Expr_unary::OK_PRE_DECREMENT, expr_sub);
            return true;
        }

        if (is_unary_operator_token(m_la.kind)) {
            Expr_unary::Operator op = Expr_unary::OK_POSITIVE;
            if (!parse_unary_operator(op) ||
                !parse_unary_expr(expr_sub)) {
                return false;
            }
            expr = m_expr_factory->create_unary(curr_loc(), fresh_type(), op, expr_sub);
            return true;
        }

        return parse_postfix_expr(expr);
    }

    bool is_unary_operator_token(Token_kind kind) const
    {
        return kind == TK_PLUS || kind == TK_MINUS || kind == TK_TILDE || kind == TK_BANG;
    }

    bool parse_unary_operator(Expr_unary::Operator &op)
    {
        if (accept(TK_PLUS)) {
            op = Expr_unary::OK_POSITIVE;
            return true;
        }
        if (accept(TK_MINUS)) {
            op = Expr_unary::OK_NEGATIVE;
            return true;
        }
        if (accept(TK_TILDE)) {
            op = Expr_unary::OK_BITWISE_COMPLEMENT;
            return true;
        }
        if (accept(TK_BANG)) {
            op = Expr_unary::OK_LOGICAL_NOT;
            return true;
        }
        set_error("expected unary operator", m_la.line, m_la.column);
        return false;
    }

    bool parse_postfix_expr(Expr *&expr)
    {
        if (!parse_primary_expr(expr)) {
            return false;
        }

        while (m_la.kind == TK_LBRACKET || m_la.kind == TK_DOT ||
               m_la.kind == TK_INC_OP || m_la.kind == TK_DEC_OP ||
               m_la.kind == TK_AT_OP) {
            if (!parse_postfix(expr)) {
                return false;
            }
        }
        return true;
    }

    bool parse_postfix(Expr *&expr)
    {
        Expr *expr_in = expr;
        Location loc = curr_loc();

        if (accept(TK_LBRACKET)) {
            if (!parse_expression(expr) ||
                !expect(TK_RBRACKET, "']'")) {
                return false;
            }
            expr = m_expr_factory->create_binary(
                loc, fresh_type(), Expr_binary::OK_ARRAY_SUBSCRIPT, expr_in, expr);
            return true;
        }

        if (accept(TK_DOT)) {
            Symbol *sym = nullptr;
            if (!parse_identifier(sym)) {
                return false;
            }
            expr = m_expr_factory->create_reference(curr_loc(), fresh_type(), sym);
            expr = m_expr_factory->create_binary(
                loc, fresh_type(), Expr_binary::OK_SELECT, expr_in, expr);
            return true;
        }

        if (accept(TK_INC_OP)) {
            expr = m_expr_factory->create_unary(
                curr_loc(), fresh_type(), Expr_unary::OK_POST_INCREMENT, expr_in);
            return true;
        }

        if (accept(TK_DEC_OP)) {
            expr = m_expr_factory->create_unary(
                curr_loc(), fresh_type(), Expr_unary::OK_POST_DECREMENT, expr_in);
            return true;
        }

        if (accept(TK_AT_OP)) {
            Symbol *sym = nullptr;
            if (!parse_identifier(sym)) {
                return false;
            }
            expr = m_expr_factory->create_type_annotation(loc, fresh_type(), expr_in, sym);
            return true;
        }

        set_error("expected postfix operator", m_la.line, m_la.column);
        return false;
    }

    bool parse_primary_expr(Expr *&expr)
    {
        if (is_literal_token(m_la.kind)) {
            if (!parse_literal(expr)) {
                return false;
            }
        } else if (m_la.kind == TK_IDENT) {
            if (!parse_reference_or_call(expr)) {
                return false;
            }
        } else if (accept(TK_LPAREN)) {
            if (!parse_expression(expr) ||
                !expect(TK_RPAREN, "')'")) {
                return false;
            }
            expr->mark_parenthesis();
        } else if (accept(TK_OPTION)) {
            if (!expect(TK_LPAREN, "'('")) {
                return false;
            }

            Location loc = curr_loc();
            Symbol *sym = nullptr;
            if (!parse_identifier(sym)) {
                return false;
            }

            expr = m_expr_factory->create_reference(loc, fresh_type(), sym);
            if (accept(TK_AT_OP)) {
                Location annotation_loc = curr_loc();
                if (!parse_identifier(sym)) {
                    return false;
                }
                expr = m_expr_factory->create_type_annotation(
                    annotation_loc, fresh_type(), expr, sym);
            }

            expr = m_expr_factory->create_unary(
                curr_loc(), m_type_factory->get_bool(), Expr_unary::OK_OPTION, expr);
            if (!expect(TK_RPAREN, "')'")) {
                return false;
            }
        } else {
            set_error("expected expression", m_la.line, m_la.column);
            return false;
        }

        if (m_la.kind == TK_ATTR_BEGIN) {
            Expr_attribute::Expr_attribute_vector attrs;
            Expr *the_expr = expr;
            std::string node_name;
            if (!expect(TK_ATTR_BEGIN, "'[['") ||
                !parse_attribute_expr(attrs, node_name) ||
                !expect(TK_ATTR_END, "']]'")) {
                return false;
            }
            expr = m_expr_factory->create_attribute(
                curr_loc(), fresh_type(), the_expr, attrs, node_name.c_str());
        }
        return true;
    }

    bool is_literal_token(Token_kind kind) const
    {
        return kind == TK_INTEGER_LITERAL ||
            kind == TK_FLOATING_LITERAL ||
            kind == TK_TRUE ||
            kind == TK_FALSE ||
            kind == TK_STRING_LITERAL;
    }

    static unsigned integer_value(char const *val)
    {
        unsigned base = 0;
        unsigned res = 0;

        char const *s = val;
        if (*s == '0') {
            ++s;
            if (*s == 'x' || *s == 'X') {
                ++s;
                base = 16;
            } else if (*s == 'b' || *s == 'B') {
                ++s;
                base = 2;
            } else {
                base = 8;
            }
        } else {
            base = 10;
        }

        for (;;) {
            unsigned digit = 16;
            switch (*s) {
            case '0': digit = 0; break;
            case '1': digit = 1; break;
            case '2': digit = 2; break;
            case '3': digit = 3; break;
            case '4': digit = 4; break;
            case '5': digit = 5; break;
            case '6': digit = 6; break;
            case '7': digit = 7; break;
            case '8': digit = 8; break;
            case '9': digit = 9; break;
            case 'a': case 'A': digit = 10; break;
            case 'b': case 'B': digit = 11; break;
            case 'c': case 'C': digit = 12; break;
            case 'd': case 'D': digit = 13; break;
            case 'e': case 'E': digit = 14; break;
            case 'f': case 'F': digit = 15; break;
            default: return res;
            }
            res *= base;
            res += digit;
            ++s;
        }
    }

    bool parse_literal(Expr *&expr)
    {
        if (accept(TK_INTEGER_LITERAL)) {
            unsigned u = integer_value(m_curr.text.c_str());
            Value *v = m_value_factory->get_int(int(u));
            expr = m_expr_factory->create_literal(curr_loc(), v);
            return true;
        }
        if (accept(TK_FLOATING_LITERAL)) {
            char *end = nullptr;
            errno = 0;
            float value = std::strtof(m_curr.text.c_str(), &end);
            (void)end;
            Value *v = m_value_factory->get_float(value, m_curr.text.c_str());
            expr = m_expr_factory->create_literal(curr_loc(), v);
            return true;
        }
        if (accept(TK_TRUE)) {
            Value *v = m_value_factory->get_bool(true);
            expr = m_expr_factory->create_literal(curr_loc(), v);
            return true;
        }
        if (accept(TK_FALSE)) {
            Value *v = m_value_factory->get_bool(false);
            expr = m_expr_factory->create_literal(curr_loc(), v);
            return true;
        }
        if (accept(TK_STRING_LITERAL)) {
            Value *v = m_value_factory->get_string(m_curr.text.c_str());
            expr = m_expr_factory->create_literal(curr_loc(), v);
            return true;
        }

        set_error("expected literal", m_la.line, m_la.column);
        return false;
    }

    bool parse_reference_or_call(Expr *&expr)
    {
        Symbol *sym = nullptr;
        if (!parse_identifier(sym)) {
            return false;
        }

        expr = m_expr_factory->create_reference(curr_loc(), fresh_type(), sym);

        if (m_la.kind == TK_LPAREN) {
            Type *return_type = fresh_type();
            Expr_call *call_expr = m_expr_factory->create_call(return_type, expr);
            if (!expect(TK_LPAREN, "'('")) {
                return false;
            }
            if (m_la.kind != TK_RPAREN) {
                if (!parse_arglist(call_expr)) {
                    return false;
                }
            }
            if (!expect(TK_RPAREN, "')'")) {
                return false;
            }
            expr = call_expr;
        }
        return true;
    }

    bool parse_arglist(Expr_call *call_expr)
    {
        Expr *expr = nullptr;
        if (!parse_expression(expr)) {
            return false;
        }
        call_expr->add_argument(m_expr_factory->create_argument(expr));

        while (accept(TK_COMMA)) {
            if (!parse_expression(expr)) {
                return false;
            }
            call_expr->add_argument(m_expr_factory->create_argument(expr));
        }
        return true;
    }

    bool parse_postcond_or_expr(Expr *&expr)
    {
        if (!parse_postcond_and_expr(expr)) {
            return false;
        }
        while (m_la.kind == TK_LOGICAL_OR) {
            Expr *left = expr;
            Expr *right = nullptr;
            Location loc = curr_loc();
            if (!expect(TK_LOGICAL_OR, "'||'") ||
                !parse_postcond_and_expr(right)) {
                return false;
            }
            expr = m_expr_factory->create_binary(
                loc, fresh_type(), Expr_binary::OK_LOGICAL_OR, left, right);
        }
        return true;
    }

    bool parse_postcond_and_expr(Expr *&expr)
    {
        if (!parse_postcond_primary_expr(expr)) {
            return false;
        }
        while (m_la.kind == TK_LOGICAL_AND) {
            Expr *left = expr;
            Expr *right = nullptr;
            Location loc = curr_loc();
            if (!expect(TK_LOGICAL_AND, "'&&'") ||
                !parse_postcond_primary_expr(right)) {
                return false;
            }
            expr = m_expr_factory->create_binary(
                loc, fresh_type(), Expr_binary::OK_LOGICAL_AND, left, right);
        }
        return true;
    }

    bool parse_postcond_primary_expr(Expr *&expr)
    {
        if (accept(TK_NONODE)) {
            Symbol *sym = nullptr;
            if (!expect(TK_LPAREN, "'('") ||
                !parse_identifier(sym)) {
                return false;
            }
            Expr *ref = m_expr_factory->create_reference(curr_loc(), fresh_type(), sym);
            expr = m_expr_factory->create_unary(
                curr_loc(), fresh_type(), Expr_unary::OK_NONODE, ref);
            return expect(TK_RPAREN, "')'");
        }

        if (accept(TK_MATCH)) {
            if (!expect(TK_LPAREN, "'('") ||
                !parse_node(expr) ||
                !expect(TK_RPAREN, "')'")) {
                return false;
            }
            expr = m_expr_factory->create_unary(
                curr_loc(), fresh_type(), Expr_unary::OK_MATCH, expr);
            return true;
        }

        if (accept(TK_LPAREN)) {
            if (!parse_postcond_or_expr(expr) ||
                !expect(TK_RPAREN, "')'")) {
                return false;
            }
            expr->mark_parenthesis();
            return true;
        }

        set_error("expected postcondition expression", m_la.line, m_la.column);
        return false;
    }

private:
    Compilation_unit &m_unit;
    Scanner           m_scanner;
    Token             m_curr;
    Token             m_la;
    bool              m_error;
    std::string       m_error_message;
    unsigned          m_error_line;
    unsigned          m_error_column;
    Expr_factory     *m_expr_factory;
    Type_factory     *m_type_factory;
    Value_factory    *m_value_factory;
    Rule_factory     *m_rule_factory;
    Symbol_table     *m_symtab;
    int               m_gensym_counter;
};

}  // namespace

bool mdltlc_parse_rd(
    Compilation_unit &unit,
    char const       *source,
    size_t            length,
    std::string      &error_message,
    unsigned         &error_line,
    unsigned         &error_column)
{
    Parser parser(unit, source, length);
    if (parser.parse()) {
        return true;
    }

    error_message = parser.error_message();
    error_line = parser.error_line();
    error_column = parser.error_column();
    return false;
}
