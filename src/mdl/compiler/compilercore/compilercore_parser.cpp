/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "compilercore_errors.h"

// include the generated parser from compilercore_parser.atg
#include "Parser.cpp"

namespace mi {
namespace mdl {

void Parser::set_module(Module *module, bool enable_mdl_next, bool enable_experimental)
{
    m_module = module;
    m_enable_mdl_next = enable_mdl_next;
    m_enable_experimental = enable_experimental;
    m_name_factory = m_module->get_name_factory();
    m_type_factory = m_module->get_type_factory();
    m_expression_factory = m_module->get_expression_factory();
    m_value_factory = m_module->get_value_factory();
    m_statement_factory = m_module->get_statement_factory();
    m_declaration_factory = m_module->get_declaration_factory();
    m_annotation_factory = m_module->get_annotation_factory();

    m_last_identifier_unicode = false;
    m_unicode_identifiers_supported = false;
    m_unicode_error_emitted = false;
    m_trailing_comma_allowed = false;
}

// Parses an expression instead of a full MDL document.
IExpression const *Parser::parse_expression()
{
    IExpression const *expr = NULL;

    t = NULL;
    la = dummyToken = scanner->CreateToken();
    la->val = "Dummy Token";
    Get();
    expression(expr);
    Expect(0);

    return expr;
}

// Check for ',' !TOK.
bool Parser::non_trailing_comma(int tok)
{
    scanner->ResetPeek();
    Token *x = la;
    if (x->kind != TOK_COMMA) {
        return false;
    }
    x = scanner->Peek();
    return x->kind != tok;
}

// Check if trailing comma is allowed and the next token is a comma.
bool Parser::trailing_comma_allowed()
{
    if (!m_trailing_comma_allowed) {
        return false;
    }
    return see_comma();
}

// Skip square brackets.
bool Parser::skip_square_brackets(Token *&x)
{
    for (;;) {
        x = scanner->Peek();
        if (x->kind == TOK_LEFT_SQUARE_BRACKET) {
            if (!skip_square_brackets(x))
                return false;
        } else if (x->kind == TOK_RIGHT_SQUARE_BRACKET) {
            x = scanner->Peek();
            return true;
        } else if (x->kind == TOK_EOF) {
            break;
        }
    }
    return false;
}

// Skip a type.
bool Parser::skip_type(Token *&x)
{
    if ((x->kind == TOK_UNIFORM) || (x->kind == TOK_VARYING)) {
        x = scanner->Peek();
    }

    if (x->kind == TOK_AUTO) {
        x = scanner->Peek();
        return true;
    }

    switch (x->kind) {
    case TOK_SCOPE:
        x = scanner->Peek();
        return skip_type(x);
    case TOK_IDENT:
        x = scanner->Peek();
        if (x->kind == TOK_SCOPE) {
            x = scanner->Peek();
            return skip_type(x);
        } else if (x->kind == TOK_LEFT_SQUARE_BRACKET) {
            return skip_square_brackets(x);
        }
        return true;
    case TOK_BOOL:
    case TOK_BOOL2:
    case TOK_BOOL3:
    case TOK_BOOL4:
    case TOK_INT:
    case TOK_INT2:
    case TOK_INT3:
    case TOK_INT4:
    case TOK_FLOAT:
    case TOK_FLOAT2:
    case TOK_FLOAT3:
    case TOK_FLOAT4:
    case TOK_FLOAT2X2:
    case TOK_FLOAT2X3:
    case TOK_FLOAT2X4:
    case TOK_FLOAT3X2:
    case TOK_FLOAT3X3:
    case TOK_FLOAT3X4:
    case TOK_FLOAT4X2:
    case TOK_FLOAT4X3:
    case TOK_FLOAT4X4:
    case TOK_DOUBLE:
    case TOK_DOUBLE2:
    case TOK_DOUBLE3:
    case TOK_DOUBLE4:
    case TOK_DOUBLE2X2:
    case TOK_DOUBLE2X3:
    case TOK_DOUBLE2X4:
    case TOK_DOUBLE3X2:
    case TOK_DOUBLE3X3:
    case TOK_DOUBLE3X4:
    case TOK_DOUBLE4X2:
    case TOK_DOUBLE4X3:
    case TOK_DOUBLE4X4:
    case TOK_COLOR:
    case TOK_STRING:
    case TOK_BSDF:
    case TOK_EDF:
    case TOK_VDF:
    case TOK_LIGHT_PROFILE:
    case TOK_TEXTURE_2D:
    case TOK_TEXTURE_3D:
    case TOK_TEXTURE_CUBE:
    case TOK_TEXTURE_PTEX:
    case TOK_BSDF_MEASUREMENT:
    case TOK_INTENSITY_MODE:
    case TOK_MATERIAL:
    case TOK_MATERIAL_EMISSION:
    case TOK_MATERIAL_GEOMETRY:
    case TOK_MATERIAL_SURFACE:
    case TOK_MATERIAL_VOLUME:
    case TOK_HAIR_BSDF:
        x = scanner->Peek();
        if (x->kind == TOK_LEFT_SQUARE_BRACKET)
            return skip_square_brackets(x);
        return true;
    default:
        return false;
    }
}

// If we see either "import" or "using" or "export" followed by "import" or "using"
// during lookahead, we assume that the declaration is an import declaration.
bool Parser::is_import_declaration()
{
    scanner->ResetPeek();
    Token *x = la;
    if (x->kind == TOK_EXPORT) {
        x = scanner->Peek();
    }
    return (x->kind == TOK_IMPORT) || (x->kind == TOK_USING);
}

// Check for ident '='
bool Parser::is_namespace_alias()
{
    scanner->ResetPeek();
    Token *x = la;
    if (x->kind != TOK_IDENT) {
        return false;
    }
    x = scanner->Peek();
    return x->kind == TOK_EQUAL;
}

// If we see a type followed by an identifier during lookahead,
// we assume that this is a declaration.
bool Parser::is_declaration()
{
    scanner->ResetPeek();
    Token *x = la;
    if (!skip_type(x))
        return false;
    return (x->kind == TOK_IDENT) || (x->kind == TOK_ANNOTATION_BLOCK_BEGIN);
}

// Check for C-style cast: We have already seen '(' expr '), check for start(postfix_expr)
bool Parser::is_c_style_cast()
{
    scanner->ResetPeek();
    Token *x = la;
    if (x->kind != TOK_LEFT_PARENTHESIS)
        return false;
    x = scanner->Peek();
    if (!skip_type(x))
        return false;
    if (x->kind != TOK_RIGHT_PARENTHESIS)
        return false;
    x = scanner->Peek();

    switch (x->kind) {
    // start of unary expression
    case TOK_TILDE:
    case TOK_BANG:
    case TOK_PLUS:
    case TOK_MINUS:
    case TOK_INC_OP:
    case TOK_DEC_OP:
    case TOK_CAST:
    // start of primary expression (subset of unary)
    case TOK_TRUE:
    case TOK_FALSE:
    case TOK_INTEGER_LITERAL:
    case TOK_FLOATING_LITERAL:
    case TOK_FRACT_LITERAL:
    case TOK_STRING_LITERAL:
    case TOK_SCOPE:
    case TOK_BOOL:
    case TOK_BOOL2:
    case TOK_BOOL3:
    case TOK_BOOL4:
    case TOK_INT:
    case TOK_INT2:
    case TOK_INT3:
    case TOK_INT4:
    case TOK_FLOAT:
    case TOK_FLOAT2:
    case TOK_FLOAT3:
    case TOK_FLOAT4:
    case TOK_FLOAT2X2:
    case TOK_FLOAT2X3:
    case TOK_FLOAT2X4:
    case TOK_FLOAT3X2:
    case TOK_FLOAT3X3:
    case TOK_FLOAT3X4:
    case TOK_FLOAT4X2:
    case TOK_FLOAT4X3:
    case TOK_FLOAT4X4:
    case TOK_DOUBLE:
    case TOK_DOUBLE2:
    case TOK_DOUBLE3:
    case TOK_DOUBLE4:
    case TOK_DOUBLE2X2:
    case TOK_DOUBLE2X3:
    case TOK_DOUBLE2X4:
    case TOK_DOUBLE3X2:
    case TOK_DOUBLE3X3:
    case TOK_DOUBLE3X4:
    case TOK_DOUBLE4X2:
    case TOK_DOUBLE4X3:
    case TOK_DOUBLE4X4:
    case TOK_COLOR:
    case TOK_STRING:
    case TOK_BSDF:
    case TOK_EDF:
    case TOK_VDF:
    case TOK_LIGHT_PROFILE:
    case TOK_TEXTURE_2D:
    case TOK_TEXTURE_3D:
    case TOK_TEXTURE_CUBE:
    case TOK_TEXTURE_PTEX:
    case TOK_BSDF_MEASUREMENT:
    case TOK_INTENSITY_MODE:
    case TOK_MATERIAL:
    case TOK_MATERIAL_EMISSION:
    case TOK_MATERIAL_GEOMETRY:
    case TOK_MATERIAL_SURFACE:
    case TOK_MATERIAL_VOLUME:
    case TOK_HAIR_BSDF:
    case TOK_IDENT:
        return true;
    case TOK_LEFT_PARENTHESIS:
        // while '(' is a start of a primary expression, do not treat it as a c-style cast
        // here:
        // (t)(f) *is* legal
        return false;
    default:
        return false;
    }
}

// If we see an identifier followed by a colon during lookahead,
// we assume that the argument is named.
bool Parser::is_named_argument()
{
    scanner->ResetPeek();
    Token *x = la;
    if (x->kind == TOK_COMMA)
        x = scanner->Peek();
    if (x->kind == TOK_IDENT) {
        x = scanner->Peek();
        return x->kind == TOK_COLON;
    } else {
        return false;
    }
}

// Check if the next two tokens are coupled right brackets
bool Parser::is_anno_block_end()
{
    scanner->ResetPeek();
    if (la->kind == TOK_RIGHT_SQUARE_BRACKET) {
        int line = la->line;
        int col = la->col;
        Token *x = scanner->Peek();
        if (x->kind != TOK_RIGHT_SQUARE_BRACKET)
            return false;
        return line == x->line && col + 1 == x->col;
    }
    return false;
}

// Check if the next two tokens are '[' and ']
bool Parser::is_array_constructor()
{
    scanner->ResetPeek();
    if (la->kind == TOK_LEFT_SQUARE_BRACKET) {
        Token *x = scanner->Peek();
        if (x->kind == TOK_RIGHT_SQUARE_BRACKET)
            return true;
    }
    return false;
}

// Check if the next three tokens are '(', '*', and ')'.
bool Parser::is_clone()
{
    scanner->ResetPeek();
    if (la->kind == TOK_LEFT_PARENTHESIS) {
        Token *x = scanner->Peek();
        if (x->kind == TOK_STAR) {
            x = scanner->Peek();
            if (x->kind == TOK_RIGHT_PARENTHESIS)
                return true;
        }
    }
    return false;
}

// Check if we see '::' and IDENT/UNICODE_IDENT
bool Parser::is_scope_name()
{
    scanner->ResetPeek();
    Token *x = la;
    if (x->kind != TOK_SCOPE)
        return false;
    x = scanner->Peek();
    return x->kind == TOK_IDENT || (x->kind == TOK_UNICODE_IDENT && unicode_identifier_support());
}

// Check if we see '::' and '..'.
bool Parser::is_scope_dotdot()
{
    scanner->ResetPeek();
    Token *x = la;
    if (x->kind != TOK_SCOPE)
        return false;
    x = scanner->Peek();
    return x->kind == TOK_DOTDOT;
}

void Parser::unicode_identifiers_not_supported_error(Token const *t) {
    if (!m_unicode_error_emitted) {
        m_unicode_error_emitted = true;
        errors->Error(t->line, t->col,
            UNICODE_IDENTIFIERS_NOT_SUPPORTED,
            Error_params(m_alloc));
    }
}

ISimple_name const *Parser::to_simple(Token const *t)
{
    ISymbol const *symbol = errDist > 0 ?
        m_name_factory->create_symbol(t->val) :
        m_name_factory->get_error_symbol();
    return m_name_factory->create_simple_name(
        symbol,
        t->line,
        t->col,
        t->line,
        t->col + utf8_len(t->val) - 1);
}

ISimple_name const *Parser::to_simple(ISimple_name const *name)
{
    if (name == NULL) {
        ISymbol const *symbol = m_name_factory->get_error_symbol();
        return m_name_factory->create_simple_name(
            symbol,
            t->line,
            t->col,
            t->line,
            t->col);
    }
    return name;
}

// Create a simple name from a utf8 string and a position.
ISimple_name const *Parser::to_simple(char const *utf8_s, int sl, int sc, int el, int ec)
{
    ISymbol const *symbol = m_name_factory->create_symbol(utf8_s);
    return m_name_factory->create_simple_name(
        symbol, sl, sc, el, ec);
}

// Create a qualified name from a simple name.
IQualified_name *Parser::to_qualified(ISimple_name const *simple_name)
{
    IQualified_name *qualified_name = m_name_factory->create_qualified_name();
    qualified_name->add_component(to_simple(simple_name));
    return qualified_name;
}

// Create a qualified error name.
IQualified_name *Parser::qualified_error()
{
    ISymbol const *symbol = m_name_factory->create_symbol("<ERROR>");
    ISimple_name const *simple_name = m_name_factory->create_simple_name(
        symbol,
        t->line,
        t->col,
        t->line,
        t->col);
    return to_qualified(simple_name);
}

// Create a type name from a type name prefix and a token.
IType_name *Parser::to_type(IType_name *prefix, Token *tok)
{
    IType_name *type_name = prefix;
    IQualified_name *qualified_name;

    if (type_name) {
        qualified_name = type_name->get_qualified_name();
    } else {
        qualified_name = m_name_factory->create_qualified_name();
        type_name = m_name_factory->create_type_name(qualified_name, tok->line, tok->col);
    }
    qualified_name->add_component(to_simple(tok));
    Position &tpos = type_name->access_position();
    Position const &qpos = qualified_name->access_position();
    tpos.set_end_line(qpos.get_end_line());
    tpos.set_end_column(qpos.get_end_column());
    return type_name;
}

// Create a type name from a type name prefix and a token.
IType_name *Parser::to_unicode_type(IType_name *prefix, Token *tok)
{
    IType_name *type_name = prefix;
    IQualified_name *qualified_name;

    if (type_name) {
        qualified_name = type_name->get_qualified_name();
    } else {
        qualified_name = m_name_factory->create_qualified_name();
        type_name = m_name_factory->create_type_name(qualified_name, tok->line, tok->col);
    }
    string utf8(tok->val, m_alloc);
    int sl = tok->line;
    int sc = tok->col;
    int el = tok->col;
    int ec = tok->col + utf8_len(tok->val) - 1;

    utf8 = convert_escape_sequences_skip_quotes(t->val);
    //tmp = tmp.substr(1, tmp.size() - 2);

    ISimple_name const *simple_name = to_simple(utf8.c_str(), sl, sc, el, ec);
    qualified_name->add_component(simple_name);
    Position &tpos = type_name->access_position();
    Position const &qpos = qualified_name->access_position();
    tpos.set_end_line(qpos.get_end_line());
    tpos.set_end_column(qpos.get_end_column());
    return type_name;
}

// Create a type name from a simple name.
IType_name *Parser::to_type(ISimple_name const *simple_name)
{
    IQualified_name *qualified_name = to_qualified(simple_name);
    IType_name *type_name = m_name_factory->create_type_name(qualified_name);
    Position &q_pos = qualified_name->access_position();
    Position &t_pos = type_name->access_position();
    t_pos.set_start_line(q_pos.get_start_line());
    t_pos.set_start_column(q_pos.get_start_column());
    t_pos.set_end_line(q_pos.get_end_line());
    t_pos.set_end_column(q_pos.get_end_column());
    return type_name;
}

// Create an "auto" typename
IType_name *Parser::auto_type(Token *tok)
{
    return to_type(NULL, tok);
}

// Create a reference from a type name.
IExpression_reference const *Parser::to_reference(
    IType_name const *type_name,
    bool             is_array_con)
{
    IExpression_reference *result = m_expression_factory->create_reference(type_name);
    if (is_array_con) {
        result->set_array_constructor();
    }
    Position const &t_pos = type_name->access_position();
    Position &r_pos = result->access_position();
    r_pos.set_start_line(t_pos.get_start_line());
    r_pos.set_start_column(t_pos.get_start_column());
    r_pos.set_end_line(t_pos.get_end_line());
    r_pos.set_end_column(t_pos.get_end_column());
    return result;
}

// Create a reference from a simple name.
IExpression_reference const *Parser::to_reference(ISimple_name const *simple_name)
{
    return to_reference(to_type(simple_name));
}

// Create a statement from an expression.
IStatement_expression const *Parser::to_statement(IExpression const *expr)
{
    expr = check_expr(expr);
    IStatement_expression *result = m_statement_factory->create_expression(expr);
    if (expr != NULL) {
        Position const &pos = expr->access_position();
        Position &r_pos = result->access_position();
        r_pos.set_start_line(pos.get_start_line());
        r_pos.set_start_column(pos.get_start_column());
        r_pos.set_end_line(pos.get_end_line());
        r_pos.set_end_column(pos.get_end_column());
    }
    return result;
}

// Create an integer value.
char const *Parser::integer_value(char const *val, bool &overflow, unsigned long &value)
{
    int base = 0;
    unsigned long res = 0;
    unsigned long maxv = 0, maxd = 0;
    overflow = false;

    char const *s = val;
    if (*s == '0') {
        ++s;
        if (*s == 'x' || *s == 'X') {
            ++s;
            base = 16;
            maxv = 0x10000000;  // 0x10000000 * 16 + 0 = 0x100000000
        } else if (*s == 'b' || *s == 'B') {
            ++s;
            base = 2;
            maxv = 0x80000000;  // 0x80000000 *  2 + 0 = 0x100000000
        } else {
            base = 8;
            maxv = 0x20000000;  // 0x20000000 *  8 + 0 = 0x100000000
        }
    } else {
        base = 10;
        maxv = 0x19999999;      // 0x19999999 * 10 + 6 = 0x100000000
        maxd = 6;
    }

    for (;;) {
        unsigned long digit = 16;
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
        case 'a': digit = 10; break;
        case 'b': digit = 11; break;
        case 'c': digit = 12; break;
        case 'd': digit = 13; break;
        case 'e': digit = 14; break;
        case 'f': digit = 15; break;
        case 'A': digit = 10; break;
        case 'B': digit = 11; break;
        case 'C': digit = 12; break;
        case 'D': digit = 13; break;
        case 'E': digit = 14; break;
        case 'F': digit = 15; break;
        default: goto out;
        }

        // beware of C++ semantics: gcc will optimize simple overflow tests away :-(
        if (res >= maxv && digit >= maxd) {
            overflow = true;
        }

        res *= base;
        res += digit;
        ++s;
    }
out:
    /* ignore overflow on sign bit, this cannot be checked reliable here
    if (res >= 0x80000000) {
        // overflow on sign bit
        overflow = true;
    }
    */
    value = res;
    return s;
}

// Check an expression.
IType_name *Parser::check_type(IType_name *type) {
    if (type == ERR_TYPE) {
        IQualified_name *q_name = qualified_error();
        IType_name      *result = m_name_factory->create_type_name(q_name);
        Position const  &t_pos  = q_name->access_position();
        Position        &r_pos  = result->access_position();
        r_pos.set_start_line(t_pos.get_start_line());
        r_pos.set_start_column(t_pos.get_start_column());
        r_pos.set_end_line(t_pos.get_end_line());
        r_pos.set_end_column(t_pos.get_end_column());
        return result;
    }
    return type;
}

    // Create an unary expression.
IExpression_unary *Parser::create_unary(
    IExpression_unary::Operator const op,
    IExpression const                 *argument)
{
    argument = check_expr(argument);
    Position const    &pos  = argument->access_position();
    IExpression_unary *expr = m_expression_factory->create_unary(
        op,
        argument,
        pos.get_start_line(),
        pos.get_start_column(),
        pos.get_end_line(),
        pos.get_end_column());
    return expr;
}

// Create a binary expression.
IExpression_binary *Parser::create_binary(
    IExpression_binary::Operator const op,
    IExpression const                  *left,
    IExpression const                  *right)
{
    left  = check_expr(left);
    right = check_expr(right);
    IExpression_binary *expr = m_expression_factory->create_binary(op, left, right);
    Position const &l_pos = left->access_position();
    Position const &r_pos = right->access_position();
    Position &e_pos = expr->access_position();
    e_pos.set_start_line(l_pos.get_start_line());
    e_pos.set_start_column(l_pos.get_start_column());
    e_pos.set_end_line(r_pos.get_end_line());
    e_pos.set_end_column(r_pos.get_end_column());
    return expr;
}

// Create a conditional expression.
IExpression_conditional *Parser::create_conditional(
    IExpression const *cond,
    IExpression const *true_expr,
    IExpression const *false_expr)
{
    cond       = check_expr(cond);
    true_expr  = check_expr(true_expr);
    false_expr = check_expr(false_expr);
    Position const &s_pos = cond->access_position();
    Position const &e_pos = false_expr->access_position();
    return m_expression_factory->create_conditional(
        cond,
        true_expr,
        false_expr,
        s_pos.get_start_line(),
        s_pos.get_start_column(),
        e_pos.get_end_line(),
        e_pos.get_end_column());
}

// Create a positional argument.
IArgument_positional const *Parser::create_positional_argument(IExpression const *expr)
{
    expr = check_expr(expr);
    Position const &e_pos = expr->access_position();
    return m_expression_factory->create_positional_argument(
            expr,
            e_pos.get_start_line(),
            e_pos.get_start_column(),
            e_pos.get_end_line(),
            e_pos.get_end_column());
}

    // Create a named argument.
IArgument_named const *Parser::create_named_argument(
    ISimple_name const *parameter_name,
    IExpression const *expr)
{
    expr = check_expr(expr);
    Position const &p_pos = parameter_name->access_position();
    Position const &e_pos = expr->access_position();
    return m_expression_factory->create_named_argument(
        parameter_name,
        expr,
        p_pos.get_start_line(),
        p_pos.get_start_column(),
        e_pos.get_end_line(),
        e_pos.get_end_column());
}

// Create a parameter.
IParameter const *Parser::create_parameter(
    int                     sl,
    int                     sc,
    IType_name const        *type_name,
    ISimple_name const      *name,
    IExpression const       *init,
    IAnnotation_block const *annotations)
{
    init = check_expr(init);
    Position const &t_pos = type_name->access_position();
    int start_line        = sl ? sl : t_pos.get_start_line();
    int start_column      = sc ? sc : t_pos.get_start_column();
    int end_line          = t_pos.get_end_line();
    int end_column        = t_pos.get_end_column();
    if (annotations != NULL) {
        Position const &a_pos = annotations->access_position();
        end_line   = a_pos.get_end_line();
        end_column = a_pos.get_end_column();
    } else if (init != NULL) {
        Position const &i_pos = init->access_position();
        end_line   = i_pos.get_end_line();
        end_column = i_pos.get_end_column();
    }
    IParameter const *param = m_declaration_factory->create_parameter(
                                    type_name,name,init,annotations,
                                    start_line,start_column,end_line,end_column);
    return param;
}

// Create a declaration.
IStatement_declaration *Parser::create_declaration(IDeclaration const *decl)
{
    decl = check_decl(decl);
    IStatement_declaration *stmt  = m_statement_factory->create_declaration(decl);
    Position const         &d_pos = decl->access_position();
    Position               &s_pos = stmt->access_position();
    s_pos.set_start_line(d_pos.get_start_line());
    s_pos.set_start_column(d_pos.get_start_column());
    s_pos.set_end_line(d_pos.get_end_line());
    s_pos.set_end_column(d_pos.get_end_column());
    return stmt;
}

// Add a declaration.
void Parser::add_declaration(IDeclaration const *decl)
{
    decl = check_decl(decl);
    m_module->add_declaration(decl);
}

// Add an annotation.
void Parser::add_annotation(
    IAnnotation_block *&annotations,
    IAnnotation const *anno)
{
    // just ignore wrong annotations
    if (anno != NULL) {
        annotations->add_annotation(anno);
    }
}

// Mark that an expression was in parenthesis.
void Parser::mark_parenthesis(IExpression const *expr)
{
    if (expr != ERR_EXPR) {
        const_cast<IExpression *>(expr)->mark_parenthesis();
    }
}

// Convert escape sequences in string literals and skip quotes.
string Parser::convert_escape_sequences_skip_quotes(char const *s)
{
    size_t l = strlen(s);  // need byte length here
    string res(m_alloc);
    res.reserve(l - 2);

    char const *p = s + 1;     // skip first quote
    char const *e = s + l - 1; // skip last quote

    while (p < e) {
        unsigned cp = utf8_next(p);
        if (cp == '\\') {
            char const *old_p = p;
            cp = utf8_next(p);
            switch (cp) {
            case 'a':  utf8_append(res, '\a'); break;
            case 'b':  utf8_append(res, '\b'); break;
            case 'f':  utf8_append(res, '\f'); break;
            case 'n':  utf8_append(res, '\n'); break;
            case 'r':  utf8_append(res, '\r'); break;
            case 't':  utf8_append(res, '\t'); break;
            case 'v':  utf8_append(res, '\v'); break;
            case '\\': utf8_append(res, '\\'); break;
            case '\'': utf8_append(res, '\''); break;
            case '"':  utf8_append(res, '"'); break;
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
                {
                    unsigned code = 0;
                    bool exit = false;
                    do {
                        switch (cp) {
                        case '0': code = code * 8 + 0; break;
                        case '1': code = code * 8 + 1; break;
                        case '2': code = code * 8 + 2; break;
                        case '3': code = code * 8 + 3; break;
                        case '4': code = code * 8 + 4; break;
                        case '5': code = code * 8 + 5; break;
                        case '6': code = code * 8 + 6; break;
                        case '7': code = code * 8 + 7; break;
                        default: p = old_p; exit = true; break;
                        }
                        if (!exit) {
                            old_p = p;
                            cp = utf8_next(p);
                        }
                        } while (!exit);
                        utf8_append(res, code);
                }
                break;
            case 'x':
            case 'u':
            case 'U':
                {
                    char coding = char(cp);
                    unsigned code = 0;
                    unsigned range = 0xFFFFFFFF;
                    unsigned enforce_digits = 0;

                    if (coding == 'x') {
                        range = 0xFF;
                    } else if (coding == 'u') {
                        range = 0xFFFF;
                        enforce_digits = 4;
                    } else {
                        // 'U'
                        enforce_digits = 8;
                    }
                    bool exit = false;
                    bool overrun = false;
                    char const *start = p;
                    char const *old_p = p;
                    cp = utf8_next(p);
                    if (!isxdigit(cp)) {
                        errors->Error(
                            t->line,
                            int(t->col + p - s),
                            ESCAPE_USED_WITHOUT_HEX_DIDITS,
                            Error_params(m_alloc).add_char(coding)
                        );
                        utf8_append(res, cp);
                    } else {
                        unsigned num_digits = 0;
                        for (;;) {
                            switch (cp) {
                            case '0': code = code * 16 + 0; break;
                            case '1': code = code * 16 + 1; break;
                            case '2': code = code * 16 + 2; break;
                            case '3': code = code * 16 + 3; break;
                            case '4': code = code * 16 + 4; break;
                            case '5': code = code * 16 + 5; break;
                            case '6': code = code * 16 + 6; break;
                            case '7': code = code * 16 + 7; break;
                            case '8': code = code * 16 + 8; break;
                            case '9': code = code * 16 + 9; break;
                            case 'a':
                            case 'A': code = code * 16 + 10; break;
                            case 'b':
                            case 'B': code = code * 16 + 11; break;
                            case 'c':
                            case 'C': code = code * 16 + 12; break;
                            case 'd':
                            case 'D': code = code * 16 + 13; break;
                            case 'e':
                            case 'E': code = code * 16 + 14; break;
                            case 'f':
                            case 'F': code = code * 16 + 15; break;
                            default: p = old_p; exit = true; break;
                            }
                            if (code > range) {
                                overrun = true;
                            }
                            if (exit) {
                                break;
                            }
                            ++num_digits;
                            if (num_digits == enforce_digits) {
                                break;
                            }
                            old_p = p;
                            cp = utf8_next(p);
                        }
                        if (num_digits < enforce_digits) {
                            string bad_code(start, p, m_alloc);
                            errors->Error(
                                t->line,
                                int(t->col + p - s),
                                INCOMPLETE_UNIVERSAL_CHARACTER_NAME,
                                Error_params(m_alloc).add_char(coding).add(bad_code)
                            );
                        } else if (coding == 'u' || coding == 'U') {
                            // Because surrogate code points are not Unicode scalar values,
                            // any UTF-8 byte sequence that would otherwise map to
                            // code points U+D800..U+DFFF is illformed
                            bool error = 0xD800 <= code && code <= 0xDFFF;

                            error |= code > COCO_UNICODE_MAX;
                            if (error) {
                                string bad_code(start, p, m_alloc);
                                errors->Error(
                                    t->line,
                                    int(t->col + p - s),
                                    INVALID_UNIVERSAL_CHARACTER_ENCODING,
                                    Error_params(m_alloc).add_char(coding).add(bad_code)
                                );
                                 // replacement character
                                code = 0xFFFD;
                            }
                        }
                        if (overrun) {
                            string bad_code(start, p, m_alloc);
                            errors->Warning(
                                t->line,
                                int(t->col + p - start),
                                ESCAPE_SEQUENCE_OUT_OF_RANGE,
                                Error_params(m_alloc).add_char(coding).add(bad_code)
                            );
                        }
                        utf8_append(res, code);
                    }
                }
                break;
            case '\0':
                utf8_append(res, cp); break;
            default:
                {
                    char buf[32];
                    snprintf(buf, sizeof(buf), "Unknown escape sequence '\\%c'", cp);
                    errors->Warning(t->line, int(t->col + p - s), buf);
                    utf8_append(res, cp);
                }
                break;
            }
        } else {
            utf8_append(res, cp);
        }
    }
    return res;
}


}  // compiler
}  // mdl
