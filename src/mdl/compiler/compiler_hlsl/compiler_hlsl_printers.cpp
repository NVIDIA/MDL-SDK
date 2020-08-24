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

#include <cstdarg>
#include <cstdio>

#include <mdl/compiler/compilercore/compilercore_streams.h>

#include "compiler_hlsl_assert.h"
#include "compiler_hlsl_compilation_unit.h"
#include "compiler_hlsl_definitions.h"
#include "compiler_hlsl_symbols.h"
#include "compiler_hlsl_types.h"
#include "compiler_hlsl_exprs.h"
#include "compiler_hlsl_values.h"
#include "compiler_hlsl_printers.h"
#include "compiler_hlsl_tools.h"

#ifdef MI_PLATFORM_WINDOWS
#  define FMT_BIT64 "I64"
#  define FMT_BIT32 ""
#else
#  define FMT_BIT64 "ll"
#  define FMT_BIT32 ""
#endif

namespace mi {
namespace mdl {
namespace hlsl {

// on x86 and NV hardware, quiet NaNs are encoded with a "is silent" bit
#define QNAN_BIT    0x00400000
#define QNAN_VALUE  0x7FC00000


inline bool isfinite(double v)
{
    return -HUGE_VAL < v && v < HUGE_VAL;
}

/// Decode a float values as an unsigned integer.
unsigned int decode_value_as_uint(float value)
{
    union {
        float s;
        unsigned int t;
    } bin_conv;
    bin_conv.s = value;
    return bin_conv.t;
}

bool is_encoded_value(float value)
{
    return (int)decode_value_as_uint(value) > (int)QNAN_VALUE;
}

#define C_PREPROCESSOR C_ANNOTATION

Printer::Printer(IAllocator *alloc, IOutput_stream *ostr)
: Base(alloc)
, m_indent(0)
, m_color_output(false)
, m_enable_loc(false)
, m_last_file_id(~0u)
, m_ostr(ostr, mi::base::DUP_INTERFACE)
, m_c_ostr()
, m_color_stack()
, m_curr_unit(NULL)
{
    ::memset(m_priority_map, 0, sizeof(m_priority_map));

    int prio = 1;

    // priority 17: sequence
    m_priority_map[Expr::OK_SEQUENCE]                        = prio;
    ++prio;

    // priority 16: assignment, arithmetic assignments
    m_priority_map[Expr::OK_ASSIGN]                          = prio;
    m_priority_map[Expr::OK_MULTIPLY_ASSIGN]                 = prio;
    m_priority_map[Expr::OK_DIVIDE_ASSIGN]                   = prio;
    m_priority_map[Expr::OK_MODULO_ASSIGN]                   = prio;
    m_priority_map[Expr::OK_PLUS_ASSIGN]                     = prio;
    m_priority_map[Expr::OK_MINUS_ASSIGN]                    = prio;
    m_priority_map[Expr::OK_BITWISE_AND_ASSIGN]              = prio;
    m_priority_map[Expr::OK_BITWISE_OR_ASSIGN]               = prio;
    m_priority_map[Expr::OK_BITWISE_XOR_ASSIGN]              = prio;
    m_priority_map[Expr::OK_SHIFT_LEFT_ASSIGN]               = prio;
    m_priority_map[Expr::OK_SHIFT_RIGHT_ASSIGN]              = prio;
    ++prio;

    // priority 15: selection
    m_priority_map[Expr::OK_TERNARY]                         = prio;
    ++prio;

    // priority 14: logical exclusive or
    m_priority_map[Expr::OK_LOGICAL_OR]                      = prio;
    ++prio;

    // priority 13: logical exclusive or
    m_priority_map[Expr::OK_LOGICAL_XOR]                     = prio;
    ++prio;

    // priority 12: logical and
    m_priority_map[Expr::OK_LOGICAL_AND]                     = prio;
    ++prio;

    // priority 11: bit-wise inclusive or
    m_priority_map[Expr::OK_BITWISE_OR]                      = prio;
    ++prio;

    // priority 10: bit-wise exclusive or
    m_priority_map[Expr::OK_BITWISE_XOR]                     = prio;
    ++prio;

    // priority 9: bit-wise and
    m_priority_map[Expr::OK_BITWISE_AND]                     = prio;
    ++prio;

    // priority 8: equality
    m_priority_map[Expr::OK_EQUAL]                           = prio;
    m_priority_map[Expr::OK_NOT_EQUAL]                       = prio;
    ++prio;

    // priority 7: relational
    m_priority_map[Expr::OK_LESS]                            = prio;
    m_priority_map[Expr::OK_LESS_OR_EQUAL]                   = prio;
    m_priority_map[Expr::OK_GREATER]                         = prio;
    m_priority_map[Expr::OK_GREATER_OR_EQUAL]                = prio;
    ++prio;

    // priority 6: bit-wise shift
    m_priority_map[Expr::OK_SHIFT_LEFT]                      = prio;
    m_priority_map[Expr::OK_SHIFT_RIGHT]                     = prio;
    ++prio;

    // priority 5: additive
    m_priority_map[Expr::OK_PLUS]                            = prio;
    m_priority_map[Expr::OK_MINUS]                           = prio;
    ++prio;

    // priority 4: multiplicative
    m_priority_map[Expr::OK_MULTIPLY]                        = prio;
    m_priority_map[Expr::OK_DIVIDE]                          = prio;
    m_priority_map[Expr::OK_MODULO]                          = prio;
    ++prio;

    // priority 3.5: typecast
    m_priority_map[Expr::OK_TYPECAST]                        = prio;
    ++prio;

    // priority 3: prefix increment and decrement, unary
    m_priority_map[Expr::OK_PRE_INCREMENT]                   = prio;
    m_priority_map[Expr::OK_PRE_DECREMENT]                   = prio;
    m_priority_map[Expr::OK_POSITIVE]                        = prio;
    m_priority_map[Expr::OK_NEGATIVE]                        = prio;
    m_priority_map[Expr::OK_BITWISE_COMPLEMENT]              = prio;
    m_priority_map[Expr::OK_LOGICAL_NOT]                     = prio;
    m_priority_map[Expr::OK_POINTER_DEREF]                   = prio;
    ++prio;

    // priority 2: array subscript, function call and constructor structure,
    // field or method selector, swizzle,  post fix increment and decrement
    // field access from pointer
    m_priority_map[Expr::OK_POST_INCREMENT]                  = prio;
    m_priority_map[Expr::OK_POST_DECREMENT]                  = prio;
    m_priority_map[Expr::OK_SELECT]                          = prio;
    m_priority_map[Expr::OK_ARRAY_SUBSCRIPT]                 = prio;
    m_priority_map[Expr::OK_CALL]                            = prio;
    m_priority_map[Expr::OK_ARROW]                           = prio;
    ++prio;
}

// Format print.
void Printer::printf(char const *format, ...)
{
    char buffer[1024];
    va_list ap;
    va_start(ap, format);
    vsnprintf(buffer, sizeof(buffer), format, ap);
    buffer[sizeof(buffer) - 1] = '\0';
    va_end(ap);
    m_ostr->write(buffer);
}

// Prints a newline and do indentation.
void Printer::nl(unsigned count)
{
    for (unsigned i = 0; i < count; ++i)
        print('\n');
    indent(m_indent);
}

// Un-Indent output.
void Printer::un_indent(int depth)
{
    for (int i = 0; i < depth; ++i) {
        m_ostr->unput(' ');
        m_ostr->unput(' ');
        m_ostr->unput(' ');
        m_ostr->unput(' ');
    }
}

// Set the given color.
void Printer::color(ISyntax_coloring::Syntax_elements code)
{
    if (m_color_output) {
        if (code == C_DEFAULT) {
            if (m_color_stack.empty()) {
                m_c_ostr->reset_color();
                return;
            }
            code = m_color_stack.top();
        }
        Color_entry const &c = m_color_table[code];
        m_c_ostr->set_color(c.fg_color, c.fg_bold, /*background=*/false);
        m_c_ostr->set_color(c.bg_color, c.bg_bold, /*background=*/true);
    }
}

// Set the given color and push it on the color stack.
void Printer::push_color(ISyntax_coloring::Syntax_elements code)
{
    if (m_color_output) {
        m_color_stack.push(code);
        color(code);
    }
}

// Remove one entry from the color stack.
void Printer::pop_color()
{
    if (m_color_output) {
        m_color_stack.pop();
        color(C_DEFAULT);
    }
}

// Print a keyword.
void Printer::keyword(char const *w)
{
    push_color(C_KEYWORD);
    print(w);
    pop_color();
}

// Print a literal.
void Printer::literal(char const *w)
{
    push_color(C_LITERAL);
    print(w);
    pop_color();
}

// Print a type part.
void Printer::typepart(char const *w)
{
    push_color(C_TYPE);
    print(w);
    pop_color();
}

// Returns the priority of an operator.
int Printer::get_priority(int op) const
{
    HLSL_ASSERT(0 <= op && op <= Expr::OK_LAST);
    return m_priority_map[op];
}

// Indent output.
void Printer::indent(int depth)
{
    for (int i = 0; i < depth; ++i)
        print("    ");
}

// Print string.
void Printer::print(const char *string)
{
    m_ostr->write(string);
}

// Print character.
void Printer::print(char c)
{
    printf("%c", c);
}

// Print boolean.
void Printer::print(bool b)
{
    print(b ? "true" : "false");
}

// Print 32bit integer.
void Printer::print(int32_t n)
{
    printf("%d", n);
}

// Print 32bit unsigned integer.
void Printer::print(uint32_t n)
{
    printf("%u", n);
}

// Print 64bit integer.
void Printer::print(int64_t n)
{
    printf("%" FMT_BIT64 "d", n);
}

// Print 64bit unsigned integer.
void Printer::print(uint64_t n)
{
    printf("%" FMT_BIT64 "u", n);
}

// Print double.
void Printer::print(double d)
{
    printf("%.16g", d);
}

// Print a symbol.
void Printer::print(Symbol *sym)
{
    print(sym->get_name());
}

// Print type.
void Printer::print(Type *type)
{
    push_color(C_TYPE);
    print_type(type);
    pop_color();
}

// Print a location.
void Printer::print_location(Location const &loc)
{
    if (m_enable_loc) {
        if (loc.get_line() != 0) {
            un_indent(m_indent);
            push_color(C_PREPROCESSOR);
            printf("#line %u", loc.get_line());

            if (m_curr_unit != NULL && loc.get_file_id() != m_last_file_id) {
                m_last_file_id = loc.get_file_id();
                char const *fname = m_curr_unit->get_filename_by_id(m_last_file_id);

                if (fname != NULL)
                    printf(" \"%s\"", fname);
            }
            pop_color();
            nl();
        }
    }
}

// Print a type with current color.
void Printer::print_type(Type *type, Symbol *name)
{
    if (Type_function *f_type = as<Type_function>(type)) {
        Type *ret_type = f_type->get_return_type();

        print_type(ret_type);

        print(" ");
        if (name != NULL)
            print(name);
        else
            print("(*)");
        print("(");

        for (size_t i = 0, n = f_type->get_parameter_count(); i < n; ++i) {
            if (i > 0)
                print(", ");

            Type_function::Parameter *param = f_type->get_parameter(i);
            switch (param->get_modifier()) {
            case Type_function::Parameter::PM_IN:
                break;
            case Type_function::Parameter::PM_OUT:
                keyword("out");
                print(" ");
                break;
            case Type_function::Parameter::PM_INOUT:
                keyword("inout");
                print(" ");
                break;
            }
            print(param->get_type());
        }
        print(")");
    } else {
        Type::Modifiers m = type->get_type_modifiers();
        if (m & Type::MK_CONST) {
            keyword("const");
            print(" ");
        }
        if (m & Type::MK_COL_MAJOR) {
            keyword("col_major");
            print(" ");
        }
        if (m & Type::MK_ROW_MAJOR) {
            keyword("row_major");
            print(" ");
        }
        print(type->get_sym());
    }
}

// Print a definition.
void Printer::print(Definition const *def)
{
    if (Def_function const *op_def = as<Def_function>(def)) {
        print_type(op_def->get_type(), op_def->get_symbol());
    } else if (Def_operator const *op_def = as<Def_operator>(def)) {
        print_type(op_def->get_type(), op_def->get_symbol());
    } else {
        print_type(def->get_type());
        print(" ");
        print(def->get_symbol());
    }
}

/// Ensures that the ascii representation of a float constant
/// has a '.' and add the given format character.
static char *to_float_constant(char *s)
{
    bool add_dot = true;
    char *end = s + strlen(s);
    for (char *p = end - 1; p >= s; --p) {
        if (*p == '.' || *p == 'e' || *p == 'E') {
            add_dot = false;
            break;
        }
    }
    if (add_dot) {
        *end++ = '.';
        *end++ = '0';
    }
    *end++ = '\0';

    return s;
}

// Print a value.
void Printer::print_value(Value *value)
{
    push_color(C_LITERAL);
    switch (value->get_kind()) {
    case Value::VK_BAD:
        push_color(C_ERROR);
        print("<BAD>");
        pop_color();
        break;
    case Value::VK_VOID:
        print("(void)0");
        break;
    case Value::VK_BOOL:
        {
            Value_bool *v = cast<Value_bool>(value);

            print(v->get_value());
            break;
        }
    case Value::VK_INT:
        {
            Value_int_32 *v = cast<Value_int_32>(value);

            print(v->get_value());
            break;
        }
    case Value::VK_UINT:
        {
            Value_uint_32 *v = cast<Value_uint_32>(value);

            print(v->get_value());
            print("u");
            break;
        }
    case Value::VK_MIN12INT:
        {
            Value_int_12 *v = cast<Value_int_12>(value);

            print(v->get_value());
            break;
        }
    case Value::VK_MIN16INT:
        {
            Value_int_16 *v = cast<Value_int_16>(value);

            print(v->get_value());
            break;
        }
    case Value::VK_MIN16UINT:
        {
            Value_uint_16 *v = cast<Value_uint_16>(value);

            print(v->get_value());
            break;
        }
    case Value::VK_HALF:
        {
            Value_half *v = cast<Value_half>(value);
            char buf[64];

            float f = v->get_value();
            if (isfinite(f)) {
                snprintf(buf, sizeof(buf) - 2, "%.7g", f);
                buf[sizeof(buf) - 3] = '\0';
                print(to_float_constant(buf));
                print('h');
            } else if (f == +HUGE_VAL) {
                print("(+1.0h/0.0h)");
            } else if (f == -HUGE_VAL) {
                print("(-1.0h/0.0h)");
            } else {
                print("(0.0h/0.0h)");
            }
            break;
        }
    case Value::VK_FLOAT:
        {
            Value_float *v = cast<Value_float>(value);
            char buf[64];

            float f = v->get_value();
            if (isfinite(f)) {
                snprintf(buf, sizeof(buf) - 2, "%.7g", f);
                buf[sizeof(buf) - 3] = '\0';
                print(to_float_constant(buf));
            } else if (f == +HUGE_VAL) {
                print("(+1.0/0.0)");
            } else if (f == -HUGE_VAL) {
                print("(-1.0/0.0)");
            } else if (is_encoded_value(f)) {
                print("asfloat(");
                print(decode_value_as_uint(f));
                print("u)");
            } else {
                print("(0.0/0.0)");
            }
            break;
        }
    case Value::VK_DOUBLE:
        {
            Value_double *v = cast<Value_double>(value);
            char buf[64];

            double d = v->get_value();
            if (isfinite(d)) {
                snprintf(buf, sizeof(buf) - 2, "%.16g", d);
                buf[sizeof(buf) - 3] = '\0';
                print(to_float_constant(buf));
                print('l');
            } else if (d == +HUGE_VAL) {
                print("(+1.0l/0.0l)");
            } else if (d == -HUGE_VAL) {
                print("(-1.0l/0.0l)");
            } else {
                print("(0.0l/0.0l)");
            }
            break;
        }
    case Value::VK_VECTOR:
    case Value::VK_MATRIX:
        {
            Value_compound *v = cast<Value_compound>(value);

            print_type(v->get_type());
            print("(");
            for (size_t i = 0, n = v->get_component_count(); i < n; ++i) {
                if (i > 0)
                    print(", ");
                print(v->get_value(i));
            }
            print(")");
            break;
        }
    case Value::VK_ARRAY:
    case Value::VK_STRUCT:
        {
            Value_compound *v = cast<Value_compound>(value);

            print("{ ");
            for (size_t i = 0, n = v->get_component_count(); i < n; ++i) {
                if (i > 0)
                    print(", ");
                print_value(v->get_value(i));
            }
            print(" }");
            break;
        }
    }
    pop_color();
}

// Print value.
void Printer::print(Value *value)
{
    print_value(value);
}

// Print expression.
void Printer::print(Expr const *expr)
{
    print(expr, /*priority=*/0);
}

/// Check if an extra parenthesis around a given expression is necessary to silence
/// HLSL compiler warnings if the parent expression is a binary operation op.
static bool need_extra_paren(
    Expr_binary::Operator op,
    Expr const            *expr)
{
    if (op == Expr_binary::OK_BITWISE_AND || op == Expr_binary::OK_BITWISE_OR) {
        if (Expr_binary const *bexpr = as<Expr_binary>(expr)) {
            return bexpr->get_operator() != op;
        }
    }
    return false;
}

// Print expression.
void Printer::print(
    Expr const *expr,
    int         priority)
{
    if (expr->in_parenthesis())
        print("(");
    switch (expr->get_kind()) {
    case Expr::EK_INVALID:
        push_color(C_ERROR);
        print("<ERROR>");
        pop_color();
        break;
    case Expr::EK_LITERAL:
        {
            Expr_literal const *lit = cast<Expr_literal>(expr);
            print(lit->get_value());
        }
        break;
    case Expr::EK_REFERENCE:
        {
            Expr_ref const *ref = cast<Expr_ref>(expr);

            push_color(C_ENTITY);
            print(ref->get_name());
            pop_color();
        }
        break;
    case Expr::EK_UNARY:
        {
            Expr_unary const     *uexpr      = cast<Expr_unary>(expr);
            Expr_unary::Operator op          = uexpr->get_operator();
            int                  op_priority = get_priority(op);

            if (op_priority <= priority)
                print("(");

            const char *prefix = NULL, *postfix = NULL;
            switch (op) {
            case Expr_unary::OK_BITWISE_COMPLEMENT:
                prefix = "~";
                break;
            case Expr_unary::OK_LOGICAL_NOT:
                prefix = "!";
                break;
            case Expr_unary::OK_POSITIVE:
                prefix = "+";
                break;
            case Expr_unary::OK_NEGATIVE:
                prefix = "-";
                break;
            case Expr_unary::OK_PRE_INCREMENT:
                prefix = "++";
                break;
            case Expr_unary::OK_PRE_DECREMENT:
                prefix = "--";
                break;
            case Expr_unary::OK_POST_INCREMENT:
                postfix = "++";
                break;
            case Expr_unary::OK_POST_DECREMENT:
                postfix = "--";
                break;
            case Expr_unary::OK_POINTER_DEREF:
                prefix = "*";
                break;
            }

            if (prefix != NULL)
                keyword(prefix);
            print(uexpr->get_argument(), op_priority);
            if (postfix != NULL)
                keyword(postfix);

            if (op_priority <= priority)
                print(")");
        }
        break;
    case Expr::EK_BINARY:
        {
            Expr_binary const     *bexpr      = cast<Expr_binary>(expr);
            Expr_binary::Operator op          = bexpr->get_operator();
            int                   op_priority = get_priority(op);

            if (op_priority < priority)
                print("(");

            Expr const *left = bexpr->get_left_argument();
            bool left_needs_paren = need_extra_paren(op, left);
            print(left, left_needs_paren ? 255 : op_priority);

            char const *infix = NULL, *postfix = NULL;
            switch (op) {
            case Expr_binary::OK_SELECT:
                infix = ".";
                break;
            case Expr_binary::OK_ARROW:
                infix = "->";
                break;
            case Expr_binary::OK_ARRAY_SUBSCRIPT:
                infix = "["; postfix = "]";
                break;
            case Expr_binary::OK_MULTIPLY:
                infix = " * ";
                break;
            case Expr_binary::OK_DIVIDE:
                infix = " / ";
                break;
            case Expr_binary::OK_MODULO:
                infix = " % ";
                break;
            case Expr_binary::OK_PLUS:
                infix = " + ";
                break;
            case Expr_binary::OK_MINUS:
                infix = " - ";
                break;
            case Expr_binary::OK_SHIFT_LEFT:
                infix = " << ";
                break;
            case Expr_binary::OK_SHIFT_RIGHT:
                infix = " >> ";
                break;
            case Expr_binary::OK_LESS:
                infix = " < ";
                break;
            case Expr_binary::OK_LESS_OR_EQUAL:
                infix = " <= ";
                break;
            case Expr_binary::OK_GREATER_OR_EQUAL:
                infix = " >= ";
                break;
            case Expr_binary::OK_GREATER:
                infix = " > ";
                break;
            case Expr_binary::OK_EQUAL:
                infix = " == ";
                break;
            case Expr_binary::OK_NOT_EQUAL:
                infix = " != ";
                break;
            case Expr_binary::OK_BITWISE_AND:
                infix = " & ";
                break;
            case Expr_binary::OK_BITWISE_OR:
                infix = " | ";
                break;
            case Expr_binary::OK_BITWISE_XOR:
                infix = " ^ ";
                break;
            case Expr_binary::OK_LOGICAL_AND:
                infix = " && ";
                break;
            case Expr_binary::OK_LOGICAL_OR:
                infix = " || ";
                break;
            case Expr_binary::OK_LOGICAL_XOR:
                infix = " ^^ ";
                break;
            case Expr_binary::OK_ASSIGN:
                infix = " = ";
                break;
            case Expr_binary::OK_MULTIPLY_ASSIGN:
                infix = " *= ";
                break;
            case Expr_binary::OK_DIVIDE_ASSIGN:
                infix = " /= ";
                break;
            case Expr_binary::OK_MODULO_ASSIGN:
                infix = " %= ";
                break;
            case Expr_binary::OK_PLUS_ASSIGN:
                infix = " += ";
                break;
            case Expr_binary::OK_MINUS_ASSIGN:
                infix = " -= ";
                break;
            case Expr_binary::OK_SHIFT_LEFT_ASSIGN:
                infix = " <<= ";
                break;
            case Expr_binary::OK_SHIFT_RIGHT_ASSIGN:
                infix = " >>= ";
                break;
            case Expr_binary::OK_BITWISE_AND_ASSIGN:
                infix = " &= ";
                break;
            case Expr_binary::OK_BITWISE_XOR_ASSIGN:
                infix = " ^= ";
                break;
            case Expr_binary::OK_BITWISE_OR_ASSIGN:
                infix = " |= ";
                break;
            case Expr_binary::OK_SEQUENCE:
                infix = ", ";
                break;
            }

            if (infix) {
                print(infix);
            } else {
                print(" ");
                push_color(C_ERROR);
                print("<ERROR>");
                pop_color();
                print(" ");
            }

            // no need to put the rhs in parenthesis for the index operator
            int right_priority = op == Expr_binary::OK_ARRAY_SUBSCRIPT ? 0 : op_priority + 1;

            Expr const *right = bexpr->get_right_argument();
            bool right_needs_paren = need_extra_paren(op, right);
            print(right, right_needs_paren ? 255 : right_priority);

            if (postfix != NULL) {
                print(postfix);
            }

            if (op_priority < priority)
                print(")");
        }
        break;
    case Expr::EK_CONDITIONAL:
        {
            Expr_conditional const *cexpr      = cast<Expr_conditional>(expr);
            int                    op_priority = get_priority(Expr::OK_TERNARY);

            if (op_priority < priority)
                print("(");

            print(cexpr->get_condition(), op_priority);
            print(" ? ");
            print(cexpr->get_true(), op_priority);
            print(" : ");
            print(cexpr->get_false(), op_priority);

            if (op_priority < priority)
                print(")");
        }
        break;
    case Expr::EK_CALL:
        {
            Expr_call const *cexpr = cast<Expr_call>(expr);
            if (cexpr->is_typecast()) {
                int op_priority = get_priority(Expr::OK_TYPECAST);

                if (op_priority < priority)
                    print("(");

                print('(');
                print(cexpr->get_callee(), op_priority);
                print(')');

                Expr const *arg = cexpr->get_argument(0);
                print(arg, op_priority);

                if (op_priority < priority)
                    print(")");
            } else {
                int op_priority = get_priority(Expr::OK_CALL);

                if (op_priority < priority)
                    print("(");

                print(cexpr->get_callee(), op_priority);

                print('(');

                int arg_priority = get_priority(Expr::OK_TERNARY);
                for (size_t i = 0, n = cexpr->get_argument_count(); i < n; ++i) {
                    Expr const *arg = cexpr->get_argument(i);

                    if (i > 0)
                        print(", ");
                    print(arg, arg_priority);
                }
                print(')');

                if (op_priority < priority)
                    print(")");
            }
        }
        break;
    case Expr::EK_COMPOUND:
        {
            Expr_compound const *cexpr = cast<Expr_compound>(expr);

            print("{ ");

            int arg_priority = get_priority(Expr::OK_SEQUENCE);
            for (size_t i = 0, n = cexpr->get_element_count(); i < n; ++i) {
                Expr const *elem = cexpr->get_element(i);

                if (i > 0)
                    print(", ");

                print(elem, arg_priority);
            }
            print(" }");
        }
        break;
    }
    if (expr->in_parenthesis())
        print(")");
}

/// Print a statement.
void Printer::print_condition(Stmt const *stmt)
{
    switch (stmt->get_kind()) {
    case Stmt::SK_EXPRESSION:
        if (Expr const *expr = cast<Stmt_expr>(stmt)->get_expression()) {
            print(expr);
        }
        break;
    case Stmt::SK_DECLARATION:
        {
            Declaration const *decl = cast<Stmt_decl>(stmt)->get_declaration();
            switch (decl->get_kind()) {
            case Declaration::DK_VARIABLE:
                {
                    Declaration_variable const *vdecl = cast<Declaration_variable>(decl);

                    Type_name const *tn = vdecl->get_type_name();
                    push_color(C_TYPE);
                    print(tn);
                    pop_color();

                    if (!vdecl->empty()) {
                        print(' ');

                        bool need_comma = false;
                        for (Declaration_variable::const_iterator
                            it(vdecl->begin()), end(vdecl->end());
                            it != end;
                            ++it)
                        {
                            if (need_comma) {
                                print(", ");
                            }
                            need_comma = true;
                            Init_declarator const *init = it;
                            print(init);
                        }
                    }
                }
                break;
            default:
                // other kinds are forbidden for a condition
                push_color(C_ERROR);
                print("<ERROR>");
                pop_color();
                break;
            }
        }
        break;
    default:
        // other kinds are forbidden for a condition
        push_color(C_ERROR);
        print("<ERROR>");
        pop_color();
        break;
    }
}

// Print statement.
void Printer::print(Stmt const *stmt)
{
    print_location(stmt->get_location());

    switch (stmt->get_kind()) {
    case Stmt::SK_INVALID:
        push_color(C_ERROR);
        print("<ERROR>");
        pop_color();
        break;
    case Stmt::SK_COMPOUND:
        {
            Stmt_compound const *block = cast<Stmt_compound>(stmt);
            print('{');
            ++m_indent;

            for (Stmt_compound::const_iterator it(block->begin()), end(block->end());
                 it != end;
                 ++it)
            {
                nl();
                Stmt const *s = it;
                print(s);
            }

            --m_indent;
            nl();
            print('}');
        }
        break;
    case Stmt::SK_DECLARATION:
        {
            Declaration const *decl = cast<Stmt_decl>(stmt)->get_declaration();
            print_decl(decl, /*embedded=*/true);
            print(';');
        }
        break;
    case Stmt::SK_EXPRESSION:
        {
            if (Expr const *expr = cast<Stmt_expr>(stmt)->get_expression()) {
                print(expr);
            }
            print(';');
        }
        break;
    case Stmt::SK_IF:
        {
            Stmt_if const *istmt = cast<Stmt_if>(stmt);

            if (HLSL_attribute const *attr = istmt->get_attribute())
                print_attr(attr);

            keyword("if");
            print(" (");
            print(istmt->get_condition());
            print(')');

            Stmt const *then_stmt = istmt->get_then_statement();
            bool is_block = is<Stmt_compound>(then_stmt);

            bool need_extra_block = false;
            if (is_block) {
                print(' ');
            } else {
                if (is<Stmt_if>(then_stmt)) {
                    need_extra_block = true;
                    print(" {");
                }
                ++m_indent;
                nl();
            }
            print(then_stmt);
            if (!is_block) {
                --m_indent;
                if (need_extra_block) {
                    nl();
                    print('}');
                }
            }

            if (Stmt const *else_stmt = istmt->get_else_statement()) {
                if (is_block) {
                    print(' ');
                } else {
                    nl();
                }
                keyword("else");

                // handle else if
                is_block = is<Stmt_compound>(else_stmt) ||is<Stmt_if>(else_stmt);

                if (is_block) {
                    print(' ');
                } else {
                    ++m_indent;
                    nl();
                }
                print(else_stmt);
                if (!is_block)
                    --m_indent;
            }
        }
        break;
    case Stmt::SK_CASE:
        {
            Stmt_case const *cstmt = cast<Stmt_case>(stmt);

            if (Expr const *label = cstmt->get_label()) {
                keyword("case");
                print(' ');
                print(label);
            } else {
                keyword("default");
            }
            print(':');
        }
        break;
    case Stmt::SK_SWITCH:
        {
            Stmt_switch const *sstmt = cast<Stmt_switch>(stmt);

            if (HLSL_attribute const *attr = sstmt->get_attribute())
                print_attr(attr);

            keyword("switch");
            print(" (");
            print(sstmt->get_condition());
            print(") {");
            ++m_indent;

            for (Stmt_switch::const_iterator it(sstmt->begin()), end(sstmt->end());
                it != end;
                ++it)
            {
                Stmt const *s = it;
                bool is_case = is<Stmt_case>(s);
                if (is_case)
                    --m_indent;
                nl();
                print(s);
                if (is_case)
                    ++m_indent;
            }

            --m_indent;
            nl();
            print('}');
        }
        break;
    case Stmt::SK_WHILE:
        {
            Stmt_while const *wstmt = cast<Stmt_while>(stmt);

            if (HLSL_attribute const *attr = wstmt->get_attribute())
                print_attr(attr);

            keyword("while");
            print(" (");
            print_condition(wstmt->get_condition());
            print(')');

            Stmt const *body = wstmt->get_body();
            bool is_block = is<Stmt_compound>(body);

            if (is_block) {
                print(' ');
            } else {
                ++m_indent;
                nl();
            }
            print(body);
            if (!is_block) {
                --m_indent;
            }
        }
        break;
    case Stmt::SK_DO_WHILE:
        {
            Stmt_do_while const *wstmt = cast<Stmt_do_while>(stmt);

            if (HLSL_attribute const *attr = wstmt->get_attribute())
                print_attr(attr);

            keyword("do");

            Stmt const *body = wstmt->get_body();
            bool is_block = is<Stmt_compound>(body);

            if (is_block) {
                print(' ');
            } else {
                ++m_indent;
                nl();
            }
            print(body);
            if (is_block) {
                print(' ');
            } else {
                --m_indent;
                nl();
            }
            keyword("while");
            print(" (");
            print(wstmt->get_condition());
            print(");");
        }
        break;
    case Stmt::SK_FOR:
        {
            Stmt_for const *fstmt = cast<Stmt_for>(stmt);

            if (HLSL_attribute const *attr = fstmt->get_attribute())
                print_attr(attr);

            keyword("for");
            print(" (");
            if (Stmt const *init = fstmt->get_init()) {
                print(init);
            } else {
                print(';');
            }
            if (Stmt const *cond = fstmt->get_condition()) {
                print(' ');
                print(cond);
            } else {
                print(';');
            }
            if (Expr const *update = fstmt->get_update()) {
                print(' ');
                print(update);
            }
            print(')');

            Stmt const *body = fstmt->get_body();
            bool is_block = is<Stmt_compound>(body);

            if (is_block) {
                print(' ');
            } else {
                ++m_indent;
                nl();
            }
            print(body);
            if (!is_block) {
                --m_indent;
            }
        }
        break;
    case Stmt::SK_BREAK:
        keyword("break");
        print(';');
        break;
    case Stmt::SK_CONTINUE:
        keyword("continue");
        print(';');
        break;
    case Stmt::SK_RETURN:
        {
            Stmt_return const *rstmt = cast<Stmt_return>(stmt);

            keyword("return");
            if (Expr const *expr = rstmt->get_expression()) {
                print(' ');
                print(expr);
            }
            print(';');
        }
        break;
    case Stmt::SK_DISCARD:
        keyword("discard");
        print(';');
        break;
    }
}

// Print declaration.
void Printer::print_decl(
    Declaration const *decl,
    bool              embedded)
{
    if (!embedded)
        print_location(decl->get_location());

    switch (decl->get_kind()) {
    case Declaration::DK_INVALID:
        push_color(C_ERROR);
        print("<ERROR>");
        pop_color();
        if (!embedded)
            print(';');
        break;
    case Declaration::DK_VARIABLE:
        {
            Declaration_variable const *vdecl = cast<Declaration_variable>(decl);
            
            Type_name const *tn = vdecl->get_type_name();
            push_color(C_TYPE);
            print(tn);
            pop_color();

            if (!vdecl->empty()) {
                print(' ');

                bool need_comma = false;
                for (Declaration_variable::const_iterator it(vdecl->begin()), end(vdecl->end());
                     it != end;
                     ++it)
                {
                    if (need_comma) {
                        print(", ");
                    }
                    need_comma = true;
                    Init_declarator const *init = it;
                    print(init);
                }
            }
            if (!embedded)
                print(';');
        }
        break;
    case Declaration::DK_PARAM:
        {
            Declaration_param const *pdecl = cast<Declaration_param>(decl);

            Type_name const *tn = pdecl->get_type_name();
            push_color(C_TYPE);
            print(tn);
            pop_color();

            print(' ');
            if (Name const *name = pdecl->get_name()) {
                push_color(C_ENTITY);
                print(name);
                pop_color();

                Array_specifiers const &as = pdecl->get_array_specifiers();
                for (Array_specifiers::const_iterator it(as.begin()), end(as.end());
                    it != end;
                    ++it)
                {
                    Array_specifier const *spec = it;
                    print(spec);
                }

                if (Expr const *init = pdecl->get_default_argument()) {
                    print(" = ");
                    print(init, get_priority(Expr::OK_ASSIGN));
                }
            } else {
                push_color(C_COMMENT);
                print("/*unused*/");
                pop_color();
            }
        }
        break;
    case Declaration::DK_FUNCTION:
        {
            Declaration_function const *fdecl = cast<Declaration_function>(decl);

            Type_name const *tn = fdecl->get_ret_type();
            push_color(C_TYPE);
            print(tn);
            pop_color();
            print(' ');

            Name const *name = fdecl->get_identifier();
            push_color(C_ENTITY);
            print(name);
            pop_color();

            size_t n_params = fdecl->get_param_count();
            if (n_params == 0) {
                print('(');
                keyword("void");
                print(')');
            } else {
                bool vertical = n_params > 3;
                print('(');
                if (vertical) {
                    ++m_indent;
                    nl();
                }

                bool need_comma = false;
                for (Declaration_function::const_iterator it(fdecl->begin()), end(fdecl->end());
                     it != end;
                     ++it)
                {
                    if (need_comma) {
                        if (vertical) {
                            print(',');
                            nl();
                        } else {
                            print(", ");
                        }
                    }
                    need_comma = true;
                    Declaration const *param = it;
                    print(param);
                }

                print(')');
                if (vertical)
                    --m_indent;
            }
            if (Stmt const *body = fdecl->get_body()) {
                nl();
                print(body);
            } else {
                print(';');
            }
        }
        break;
    case Declaration::DK_FIELD:
        {
            Declaration_field const *fdecl = cast<Declaration_field>(decl);
            
            Type_name const *tn = fdecl->get_type_name();
            push_color(C_TYPE);
            print(tn);
            pop_color();
            print(' ');

            bool need_comma = false;
            for (Declaration_field::const_iterator it(fdecl->begin()), end(fdecl->end());
                 it != end;
                 ++it)
            {
                if (need_comma) {
                    print(", ");
                }
                need_comma = true;

                Field_declarator const *field = it;

                Name const *name = field->get_name();
                push_color(C_ENTITY);
                print(name);
                pop_color();

                Array_specifiers const &as = field->get_array_specifiers();
                for (Array_specifiers::const_iterator it(as.begin()), end(as.end());
                     it != end;
                     ++it)
                {
                    Array_specifier const *spec = it;
                    print(spec);
                }
            }
        }
        break;
    case Declaration::DK_STRUCT:
        {
            Declaration_struct const *sdecl = cast<Declaration_struct>(decl);

            keyword("struct");
            print(' ');
            if (Name const *name = sdecl->get_name()) {
                print(name);
                print(' ');
            }
            print("{");
            ++m_indent;
            for (Declaration_struct::const_iterator it(sdecl->begin()), end(sdecl->end());
                 it != end;
                 ++it)
            {
                nl();

                Declaration const *decl = it;
                print(decl);
                print(';');
            }
            --m_indent;
            nl();
            print("}");
            if (!embedded)
                print(';');
        }
        break;
    case Declaration::DK_INTERFACE:
        {
            Declaration_interface const *idecl = cast<Declaration_interface>(decl);

            Type_qualifier const &tq = idecl->get_qualifier();
            print(&tq);

            Name const *name = idecl->get_name();
            push_color(C_ENTITY);
            print(name);
            pop_color();

            print(" {");
            ++m_indent;

            for (Declaration_interface::const_iterator it(idecl->begin()), end(idecl->end());
                 it != end;
                 ++it)
            {
                nl();
                Declaration const *field = it;
                print(field);
                print(';');
            }

            --m_indent;
            nl();
            print('}');
            if (!embedded)
                print(';');
        }
        break;
    case Declaration::DK_QUALIFIER:
        {
            Declaration_qualified const *qdecl = cast<Declaration_qualified>(decl);

            Type_qualifier const *tq = &qdecl->get_qualifier();
            print(tq);

            bool need_comma = false;
            for (Declaration_qualified::const_iterator it(qdecl->begin()), end(qdecl->end());
                 it != end;
                 ++it)
            {
                if (need_comma) {
                    print(", ");
                }
                need_comma = true;
                Instance_name const *iname = it;
                print(iname);
            }
            if (!embedded)
                print(';');
        }
        break;
    }
}

// Print an attribute.
void Printer::print_attr(
    HLSL_attribute const *attr)
{
    if (attr == NULL)
        return;

    print("[ ");

    switch (attr->get_kind()) {
    case HLSL_attribute::ATTR_FASTOPT:
        keyword("fastopt");
        break;
    case HLSL_attribute::ATTR_UNROLL:
        {
            keyword("unroll");

            HLSL_attribute_unroll const *unroll_attr =
                static_cast<HLSL_attribute_unroll const *>(attr);
            if (unroll_attr->get_unroll_count() != NULL) {
                print('(');
                print_value(unroll_attr->get_unroll_count());
                print(')');
            }
            break;
        }
    case HLSL_attribute::ATTR_LOOP:
        keyword("loop");
        break;
    case HLSL_attribute::ATTR_ALLOW_UAV_CONDITION:
        keyword("allow_uav_condition");
        break;
    case HLSL_attribute::ATTR_BRANCH:
        keyword("branch");
        break;
    case HLSL_attribute::ATTR_FLATTEN:
        keyword("flatten");
        break;
    case HLSL_attribute::ATTR_FORECASE:
        keyword("forecase");
        break;
    case HLSL_attribute::ATTR_CALL:
        keyword("call");
        break;
    }
    print(" ] ");
}

// Print declaration.
void Printer::print(Declaration const *decl)
{
    print_decl(decl, /*embedded=*/false);
}

// Print a type name.
void Printer::print(Type_name const *tn)
{
    Type_qualifier const &tq = tn->get_qualifier();
    print(&tq);

    if (Name const *name = tn->get_name()) {
        print(name);
    } else if (Declaration const *decl = tn->get_struct_decl()) {
        print_decl(decl, /*embedded=*/true);
    }
}

// Print a type qualifier.
void Printer::print(Type_qualifier const *tq)
{
    unsigned pq = tq->get_parameter_qualifier();
    if ((pq & PQ_INOUT) == PQ_INOUT) {
        keyword("inout");
        print(' ');
    } else if (pq & PQ_IN) {
        keyword("in");
        print(' ');
    } else if (pq & PQ_OUT) {
        keyword("out");
        print(' ');
    }

    unsigned sq = tq->get_storage_qualifiers();
    if (sq & SQ_EXTERN) {
        keyword("extern");
        print(' ');
    }
    if (sq & SQ_PRECISE) {
        keyword("precise");
        print(' ');
    }
    if (sq & SQ_SHARED) {
        keyword("shared");
        print(' ');
    }
    if (sq & SQ_UNIFORM) {
        keyword("uniform");
        print(' ');
    }
    if (sq & SQ_GROUPSHARED) {
        keyword("groupshared");
        print(' ');
    }
    if (sq & SQ_STATIC) {
        keyword("static");
        print(' ');
    }
    if (sq & SQ_UNIFORM) {
        keyword("uniform");
        print(' ');
    }
    if (sq & SQ_VOLATILE) {
        keyword("volatile");
        print(' ');
    }

    unsigned tm = tq->get_type_modifiers();
    if (tm & TM_CONST) {
        keyword("const");
        print(' ');
    }
    if (tm & TM_ROW_MAJOR) {
        keyword("row_major");
        print(' ');
    }
    if (tm & TM_COLUMN_MAJOR) {
        keyword("column_major");
        print(' ');
    }

    switch (tq->get_interpolation_qualifier()) {
    case IQ_NONE:
        break;
    case IQ_LINEAR:
        keyword("linear");
        print(' ');
        break;
    case IQ_CENTROID:
        keyword("centroid");
        print(' ');
        break;
    case IQ_NOINTERPOLATION:
        keyword("nointerpolation");
        print(' ');
        break;
    case IQ_NOPERSPECTIVE:
        keyword("noperspective");
        print(' ');
        break;
    case IQ_SAMPLE:
        keyword("sample");
        print(' ');
        break;
    }

    if (tq->is_invariant()) {
        keyword("invariant");
        print(' ');
    }

    if (tq->is_precise()) {
        keyword("precise");
        print(' ');
    }
}

// Print an init declarator.
void Printer::print(Init_declarator const *init)
{
    Name const *name = init->get_name();
    push_color(C_ENTITY);
    print(name);
    pop_color();

    Array_specifiers const &arr_sp = init->get_array_specifiers();
    for (Array_specifiers::const_iterator it(arr_sp.begin()), end(arr_sp.end()); it != end; ++it) {
        Array_specifier const *spec = it;
        print(spec);
    }

    if (Expr const *expr = init->get_initializer()) {
        print(" = ");
        print(expr, get_priority(Expr::OK_ASSIGN));
    }
}

// Print a name.
void Printer::print(Name const *name)
{
    Symbol *sym = name->get_symbol();
    print(sym);
}

// Print an array specifier.
void Printer::print(Array_specifier const *spec)
{
    print('[');
    if (Expr const *size = spec->get_size()) {
        print(size);
    }
    print(']');
}

// Print an instance name.
void Printer::print(Instance_name const *name)
{
    push_color(C_ENTITY);
    print(name->get_name());
    pop_color();
}

// Print compilation unit.
void Printer::print(ICompilation_unit const *iunit)
{
    Compilation_unit const *unit = impl_cast<Compilation_unit>(iunit);

    Store<Compilation_unit const *> unit_store(m_curr_unit, unit);

    int last_kind = -1;
    for (Compilation_unit::const_iterator it(unit->decl_begin()), end(unit->decl_end());
         it != end;
         ++it)
    {
        Declaration const *decl = it;

        Declaration::Kind kind = decl->get_kind();

        bool need_nl = false;
        if (kind == Declaration::DK_FUNCTION) {
            Declaration_function const *fdecl = cast<Declaration_function>(decl);
            need_nl = !fdecl->is_prototype();
        } else if (kind == Declaration::DK_VARIABLE) {
            Declaration_variable const *vdecl = cast<Declaration_variable>(decl);
            need_nl = vdecl->get_type_name()->get_struct_decl() != NULL;
        } else if (kind == Declaration::DK_STRUCT) {
            need_nl = true;
        }

        if (last_kind != kind || need_nl) {
            nl();
            last_kind = kind;
        }

        print(decl);

        nl();
    }
}

// Internal print a message
void Printer::print_message(Message const *message, Message::Severity sev)
{
    char const *fname = message->get_file();
    if (fname != NULL)
        print(fname);
    Location const &loc = message->get_location();
    unsigned line   = loc.get_line();
    unsigned column = loc.get_column();
    if (line > 0) {
        print("(");
        print(line);
        print(",");
        print(column);
        print(")");
    }
    print(": ");

    IOutput_stream_colored::Color c = IOutput_stream_colored::DEFAULT;
    switch (sev) {
    case Message::MS_ERROR:
        c = IOutput_stream_colored::RED;
        break;
    case Message::MS_WARNING:
        c = IOutput_stream_colored::YELLOW;
        break;
    case Message::MS_INFO:
        c = IOutput_stream_colored::DEFAULT;
        break;
    }

    if (m_color_output)
        m_c_ostr->set_color(c);
    switch (message->get_severity()) {
    case Message::MS_ERROR:
        print("Error E");
        printf("%03i: ", message->get_code());
        break;
    case Message::MS_WARNING:
        print("Warning W");
        printf("%03i: ", message->get_code());
        break;
    case Message::MS_INFO:
        print("Note: ");
        break;
    }
    if (m_color_output)
        m_c_ostr->reset_color();

    print(message->get_string());
    print("\n");
}

// Print message.
void Printer::print(Message const *message, bool include_notes)
{
    Message::Severity sev = message->get_severity();

    print_message(message, sev);

    if (include_notes) {
        for (size_t i = 0, n = message->get_note_count(); i < n; ++i) {
            Message const *note = message->get_note(i);
            print_message(note, sev);
        }
    }
}

// Flush output.
void Printer::flush()
{
    m_ostr->flush();
}

// Enable color output.
void Printer::enable_color(bool enable)
{
    if (enable) {
        // check if the given output stream supports color
        if (IOutput_stream_colored *cstr = m_ostr->get_interface<IOutput_stream_colored>()) {
            // yes
            m_c_ostr = cstr;
        } else {
            // no cannot be enabled
            enable = false;
        }
    }
    m_color_output = enable;
}

// Enable location printing using #line directives.
void Printer::enable_locations(bool enable)
{
    m_enable_loc = enable;
}

}  // hlsl
}  // mdl
}  // mi
