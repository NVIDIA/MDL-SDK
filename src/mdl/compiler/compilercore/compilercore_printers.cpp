/******************************************************************************
 * Copyright (c) 2011-2025, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/base/interface_implement.h>
#include <mi/base/iallocator.h>
#include <mi/base/handle.h>

#include <mi/mdl/mdl_types.h>
#include <mi/mdl/mdl_values.h>
#include <mi/mdl/mdl_names.h>
#include <mi/mdl/mdl_expressions.h>
#include <mi/mdl/mdl_statements.h>
#include <mi/mdl/mdl_declarations.h>
#include <mi/mdl/mdl_modules.h>
#include <mi/mdl/mdl_messages.h>
#include <mi/mdl/mdl_symbols.h>
#include <mi/mdl/mdl_printers.h>
#include <mi/mdl/mdl_streams.h>
#include <mi/mdl/mdl_generated_dag.h>

#include <string>

#include <iostream>
#include <cstdio>
#include <cstdarg>
#include <cassert>
#include <cmath>
#include <inttypes.h>

#include "compilercore_cc_conf.h"
#include "compilercore_mangle.h"
#include "compilercore_printers.h"
#include "compilercore_streams.h"
#include "compilercore_tools.h"
#include "compilercore_wchar_support.h"

namespace mi {
namespace mdl {

inline bool isfinite(double v)
{
    return -HUGE_VAL < v && v < HUGE_VAL;
}

/// Captured output stream.
class Captured_color_stream : public Allocator_interface_implement<IOutput_stream_colored>
{
    typedef Allocator_interface_implement<IOutput_stream_colored> Base;

    enum Kind { EK_COLOR, EK_RESET, EK_STRING };

    struct Entry {
        Entry(
            Kind kind)
        : m_kind(kind), m_next(NULL)
        {
        }

        Kind const m_kind;
        Entry *m_next;
    };

    struct Entry_color : public Entry {
        Entry_color(Color color, bool bold, bool background)
        : Entry(EK_COLOR), m_color(color), m_bold(bold), m_background(background)
        {
        }

        Color const m_color;
        bool const m_bold;
        bool const m_background;
    };

    struct Entry_string : public Entry {
        Entry_string(IAllocator *alloc)
        : Entry(EK_STRING)
        , m_string(alloc)
        {
        }

        ~Entry_string() {}

        string m_string;
    };

public:
    /// Constructor.
    Captured_color_stream(IAllocator *alloc)
    : Base(alloc)
    , m_builder(alloc)
    , m_first(NULL)
    , m_last(NULL)
    {
    }

    /// Returns true if this stream supports color.
    bool has_color() const MDL_FINAL { return true; }

    /// Set the color.
    /// \param color      The color.
    /// \param bold       If true, set bold.
    /// \param background if true, set background color, else foreground color.
    void set_color(Color color, bool bold, bool background) MDL_FINAL
    {
        Entry_color *c = m_builder.create<Entry_color>(color, bold, background);
        add_entry(c);
    }

    /// Reset the color to the default
    void reset_color() MDL_FINAL
    {
        add_entry(m_builder.create<Entry>(EK_RESET));
    }

    /// Write a char to the stream.
    void write_char(char c) MDL_FINAL
    {
        string &s = get_string_buf();
        s.append(c);
    }

    /// Write a string to the stream.
    void write(char const *cstring) MDL_FINAL
    {
        string &s = get_string_buf();
        s.append(cstring);
    }

    /// Flush stream.
    void flush() MDL_FINAL { }

    /// Remove the last character from output stream if possible.
    ///
    /// \param c  remove this character from the output stream
    ///
    /// \return true if c was the last character in the stream and it was successfully removed,
    /// false otherwise
    bool unput(char c) MDL_FINAL
    {
        string &s = get_string_buf();
        size_t l = s.size();
        if (l > 0 && s[l - 1] == c) {
            s.erase(s.begin() + l - 1);
            return true;
        }
        return false;
    }

    /// Replay the captured input on another output stream.
    void replay(IOutput_stream *out) const
    {
        // check if the given output stream supports color
        mi::base::Handle<IOutput_stream_colored> cout(out->get_interface<IOutput_stream_colored>());

        if (cout.is_valid_interface()) {
            // has color
            for (Entry const *e = m_first; e != NULL; e = e->m_next) {
                switch (e->m_kind) {
                case EK_COLOR:
                    {
                        Entry_color const *c = static_cast<Entry_color const *>(e);

                        cout->set_color(c->m_color, c->m_bold, c->m_background);
                    }
                    break;
                case EK_RESET:
                    cout->reset_color();
                    break;
                case EK_STRING:
                    {
                        Entry_string const *s = static_cast<Entry_string const *>(e);
                        print_utf8(cout.get(), s->m_string.c_str());
                    }
                    break;
                }
            }
        } else {
            // no color
            for (Entry const *e = m_first; e != NULL; e = e->m_next) {
                if (e->m_kind == EK_STRING) {
                    Entry_string const *s = static_cast<Entry_string const *>(e);
                    print_utf8(out, s->m_string.c_str());
                }
            }
        }
    }

    /// Prints an escaped UTF8 string to a output buffer.
    void print_utf8(IOutput_stream *out, char const *utf8_string) const
    {
        for (char const *p = utf8_string; *p;) {
            unsigned unicode_char;
            p = utf8_to_unicode_char(p, unicode_char);

            switch (unicode_char) {
            case '\a':  printf(out, "\\a");   break;
            case '\b':  printf(out, "\\b");   break;
            case '\f':  printf(out, "\\f");   break;
            case '\n':  printf(out, "\\n");   break;
            case '\r':  printf(out, "\\r");   break;
            case '\t':  printf(out, "\\t");   break;
            case '\v':  printf(out, "\\v");   break;
            case '\\':  printf(out, "\\\\");  break;
            case '\'':  printf(out, "\\\'");  break;
            case '"':   printf(out, "\\\"");  break;
            default:

                if (unicode_char <= 0x7F) {
                    printf(out, "%c", char(unicode_char));
                } else if (unicode_char <= 0xFFFF) {
                    printf(out, "\\u%04x", unsigned(unicode_char));
                } else {
                    printf(out, "\\U%08x", unsigned(unicode_char));
                }
                break;
            }
        }
    }

    // Format print.
    void printf(IOutput_stream *out, char const *format, ...) const
    {
        char buffer[1024];
        va_list ap;
        va_start(ap, format);
        vsnprintf(buffer, sizeof(buffer), format, ap);
        buffer[sizeof(buffer) - 1] = '\0';
        va_end(ap);
        out->write(buffer);
    }

private:
    /// Ann an entry.
    void add_entry(Entry *e)
    {
        if (m_last != NULL) {
            m_last->m_next = e;
        }
        e->m_next = NULL;
        if (m_first == NULL) {
            m_first = e;
        }
        m_last = e;
    }

    /// Get the string buffer to append.
    string &get_string_buf()
    {
        Entry_string *s = NULL;

        if (m_last != NULL && m_last->m_kind == EK_STRING) {
            s = static_cast<Entry_string *>(m_last);
        } else {
            s = m_builder.create<Entry_string>(m_builder.get_allocator());
            add_entry(s);
        }
        return s->m_string;
    }

private:
    /// Destructor.
    ~Captured_color_stream() MDL_FINAL
    {
        for (Entry const *n, *e = m_first; e != NULL; e = n) {
            n = e->m_next;
            if (e->m_kind == EK_STRING) {
                // need destructor call here
                Entry_string const *s = static_cast<Entry_string const *>(e);
                s->~Entry_string();
            }
            m_builder.destroy(e);
        }
    }

private:
    Allocator_builder m_builder;
    Entry *m_first;
    Entry *m_last;
};

// Print a type.
template<typename Stream, typename Redirect, bool allow_extra_modifiers>
void Type_printer<Stream, Redirect, allow_extra_modifiers>::print_type(
    IType const   *type,
    ISymbol const *name)
{
restart:
    char const *tn = NULL;

    IType::Kind tk = type->get_kind();

    switch (tk) {
    case IType::TK_ERROR:
        this->push_color(ISyntax_coloring::C_ERROR);
        this->write("<ERROR>");
        this->pop_color();
        break;
    case IType::TK_BOOL:             tn = "bool"; break;
    case IType::TK_INT:              tn = "int"; break;
    case IType::TK_FLOAT:            tn = "float"; break;
    case IType::TK_DOUBLE:           tn = "double"; break;
    case IType::TK_STRING:           tn = "string"; break;
    case IType::TK_COLOR:            tn = "color"; break;
    case IType::TK_LIGHT_PROFILE:    tn = "light_profile"; break;
    case IType::TK_BSDF_MEASUREMENT: tn = "bsdf_measurement"; break;
    case IType::TK_VOID:             tn = "void"; break;
    case IType::TK_AUTO:             tn = "auto"; break;
    case IType::TK_ENUM:             tn = cast<IType_enum>(type)->get_symbol()->get_name(); break;
    case IType::TK_ALIAS:
        {
            IType_alias const *a_type = cast<IType_alias>(type);
            ISymbol const *sym = a_type->get_symbol();
            if (sym == NULL) {
                // this alias type has no name, deduce it
                IType::Modifiers mod = a_type->get_type_modifiers();
                if (allow_extra_modifiers && (mod & IType::MK_CONST)) {
                    if (m_show_extra_modifiers) {
                        this->push_color(ISyntax_coloring::C_COMMENT);
                        this->write("/* const */");
                        this->pop_color();
                        this->write(' ');
                    }
                } else if (mod & IType::MK_VARYING) {
                    this->write("varying ");
                } else if (mod & IType::MK_UNIFORM) {
                    this->write("uniform ");
                }
                type = a_type->get_aliased_type();
                goto restart;
            }
            tn = sym->get_name();
            break;
        }
    case IType::TK_BSDF:      tn = "bsdf";      break;
    case IType::TK_HAIR_BSDF: tn = "hair_bsdf"; break;
    case IType::TK_EDF:       tn = "edf";       break;
    case IType::TK_VDF:       tn = "vdf";       break;
    case IType::TK_STRUCT:    tn = cast<IType_struct>(type)->get_symbol()->get_name(); break;

    case IType::TK_VECTOR:
        {
            IType_vector const *v_type = cast<IType_vector>(type);
            IType const *e_type = v_type->get_element_type();

            this->redirect_print_type(e_type);
            this->write("01234"[v_type->get_size()]);
            break;
        }
    case IType::TK_MATRIX:
        {
            IType_matrix const *m_type = cast<IType_matrix>(type);
            IType_vector const *e_type = m_type->get_element_type();
            IType_atomic const *a_type = e_type->get_element_type();

            this->redirect_print_type(a_type);
            this->write("01234"[m_type->get_columns()]);
            this->write('x');
            this->write("01234"[e_type->get_size()]);
            break;
        }
    case IType::TK_ARRAY:
        {
            IType_array const *a_type = cast<IType_array>(type);
            IType const *e_type = a_type->get_element_type();

            this->redirect_print_type(e_type);
            this->write('[');
            if (a_type->is_immediate_sized()) {
                this->push_color(ISyntax_coloring::C_LITERAL);
                this->write(a_type->get_size());
                this->pop_color();
            } else {
                this->write(a_type->get_deferred_size()->get_size_symbol()->get_name());
            }
            this->write(']');
            break;
        }
    case IType::TK_FUNCTION:
        {
            // should not happen
            IType_function const *f_type   = cast<IType_function>(type);
            IType const          *ret_type = f_type->get_return_type();

            if (ret_type != NULL) {
                this->redirect_print_type(ret_type);
            } else {
                this->write("void");
            }

            this->write(' ');
            if (name != NULL) {
                this->write(name->get_name());
            } else {
                this->write("(*)");
            }
            this->write('(');

            for (size_t i = 0, n = f_type->get_parameter_count(); i < n; ++i) {
                if (i > 0) {
                    this->write(", ");
                }

                IType const *p_type;
                ISymbol const *sym;
                f_type->get_parameter(i, p_type, sym);
                this->redirect_print_type(p_type);
                this->write(' ');
                this->write(sym->get_name());
            }
            this->write(')');
            break;
        }
    case IType::TK_TEXTURE:
        {
            IType_texture const *t_type = cast<IType_texture>(type);

            char const *s = "";
            switch (t_type->get_shape()) {
            case IType_texture::TS_2D:        s ="texture_2d";         break;
            case IType_texture::TS_3D:        s = "texture_3d";        break;
            case IType_texture::TS_CUBE:      s = "texture_cube";      break;
            case IType_texture::TS_PTEX:      s = "texture_ptex";      break;
            case IType_texture::TS_BSDF_DATA: s = "texture_bsdf_data"; break;
            }
            this->write(s);
            break;
        }
    case IType::TK_PTR:
        {
            IType_pointer const *p_type = cast<IType_pointer>(type);
            IType const *e_type = p_type->get_element_type();

            this->redirect_print_type(e_type);
            this->write(' ');
            if (p_type->get_address_space() != 0) {
                this->write('<');
                this->push_color(ISyntax_coloring::C_LITERAL);
                this->write(long(p_type->get_address_space()));
                this->pop_color();
                this->write('>');
            }
            this->write('*');
            break;
        }
    case IType::TK_REF:
        {
            IType_ref const *r_type = cast<IType_ref>(type);
            IType const     *e_type = r_type->get_element_type();

            this->redirect_print_type(e_type);
            this->write(' ');
            if (r_type->get_address_space() != 0) {
                this->write('<');
                this->push_color(ISyntax_coloring::C_LITERAL);
                this->write(long(r_type->get_address_space()));
                this->pop_color();
                this->write('>');
            }
            this->write('*');
            break;
        }
    }
    if (tn != NULL) {
        this->write(tn);
    }
}

// instantiate for Printer
template class Type_printer<IOutput_stream, Printer, true>;

// instantiate for mangler
template class Type_printer<String_output_stream<Static_base>, void, false>;

// Constructor.
Printer::Printer(IAllocator *alloc, IOutput_stream *ostr)
: Base(alloc)
, m_type_printer(ostr, this, /*show_extra_modifiers=*/false)
, m_version(IMDL::MDL_LATEST_VERSION)
, m_indent(0)
, m_color_output(false)
, m_show_positions(false)
, m_show_mdl_versions(false)
, m_show_res_table(false)
, m_show_func_hashes(false)
, m_c_ostr()
, m_color_stack(Syntax_elements_stack::container_type(alloc))
, m_in_module_name(false)
{
    ::memset(m_string_quote, 0, sizeof(m_string_quote));

    m_string_quote[0] = '"';
    m_string_quote[1] = '\0';

    ::memset(m_priority_map, 0, sizeof(m_priority_map));

    int prio = 1;

    // LOW priority

    m_priority_map[IExpression::OK_SEQUENCE]                        = prio;
    ++prio;

    m_priority_map[IExpression::OK_ASSIGN]                          = prio;
    m_priority_map[IExpression::OK_MULTIPLY_ASSIGN]                 = prio;
    m_priority_map[IExpression::OK_DIVIDE_ASSIGN]                   = prio;
    m_priority_map[IExpression::OK_MODULO_ASSIGN]                   = prio;
    m_priority_map[IExpression::OK_PLUS_ASSIGN]                     = prio;
    m_priority_map[IExpression::OK_MINUS_ASSIGN]                    = prio;
    m_priority_map[IExpression::OK_BITWISE_AND_ASSIGN]              = prio;
    m_priority_map[IExpression::OK_BITWISE_OR_ASSIGN]               = prio;
    m_priority_map[IExpression::OK_BITWISE_XOR_ASSIGN]              = prio;
    m_priority_map[IExpression::OK_SHIFT_LEFT_ASSIGN]               = prio;
    m_priority_map[IExpression::OK_SHIFT_RIGHT_ASSIGN]              = prio;
    m_priority_map[IExpression::OK_UNSIGNED_SHIFT_RIGHT_ASSIGN]     = prio;
    ++prio;

    m_priority_map[IExpression::OK_TERNARY]                         = prio;
    ++prio;

    m_priority_map[IExpression::OK_LOGICAL_OR]                      = prio;
    ++prio;

    m_priority_map[IExpression::OK_LOGICAL_AND]                     = prio;
    ++prio;

    m_priority_map[IExpression::OK_BITWISE_OR]                      = prio;
    ++prio;

    m_priority_map[IExpression::OK_BITWISE_XOR]                     = prio;
    ++prio;

    m_priority_map[IExpression::OK_BITWISE_AND]                     = prio;
    ++prio;

    m_priority_map[IExpression::OK_EQUAL]                           = prio;
    m_priority_map[IExpression::OK_NOT_EQUAL]                       = prio;
    ++prio;

    m_priority_map[IExpression::OK_LESS]                            = prio;
    m_priority_map[IExpression::OK_LESS_OR_EQUAL]                   = prio;
    m_priority_map[IExpression::OK_GREATER]                         = prio;
    m_priority_map[IExpression::OK_GREATER_OR_EQUAL]                = prio;
    ++prio;

    m_priority_map[IExpression::OK_SHIFT_LEFT]                      = prio;
    m_priority_map[IExpression::OK_UNSIGNED_SHIFT_RIGHT]            = prio;
    m_priority_map[IExpression::OK_SHIFT_RIGHT]                     = prio;
    ++prio;

    m_priority_map[IExpression::OK_PLUS]                            = prio;
    m_priority_map[IExpression::OK_MINUS]                           = prio;
    ++prio;

    m_priority_map[IExpression::OK_MULTIPLY]                        = prio;
    m_priority_map[IExpression::OK_DIVIDE]                          = prio;
    m_priority_map[IExpression::OK_MODULO]                          = prio;
    ++prio;

    m_priority_map[IExpression::OK_PRE_INCREMENT]                   = prio;
    m_priority_map[IExpression::OK_PRE_DECREMENT]                   = prio;
    m_priority_map[IExpression::OK_POSITIVE]                        = prio;
    m_priority_map[IExpression::OK_NEGATIVE]                        = prio;
    m_priority_map[IExpression::OK_BITWISE_COMPLEMENT]              = prio;
    m_priority_map[IExpression::OK_LOGICAL_NOT]                     = prio;
    ++prio;

    m_priority_map[IExpression::OK_POST_INCREMENT]                  = prio;
    m_priority_map[IExpression::OK_POST_DECREMENT]                  = prio;
    m_priority_map[IExpression::OK_SELECT]                          = prio;
    m_priority_map[IExpression::OK_ARRAY_INDEX]                     = prio;
    m_priority_map[IExpression::OK_CALL]                            = prio;
    m_priority_map[IExpression::OK_CAST]                            = prio;
    ++prio;

    // HIGH priority
}

// Set the string used to quote string values, default is "\"".
void Printer::set_string_quote(char const *quote)
{
    if (quote == NULL) {
        quote = "\"";
    }
    ::strncpy(m_string_quote, quote, sizeof(m_string_quote));
    m_string_quote[sizeof(m_string_quote) - 1] = '\0';
}

// Hidden destructor.
Printer::~Printer()
{
}

// Format print.
void Printer::vprintf(char const *format, va_list ap)
{
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, ap);
    buffer[sizeof(buffer) - 1] = '\0';
    m_type_printer.write(buffer);
}

// Format print.
void Printer::printf(char const *format, ...)
{
    va_list ap;
    va_start(ap, format);
    vprintf(format, ap);
    va_end(ap);
}

// Prints a newline and do indentation.
void Printer::nl()
{
    print("\n");
    indent(m_indent);
}

// Set the given color.
void Printer::color(ISyntax_coloring::Syntax_elements code)
{
    if (m_color_output) {
        if (code == C_DEFAULT) {
            m_c_ostr->reset_color();
            return;
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
        ISyntax_coloring::Syntax_elements code = C_DEFAULT;
        if (!m_color_stack.empty()) {
            code = m_color_stack.top();
        }
        color(code);
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
    return m_priority_map[op];
}

// Indent output.
void Printer::indent(int depth)
{
    for (int i = 0; i < depth; ++i)
        print("    ");
}

// Print string.
void Printer::print(char const *string)
{
    if (string[0] != '\0') {
        m_type_printer.write(string);
    }
}

// Print an utf8 string with escapes.
void Printer::print_utf8(char const *utf8_string, bool escapes)
{
    if (escapes) {
        for (char const *p = utf8_string; *p;) {
            unsigned unicode_char;
            p = utf8_to_unicode_char(p, unicode_char);

            switch (unicode_char) {
            case '\a':  print("\\a");   break;
            case '\b':  print("\\b");   break;
            case '\f':  print("\\f");   break;
            case '\n':  print("\\n");   break;
            case '\r':  print("\\r");   break;
            case '\t':  print("\\t");   break;
            case '\v':  print("\\v");   break;
            case '\\':  print("\\\\");  break;
            case '\'':  print("\\\'");  break;
            case '"':   print("\\\"");  break;
            default:
#if 1
                if (unicode_char <= 0x7F) {
                    print(char(unicode_char));
                } else if (unicode_char <= 0xFFFF) {
                    printf("\\u%04x", unsigned(unicode_char));
                } else {
                    printf("\\U%06x", unsigned(unicode_char));
                }
#else
                printf("%Lc", unicode_char);
#endif
                break;
            }
        }
    } else {
        for (char const *p = utf8_string; *p;) {
            unsigned unicode_char;
            p = utf8_to_unicode_char(p, unicode_char);
#if 1
            if (unicode_char <= 0x7F) {
                print(char(unicode_char));
            } else if (unicode_char <= 0xFFFF) {
                printf("\\u%04x", unsigned(unicode_char));
            } else {
                printf("\\U%06x", unsigned(unicode_char));
            }
#else
            printf("%Lc", unicode_char);
#endif
        }
    }
}

// Print character.
void Printer::print(char c)
{
    m_type_printer.write(c);
}

// Print boolean.
void Printer::print(bool b)
{
    m_type_printer.write(b ? "true" : "false");
}

// Print unsigned.
void Printer::print(unsigned n)
{
    printf("%u", n);
}

// Print signed.
void Printer::print(int n)
{
    printf("%d", n);
}

// Print size_t.
void Printer::print(size_t n)
{
    printf("%" PRIuPTR, n);
}

// Print double.
void Printer::print(double d)
{
    printf("%.16g", d);
}

// Anonymous
namespace {
    bool is_non_ascii(unsigned char c) {
        return ((c & 0x80) != 0);
    }

    bool is_alphabetic(unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    }

    bool is_alphanumeric(unsigned char c) {
        return is_alphabetic(c) || (c >= '0' && c <= '9');
    }

    bool is_ident_start(unsigned char c) {
        return is_alphabetic(c);
    }

    bool is_ident_follow(unsigned char c) {
        return is_alphanumeric(c) || c == '_';
    }

    // Return true if the given string corresponds to the syntax of
    // a regular MDL identifier [a-zA-Z][a-zA-Z0-9_]*.
    bool is_identifier(char const* name)
    {
        char const* p = name;
        if (!is_ident_start(*p)) {
            // Also returns if name is the empty string.
            return false;
        }
        p++;
        while (*p) {
            unsigned char c = *p;
            if (is_non_ascii(c) || !is_ident_follow(c)) {
                return false;
            }
            p++;
        }
        return true;
    }

    // Return true if name is an identifier that must be quoted as a
    // Unicode identifier.
    bool must_quote(char const* name, bool in_scope) {
        if (in_scope) {
            return !MDL::valid_mdl_identifier(name);
        } else {
            return !is_identifier(name);
        }
    }

    // Return true if the given symbol is one of the special symbols
    // in import statements (., .. or *).
    bool is_special_import_symbol(ISymbol const* sym) {
        return sym->get_id() == ISymbol::SYM_DOT
            || sym->get_id() == ISymbol::SYM_DOTDOT
            || sym->get_id() == ISymbol::SYM_STAR;
    }
} // Anonymous


// Print a symbol.
void Printer::print(ISymbol const *sym)
{
    char const *name = sym->get_name();
    bool quote_needed = must_quote(name, m_in_module_name) && !is_special_import_symbol(sym);
    if (quote_needed)
        print('\'');
    print(name);
    if (quote_needed)
        print('\'');
}

// Print simple name.
void Printer::print(ISimple_name const *name)
{
    ISymbol const *sym = name->get_symbol();

    size_t id = sym->get_id();
    bool is_type_name = ISymbol::SYM_TYPE_FIRST <= id && id <= ISymbol::SYM_TYPE_LAST;

    if (is_type_name) {
        push_color(C_TYPE);
    }
    print(sym);
    if (is_type_name) {
        pop_color();
    }
}

// Print qualified name.
void Printer::print(IQualified_name const *name)
{
    size_t n = name->get_component_count();

    if (n == 0) {
        push_color(C_ERROR);
        print("<ERROR>");
        pop_color();
        return;
    }

    if (name->is_absolute()) {
        print("::");
    }
    {
        Store<bool> in_module_name(m_in_module_name, true);
        for (size_t i = 0; i < n - 1; ++i) {
            push_color(C_LITERAL);
            print(name->get_component(i));
            pop_color();
            print("::");
        }
    }
    print(name->get_component(n - 1));
}

// Print type name.
void Printer::print(IType_name const *name)
{
    print(name->get_qualifier());
    IQualified_name const *qualified_name = name->get_qualified_name();
    print(qualified_name);
    if (name->is_array()) {
        print('[');
        if (name->is_concrete_array()) {
            // the expression might be missing here, for incomplete arrays
            if (IExpression const *expr = name->get_array_size()) {
                print(expr);
            }
        } else {
            print('<');
            push_color(C_ENTITY);
            print(name->get_size_name());
            pop_color();
            print('>');
        }
        print(']');
    }
}

// Print type.
void Printer::print(IType const *type)
{
    push_color(C_TYPE);
    print_type(type);
    pop_color();
}

// Print a type with current color.
void Printer::print_type(IType const *type, ISymbol const *name)
{
    m_type_printer.print_type(type, name);
}

// Print a type prefix (i.e. only the package name).
void Printer::print_type_prefix(IType_enum const *e_type)
{
    ISymbol const *sym = e_type->get_symbol();
    char const    *s   = sym->get_name();

    char const *p = strrchr(s, ':');
    if (p == NULL || p == s + 1) {
        // '::' at beginning or no scope
        return;
    }
    MDL_ASSERT(p[-1] == ':');
    for (; s <= p; ++s) {
        print(*s);
    }
}

// Returns true if a variable declaration of kind T v(a); can be rewritten as T v = a;
bool Printer::can_rewite_constructor_init(IExpression const * init)
{
    // need semantic info to decide
    return false;
}

/// Ensures that the ascii representation of a float constant
/// has a '.' and add the given format character.
static char *to_float_constant(char *s, char fmt_char)
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
    }
    *end++ = fmt_char;
    *end++ = '\0';

    return s;
}

// Prints a resource value.
void Printer::print_resource(IValue_resource const *res)
{
    switch (res->get_kind()) {
    case IValue::VK_TEXTURE:
        {
            IValue_texture const *tex      = cast<IValue_texture>(res);
            IType_texture  const *tex_type = tex->get_type();

            print_type(tex_type);
            print('(');
            print(m_string_quote);
            print_utf8(tex->get_string_value(), /*escape=*/true);
            print(m_string_quote);
            if (tex->get_tag_value() != 0) {
                printf(" /* tag %d, version %u */", tex->get_tag_value(), tex->get_tag_version());
            }
            print(", ");
            char const *s = "gamma_default";
            switch (tex->get_gamma_mode()) {
            case IValue_texture::gamma_default:
                s = "gamma_default";
                break;
            case IValue_texture::gamma_linear:
                s = "gamma_linear";
                break;
            case IValue_texture::gamma_srgb:
                s = "gamma_srgb";
                break;
            }
            print("::tex::");
            print(s);

            if (is_tex_2d(tex_type) || is_tex_3d(tex_type)) {
                // 2D and 3D textures have a selector
                print(", ");
                print(m_string_quote);
                print_utf8(tex->get_selector(), /*escape=*/true);
                print(m_string_quote);
            }
            print(')');
            break;
        }
    case IValue::VK_LIGHT_PROFILE:
    case IValue::VK_BSDF_MEASUREMENT:
        {
            print_type(res->get_type());
            print('(');
            print(m_string_quote);
            print_utf8(res->get_string_value(), /*escape=*/true);
            print(m_string_quote);

            if (res->get_tag_value() != 0) {
                printf(" /* tag %d, version %u */", res->get_tag_value(), res->get_tag_version());
            }
            print(')');
            break;
        }
    default:
        MDL_ASSERT(!"unsupported ressource type");
        break;
    }
}

// Print value.
void Printer::print(IValue const *value)
{
    push_color(C_LITERAL);

    IValue::Kind kind = value->get_kind();
    switch (kind) {
    case IValue::VK_BAD:
        push_color(C_ERROR);
        print("<BAD>");
        pop_color();
        break;
    case IValue::VK_BOOL:
        {
            IValue_bool const *v = cast<IValue_bool>(value);

            print(v->get_value());
            break;
        }
    case IValue::VK_INT:
        {
            IValue_int const *v = cast<IValue_int>(value);

            print(v->get_value());
            break;
        }
    case IValue::VK_ENUM:
        {
            IValue_enum const *v      = cast<IValue_enum>(value);
            IType_enum const  *e_type = v->get_type();
            int               idx     = v->get_index();

            print_type_prefix(e_type);

            if (IType_enum::Value const *e_val = e_type->get_value(idx)) {
                print(e_val->get_symbol());
            } else {
                // should not happen
                MDL_ASSERT(!"could not find enum value name");
                print(e_type);
                print('(');
                print(idx);
                print(')');
            }
            break;
        }
    case IValue::VK_FLOAT:
        {
            IValue_float const *v = cast<IValue_float>(value);
            char buf[64];

            float f = v->get_value();
            if (isfinite(f)) {
                snprintf(buf, sizeof(buf) - 2, "%.10g", f);
                buf[sizeof(buf) - 2] = '\0';
                print(to_float_constant(buf, 'f'));
            } else if (f == +HUGE_VAL) {
                print("(+1.0f/0.0f)");
            } else if (f == -HUGE_VAL) {
                print("(-1.0f/0.0f)");
            } else {
                print("(0.0f/0.0f)");
            }
            break;
        }
    case IValue::VK_DOUBLE:
        {
            IValue_double const *v = cast<IValue_double>(value);
            char buf[64];

            double d = v->get_value();
            if (isfinite(d)) {
                snprintf(buf, sizeof(buf) - 2, "%.18g", d);
                buf[sizeof(buf) - 2] = '\0';
                print(to_float_constant(buf, 'd'));
            } else if (d == +HUGE_VAL) {
                print("(+1.0d/0.0d)");
            } else if (d == -HUGE_VAL) {
                print("(-1.0d/0.0d)");
            } else {
                print("(0.0d/0.0d)");
            }
            break;
        }
    case IValue::VK_STRING:
        {
            IValue_string const *v = cast<IValue_string>(value);

            print(m_string_quote);
            print_utf8(v->get_value(), /*escape=*/true);
            print(m_string_quote);
            break;
        }
    case IValue::VK_VECTOR:
        {
            IValue_vector const *v = cast<IValue_vector>(value);
            print_type(v->get_type());
            print('(');

            // check if all are the same
            bool all_same = true;
            IValue const *first = v->get_value(0);
            for (size_t i = 1, n = v->get_component_count(); i < n; ++i) {
                IValue const *e = v->get_value(i);
                if (first != e) {
                    all_same = false;
                    break;
                }
            }
            if (all_same) {
                // use simple conversion constructor
                print(first);
            } else {
                // use elemental constructor
                for (size_t i = 0, n = v->get_component_count(); i < n; ++i) {
                    if (i > 0) {
                        print(", ");
                    }
                    print(v->get_value(i));
                }
            }
            print(')');
            break;
        }
    case IValue::VK_MATRIX:
        {
            IValue_matrix const *v = cast<IValue_matrix>(value);
            print_type(v->get_type());
            print('(');

            // check for diag constructor.
            bool                is_diag    = true;
            IValue_vector const *first_col = cast<IValue_vector>(v->get_value(0));
            IValue const        *diag      = first_col->get_value(0);
            IType_vector const  *v_tp      = first_col->get_type();
            size_t              n_rows     = v_tp->get_size();

            for (size_t col = 0, n = v->get_component_count(); col < n; ++col) {
                IValue_vector const *v_col = cast<IValue_vector>(v->get_value(col));

                for (size_t row = 0; row < n_rows; ++row) {
                    IValue const *e = v_col->get_value(row);

                    if (col == row) {
                        if (e != diag) {
                            is_diag = false;
                            break;
                        }
                    } else {
                        if (!e->is_zero()) {
                            is_diag = false;
                            break;
                        }
                    }
                }
                if (!is_diag) {
                    break;
                }
            }

            if (is_diag) {
                // use diag constructor
                print(diag);
            } else {
                // use elemental constructor
                for (size_t i = 0, n = v->get_component_count(); i < n; ++i) {
                    if (i > 0) {
                        print(", ");
                    }
                    print(v->get_value(i));
                }
            }
            print(')');
            break;
        }
    case IValue::VK_ARRAY:
        {
            IValue_array const *v = cast<IValue_array>(value);
            IType_array const  *type = cast<IType_array>(v->get_type()->skip_type_alias());
            IType const        *e_type = type->get_element_type()->skip_type_alias();

            print_type(e_type);
            print("[](");
            for (size_t i = 0, n = v->get_component_count(); i < n; ++i) {
                if (i > 0) {
                    print(", ");
                }
                print(v->get_value(i));
            }
            print(')');
            break;
        }
    case IValue::VK_RGB_COLOR:
        {
            IValue_rgb_color const *color = cast<IValue_rgb_color>(value);
            print_type(color->get_type());
            print('(');
            for (size_t i = 0, n = color->get_component_count(); i < n; ++i) {
                if (i > 0) {
                    print(", ");
                }
                print(color->get_value(i));
            }
            print(')');
        }
        break;
    case IValue::VK_STRUCT:
        {
            IValue_struct const *v = cast<IValue_struct>(value);
            IType_struct const  *type = cast<IType_struct>(v->get_type()->skip_type_alias());

            print_type(type);
            print('(');

            size_t n = type->get_field_count();
            IType_struct::Predefined_id pid = type->get_predefined_id();
            if (pid == IType_struct::SID_MATERIAL_EMISSION) {
                if (m_version < IMDL::MDL_VERSION_1_1) {
                    // filter out intensity_mode
                    --n;
                }
            } else if (pid == IType_struct::SID_MATERIAL_VOLUME) {
                if (m_version < IMDL::MDL_VERSION_1_7) {
                    // filter out emission_intensity
                    --n;
                }
            }

            for (size_t i = 0; i < n; ++i) {
                if (i > 0) {
                    print(", ");
                }

                IType_struct::Field const *field     = type->get_field(i);
                ISymbol const             *field_sym = field->get_symbol();

                IValue const *field_value = v->get_field(field_sym);
                print(field_sym);
                print(": ");
                print(field_value);
            }
            print(')');
            break;
        }
    case IValue::VK_INVALID_REF:
        print_type(value->get_type());
        print("()");
        break;
    case IValue::VK_TEXTURE:
    case IValue::VK_LIGHT_PROFILE:
    case IValue::VK_BSDF_MEASUREMENT:
        print_resource(cast<IValue_resource>(value));
        break;
    }
    pop_color();
}

// Print expression.
void Printer::print(IExpression const *expr)
{
    if (expr->in_parenthesis()) {
        print('(');
    }
    print(expr, /*priority=*/0);
    if (expr->in_parenthesis()) {
        print(')');
    }
}

// Print expression.
void Printer::print(IExpression const *expr, int priority)
{
    switch (expr->get_kind()) {
    case IExpression::EK_INVALID:
        push_color(C_ERROR);
        print("<ERROR>");
        pop_color();
        break;
    case IExpression::EK_LITERAL:
        {
            IExpression_literal const *lit = cast<IExpression_literal>(expr);
            print(lit->get_value());
            break;
        }
    case IExpression::EK_REFERENCE:
        {
            IExpression_reference const *ref = cast<IExpression_reference>(expr);
            IType_name const            *tn  = ref->get_name();

            print(tn);
            break;
        }
    case IExpression::EK_UNARY:
        {
            IExpression_unary const     *u = cast<IExpression_unary>(expr);
            IExpression_unary::Operator op = u->get_operator();
            int                         op_priority = get_priority(op);
            IExpression const           *arg = u->get_argument();

            if (op_priority < priority) {
                print('(');
            }

            char const *prefix = NULL, *postfix = NULL;
            switch (op) {
            case IExpression_unary::OK_BITWISE_COMPLEMENT:
                prefix = "~";
                break;
            case IExpression_unary::OK_LOGICAL_NOT:
                prefix = "!";
                break;
            case IExpression_unary::OK_POSITIVE:
                prefix = "+";
                break;
            case IExpression_unary::OK_NEGATIVE:
                prefix = "-";
                break;
            case IExpression_unary::OK_PRE_INCREMENT:
                prefix = "++";
                break;
            case IExpression_unary::OK_PRE_DECREMENT:
                prefix = "--";
                break;
            case IExpression_unary::OK_POST_INCREMENT:
                postfix = "++";
                break;
            case IExpression_unary::OK_POST_DECREMENT:
                postfix = "--";
                break;
            case IExpression_unary::OK_CAST:
                keyword("cast");
                prefix = "<";
                postfix = ")";
                break;
            }

            if (prefix != NULL) {
                print(prefix);

                if (op == IExpression_unary::OK_CAST) {
                    IType_name const *tn = u->get_type_name();

                    print(tn);
                    print(">(");
                }
            }

            int prio_ofs = 0;
            switch (op) {
            case IExpression_unary::OK_BITWISE_COMPLEMENT:
            case IExpression_unary::OK_LOGICAL_NOT:
            case IExpression_unary::OK_CAST:
                // right-associative
                break;
            case IExpression_unary::OK_POSITIVE:
            case IExpression_unary::OK_NEGATIVE:
            case IExpression_unary::OK_PRE_INCREMENT:
            case IExpression_unary::OK_PRE_DECREMENT:
                // right-associative
                if (IExpression_unary const *un_arg = as<IExpression_unary>(arg)) {
                    IExpression_unary::Operator un_op = un_arg->get_operator();
                    if (
                        (un_op == op) ||
                        (op == IExpression_unary::OK_POSITIVE &&
                         un_op == IExpression_unary::OK_PRE_INCREMENT) ||
                        (op == IExpression_unary::OK_NEGATIVE &&
                         un_op == IExpression_unary::OK_PRE_DECREMENT) ||
                        (op == IExpression_unary::OK_PRE_INCREMENT &&
                         un_op == IExpression_unary::OK_POSITIVE) ||
                        (op == IExpression_unary::OK_PRE_DECREMENT &&
                         un_op == IExpression_unary::OK_NEGATIVE))
                    {
                        // ensure parenthesis
                        prio_ofs = 1;
                    }
                }
                break;
            case IExpression_unary::OK_POST_INCREMENT:
            case IExpression_unary::OK_POST_DECREMENT:
                // left associative
                prio_ofs = 1;
            }

            print(arg, op_priority + prio_ofs);

            if (postfix != NULL) {
                print(postfix);
            }

            if (op_priority < priority) {
                print(')');
            }
            break;
        }
    case IExpression::EK_BINARY:
        {
            IExpression_binary const     *b = cast<IExpression_binary>(expr);
            IExpression_binary::Operator op = b->get_operator();
            int                          op_priority = get_priority(op);

            if (op_priority < priority) {
                print('(');
            }

            IExpression const *lhs = b->get_left_argument();
            print(lhs, op_priority);

            char const *infix = NULL, *postfix = NULL;
            switch (op) {
            case IExpression_binary::OK_SELECT:
                infix = "."; break;
            case IExpression_binary::OK_ARRAY_INDEX:
                infix = "["; postfix = "]"; break;
            case IExpression_binary::OK_MULTIPLY:
                infix = " * "; break;
            case IExpression_binary::OK_DIVIDE:
                infix = " / "; break;
            case IExpression_binary::OK_MODULO:
                infix = " % "; break;
            case IExpression_binary::OK_PLUS:
                infix = " + "; break;
            case IExpression_binary::OK_MINUS:
                infix = " - "; break;
            case IExpression_binary::OK_SHIFT_LEFT:
                infix = " << "; break;
            case IExpression_binary::OK_SHIFT_RIGHT:
                infix = " >> "; break;
            case IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT:
                infix = " >>> "; break;
            case IExpression_binary::OK_LESS:
                infix = " < "; break;
            case IExpression_binary::OK_LESS_OR_EQUAL:
                infix = " <= "; break;
            case IExpression_binary::OK_GREATER_OR_EQUAL:
                infix = " >= "; break;
            case IExpression_binary::OK_GREATER:
                infix = " > "; break;
            case IExpression_binary::OK_EQUAL:
                infix = " == "; break;
            case IExpression_binary::OK_NOT_EQUAL:
                infix = " != "; break;
            case IExpression_binary::OK_BITWISE_AND:
                infix = " & "; break;
            case IExpression_binary::OK_BITWISE_XOR:
                infix = " ^ "; break;
            case IExpression_binary::OK_BITWISE_OR:
                infix = " | "; break;
            case IExpression_binary::OK_LOGICAL_AND:
                infix = " && "; break;
            case IExpression_binary::OK_LOGICAL_OR:
                infix = " || "; break;
            case IExpression_binary::OK_ASSIGN:
                infix = " = "; break;
            case IExpression_binary::OK_MULTIPLY_ASSIGN:
                infix = " *= "; break;
            case IExpression_binary::OK_DIVIDE_ASSIGN:
                infix = " /= "; break;
            case IExpression_binary::OK_MODULO_ASSIGN:
                infix = " %= "; break;
            case IExpression_binary::OK_PLUS_ASSIGN:
                infix = " += "; break;
            case IExpression_binary::OK_MINUS_ASSIGN:
                infix = " -= "; break;
            case IExpression_binary::OK_SHIFT_LEFT_ASSIGN:
                infix = " >>= "; break;
            case IExpression_binary::OK_SHIFT_RIGHT_ASSIGN:
                infix = " >>= "; break;
            case IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT_ASSIGN:
                infix = " >>>= "; break;
            case IExpression_binary::OK_BITWISE_AND_ASSIGN:
                infix = " &= "; break;
            case IExpression_binary::OK_BITWISE_XOR_ASSIGN:
                infix = " ^= "; break;
            case IExpression_binary::OK_BITWISE_OR_ASSIGN:
                infix = " |= "; break;
            case IExpression_binary::OK_SEQUENCE:
                infix = ", "; break;
            }

            if (infix != NULL) {
                print(infix);
            } else {
                print(' ');
                push_color(C_ERROR);
                print("<ERROR>");
                pop_color();
                print(' ');
            }

            IExpression const *rhs = b->get_right_argument();
            // no need to put the rhs in parenthesis for the index operator
            print(rhs, op == IExpression_binary::OK_ARRAY_INDEX ? 0 : op_priority + 1);

            if (postfix != NULL) {
                print(postfix);
            }

            if (op_priority < priority) {
                print(')');
            }
            break;
        }
    case IExpression::EK_CONDITIONAL:
        {
            IExpression_conditional const *c = cast<IExpression_conditional>(expr);
            int                           op_priority = get_priority(IExpression::OK_TERNARY);

            if (op_priority < priority) {
                print('(');
            }
            
            print(c->get_condition(), op_priority);

            print(" ? ");

            print(c->get_true(), op_priority);

            print(" : ");

            print(c->get_false(), op_priority);

            if (op_priority < priority) {
                print(')');
            }
            break;
        }
    case IExpression::EK_CALL:
        {
            IExpression_call const *c = cast<IExpression_call>(expr);
            int                    op_priority = get_priority(IExpression::OK_CALL);

            if (op_priority < priority) {
                print('(');
            }

            print(c->get_reference(), op_priority);

            int arg_priority = get_priority(IExpression::OK_TERNARY);
            int count = c->get_argument_count();
            bool vertical = (2 < count) || (1 < count && is<IArgument_named>(c->get_argument(0)));
            if (vertical) {
                ++m_indent;
            }
            print('(');
            if (vertical) {
                nl();
            }
            for (int i = 0; i < count; ++i) {
                print(c->get_argument(i), arg_priority);
                if (i < count - 1) {
                    print(',');
                    if (vertical) {
                        nl();
                    } else {
                        print(' ');
                    }
                }
            }
            print(')');
            if (vertical) {
                --m_indent;
            }

            if (op_priority < priority) {
                print(')');
            }
            break;
        }

    case IExpression::EK_LET:
        {
            IExpression_let const *l = cast<IExpression_let>(expr);
            int op_priority = get_priority(IExpression::OK_SEQUENCE);
            size_t count = l->get_declaration_count();

            if (op_priority < priority) {
                print("( ");
            }

            keyword("let");
            if (1 < count) {
                print(" {");
            }
            ++m_indent;
            for (size_t i = 0; i < count; ++i) {
                IDeclaration const *decl = l->get_declaration(i);
                nl();
                print(decl);
            }
            --m_indent;
            nl();
            if (1 < count) {
                print("} ");
            }
            keyword("in");
            ++m_indent;
            nl();
            print(l->get_expression(), op_priority);

            if (op_priority < priority) {
                print(')');
            }
            --m_indent;
            break;
        }

    }
}

// Print statement.
void Printer::print(IStatement const *stmt)
{
    print(stmt, /*is_toplevel=*/true);
}

// Print statement.
void Printer::print(IStatement const *stmt, bool is_toplevel)
{
    if (is_toplevel) {
        nl();
        print_position(stmt);
    }
    switch (stmt->get_kind()) {
    case IStatement::SK_INVALID:
        push_color(C_ERROR);
        print("<ERROR>");
        pop_color();
        print(';');
        break;
    case IStatement::SK_COMPOUND:
        {
            IStatement_compound const *blk = cast<IStatement_compound>(stmt);
            print('{');
            ++m_indent;
            for (size_t i = 0, n = blk->get_statement_count(); i < n; ++i) {
                print(blk->get_statement(i));
            }
            --m_indent;
            nl();
            print('}');
            break;
        }
    case IStatement::SK_DECLARATION:
        {
            IStatement_declaration const *d = cast<IStatement_declaration>(stmt);
            print(d->get_declaration(), is_toplevel);
            break;
        }
    case IStatement::SK_EXPRESSION:
        {
            IStatement_expression const *e = cast<IStatement_expression>(stmt);
            if (IExpression const *expr = e->get_expression()) {
                print(expr);
            }
            print(';');
            break;
        }
    case IStatement::SK_IF:
        {
            IStatement_if const *i = cast<IStatement_if>(stmt);

            keyword("if");
            print(" (");
            print(i->get_condition());
            print(')');

            IStatement const *t_stmt = i->get_then_statement();
            bool is_block = is<IStatement_compound>(t_stmt);

            if (is_block) {
                print(' ');
                print(t_stmt, /*is_toplevel=*/false);
            } else {
                bool need_extra_block = false;
                if (is<IStatement_if>(t_stmt)) {
                    need_extra_block = true;
                    print(" {");
                }
                ++m_indent;
                print(t_stmt);
                --m_indent;
                if (need_extra_block) {
                    nl();
                    print('}');
                }
            }

            if (IStatement const *e_stmt = i->get_else_statement()) {
                if (is_block) {
                    print(' ');
                } else {
                    nl();
                }
                keyword("else");
                if (is<IStatement_compound>(e_stmt) || is<IStatement_if>(e_stmt)) {
                    print(' ');
                    print(e_stmt, /*is_toplevel=*/false);
                } else {
                    ++m_indent;
                    print(e_stmt);
                    --m_indent;
                }
            }
            break;
        }
    case IStatement::SK_CASE:
        {
            IStatement_case const *c = cast<IStatement_case>(stmt);

            if (IExpression const *label = c->get_label()) {
                keyword("case");
                print(' ');
                print(label);
                print(':');
            } else {
                keyword("default");
                print(':');
            }
            ++m_indent;
            for (size_t i = 0, n = c->get_statement_count(); i < n; ++i) {
                IStatement const *s = c->get_statement(i);
                print(s);
            }
            --m_indent;
            break;
        }
    case IStatement::SK_SWITCH:
        {
            IStatement_switch const *s = cast<IStatement_switch>(stmt);

            keyword("switch");
            print(" (");
            print(s->get_condition());
            print(") {");

            for (size_t i = 0, n = s->get_case_count(); i < n; ++i) {
                IStatement const *c = s->get_case(i);

                print(c);
            }
            nl();
            print('}');
            break;
        }
    case IStatement::SK_WHILE:
        {
            IStatement_while const *w = cast<IStatement_while>(stmt);

            keyword("while");
            print(" (");
            print(w->get_condition());
            print(')');

            IStatement const *body = w->get_body();

            if (is<IStatement_compound>(body)) {
                print(' ');
                print(body, /*is_toplevel=*/false);
            } else {
                ++m_indent;
                print(body);
                --m_indent;
            }
            break;
        }
    case IStatement::SK_DO_WHILE:
        {
            IStatement_do_while const *dw = cast<IStatement_do_while>(stmt);

            keyword("do");

            IStatement const *body = dw->get_body();
            if (is<IStatement_compound>(body)) {
                print(' ');
                print(body, /*is_toplevel=*/false);
                print(' ');
                keyword("while");
            } else {
                ++m_indent;
                print(body);
                --m_indent;
                nl();
                keyword("while");
            }
            print(" (");
            print(dw->get_condition());
            print(");");
            break;
        }
    case IStatement::SK_FOR:
        {
            IStatement_for const *f = cast<IStatement_for>(stmt);
            keyword("for");
            print(" (");

            if (IStatement const *init = f->get_init()) {
                // includes the ;
                print(init, /*is_toplevel=*/false);
            } else
                print(';');

            if (IExpression const *cond = f->get_condition()) {
                print(' ');
                print(cond);
            }
            print(';');

            if (IExpression const *upd = f->get_update()) {
                print(' ');
                print(upd);
            }

            print(')');

            IStatement const *body = f->get_body();
            if (is<IStatement_compound>(body)) {
                print(' ');
                print(body, /*is_toplevel=*/false);
            } else {
                ++m_indent;
                print(body);
                --m_indent;
            }
            break;
        }
    case IStatement::SK_BREAK:
        keyword("break");
        print(';');
        break;
    case IStatement::SK_CONTINUE:
        keyword("continue");
        print(';');
        break;
    case IStatement::SK_RETURN:
        {
            IStatement_return const *r = cast<IStatement_return>(stmt);
            keyword("return");
            if (IExpression const *expr = r->get_expression()) {
                print(' ');
                print(expr);
            }
            print(';');
            break;
        }
    }
}

/// Check if two simple names are syntactically equal.
static bool equal(ISimple_name const *a, ISimple_name const *b)
{
    return a->get_symbol() == b->get_symbol();
}

/// Check if two qualified names are syntactically equal.
static bool equal(IQualified_name const *a, IQualified_name const *b)
{
    int ca = a->get_component_count();
    int cb = b->get_component_count();
    if (ca != cb) {
        return false;
    }
    for (int i = 0; i < ca; ++i) {
        if (!equal(a->get_component(i), b->get_component(i))) {
            return false;
        }
    }
    return true;
}

/// Check if two type names are syntactically equal.
static bool equal(IType_name const *a, IType_name const *b)
{
    if (a->is_array() != b->is_array()) {
        return false;
    }
    if (a->is_concrete_array() != b->is_concrete_array()) {
        return false;
    }
    if (a->is_array()) {
        if (a->is_concrete_array()) {
            bool is_incomplete = a->is_incomplete_array();
            if (is_incomplete != b->is_incomplete_array()) {
                return false;
            }

            if (!is_incomplete) {
                IExpression_literal const *a_lit = as<IExpression_literal>(a->get_array_size());
                IExpression_literal const *b_lit = as<IExpression_literal>(b->get_array_size());

                if (a_lit != NULL && b_lit != NULL) {
                    if (a_lit->get_value() != b_lit->get_value()) {
                        return false;
                    }
                } else {
                    IExpression_reference const *a_ref =
                        as<IExpression_reference>(a->get_array_size());
                    IExpression_reference const *b_ref =
                        as<IExpression_reference>(b->get_array_size());

                    if (a_ref == NULL || b_ref == NULL) {
                        return false;
                    }
                    if (!equal(a_ref->get_name(), b_ref->get_name())) {
                        return false;
                    }
                }
            }
        } else {
            if (!equal(a->get_size_name(), b->get_size_name())) {
                return false;
            }
        }
    }
    IQualified_name const *qa = a->get_qualified_name();
    IQualified_name const *qb = b->get_qualified_name();
    return qa != NULL && qb != NULL && equal(qa, qb);
}

/// Check if the declarator uses a constructor style initialization.
static bool is_constructor_init(IType_name const *type_name, IExpression const *init_exp)
{
    if (IExpression_call const *call = as<IExpression_call>(init_exp)) {
        if (IExpression_reference const *ref = as<IExpression_reference>(call->get_reference())) {
            IType_name const *constructor_name = ref->get_name();
            return equal(type_name, constructor_name);
        }
    }
    return false;
}

// Print declaration.
void Printer::print(IDeclaration const *decl)
{
    print(decl, /*is_toplevel=*/true);
}

// Print declaration.
void Printer::print(IDeclaration const *decl, bool is_toplevel)
{
    if (is_toplevel) {
        print_position(decl);
    }
    if (decl->is_exported()) {
        keyword("export");
        print(' ');
    }
    switch (decl->get_kind()) {
    case IDeclaration::DK_INVALID:
        push_color(C_ERROR);
        print("<ERROR>");
        pop_color();
        break;
    case IDeclaration::DK_IMPORT:
        {
            IDeclaration_import const *d = cast<IDeclaration_import>(decl);
            if (d->get_module_name()) {
                keyword("using");
                print(' ');
                push_color(C_LITERAL);
                Store<bool> in_module_name(m_in_module_name, true);
                print(d->get_module_name());
                pop_color();
                print(' ');
            }
            keyword("import");
            push_color(C_ENTITY);
            int count = d->get_name_count();
            for (int i = 0; i < count; i++) {
                if (i > 0) {
                    print(',');
                }
                print(' ');
                print(d->get_name(i));
            }
            pop_color();
            print(';');
            break;
        }
    case IDeclaration::DK_ANNOTATION:
        {
            IDeclaration_annotation const *d = cast<IDeclaration_annotation>(decl);
            print_mdl_versions(d->get_definition());
            keyword("annotation");
            print(' ');
            push_color(C_ANNOTATION);
            print(d->get_name());
            pop_color();
            print('(');

            for (size_t i = 0, n = d->get_parameter_count(); i < n; ++i) {
                IParameter const *parameter = d->get_parameter(i);
                if (i > 0) {
                    print(", ");
                }
                print(parameter);
            }
            print(')');

            print_anno_block(d->get_annotations(), " ");

            print(';');
            break;
        }
    case IDeclaration::DK_STRUCT_CATEGORY:
        {
            IDeclaration_struct_category const *d = cast<IDeclaration_struct_category>(decl);
            print_mdl_versions(d->get_definition());
            keyword("struct_category");
            print(' ');
            push_color(C_ENTITY);
            print(d->get_name());
            pop_color();

            print_anno_block(d->get_annotations(), " ");

            print(';');
            break;
        }
    case IDeclaration::DK_CONSTANT:
        {
            IDeclaration_constant const *d = cast<IDeclaration_constant>(decl);
            typepart("const");
            print(' ');
            print(d->get_type_name());
            print(' ');

            for (size_t i = 0, n = d->get_constant_count(); i < n; ++i) {
                if (i > 0) {
                    print(", ");
                }

                ISimple_name const *cname = d->get_constant_name(i);
                print_mdl_versions(cname->get_definition(), /*insert=*/true);

                push_color(C_ENTITY);
                print(cname);
                pop_color();

                IExpression const *init = d->get_constant_exp(i);
                if (is_constructor_init(d->get_type_name(),init)) {
                    IExpression_call const *call = cast<IExpression_call>(init);
                    int arg_count = call->get_argument_count();
                    if (arg_count == 1) {
                        // rewrite T v(a) into T v = a;
                        print(" = ");
                        print(
                            call->get_argument(0),
                            get_priority(IExpression::OK_ASSIGN),
                            /*ignore_named=*/true);
                    } else if (arg_count > 0) {
                        bool vertical = 1 < arg_count;
                        print('(');
                        if (vertical) {
                            ++m_indent;
                            nl();
                        }
                        for (int arg_idx = 0; arg_idx < arg_count; ++arg_idx) {
                            print(call->get_argument(arg_idx));
                            if (arg_idx < arg_count - 1) {
                                print(',');
                                if (vertical) {
                                    nl();
                                } else {
                                    print(' ');
                                }
                            }
                        }
                        print(')');
                        if (vertical) {
                            --m_indent;
                        }
                    }
                } else {
                    print(" = ");
                    print(init, get_priority(IExpression::OK_ASSIGN));
                }

                print_anno_block(d->get_annotations(i), " ");
            }
            print(';');
            break;
        }
    case IDeclaration::DK_TYPE_ALIAS:
        {
            IDeclaration_type_alias const *d = cast<IDeclaration_type_alias>(decl);
            keyword("typedef");
            print(' ');
            print(d->get_type_name());
            print(' ');
            print(d->get_alias_name());
            print(';');
            break;
        }
    case IDeclaration::DK_TYPE_STRUCT:
        {
            IDeclaration_type_struct const *d = cast<IDeclaration_type_struct>(decl);
            if (d->is_declarative()) {
                keyword("declarative");
                print(' ');
            }
            typepart("struct");
            print(' ');
            print(d->get_name());
            IQualified_name const *cat_name = d->get_struct_category_name();
            if (cat_name) {
                print(' ');
                keyword("in");
                print(' ');
                print(cat_name);
            }
            print_anno_block(d->get_annotations(), " ");
            print(" {");

            ++m_indent;
            for (size_t i = 0, n = d->get_field_count(); i < n; ++i) {
                nl();

                IType_name const *tname = d->get_field_type_name(i);
                print(tname);
                print(' ');

                // Get the name of the field at index.
                ISimple_name const *fname = d->get_field_name(i);
                print(fname);

                // Get the initializer of the field at index.
                if (IExpression const *init = d->get_field_init(i)) {
                    print(" = ");
                    print(init, get_priority(IExpression::OK_ASSIGN));
                }

                // Get the annotations of the field at index.
                print_anno_block(d->get_annotations(i), " ");
                print(';');
            }
            --m_indent;
            nl();
            print("};");
            break;
        }
    case IDeclaration::DK_TYPE_ENUM:
        {
            IDeclaration_type_enum const *d = cast<IDeclaration_type_enum>(decl);
            typepart("enum");
            print(' ');
            if (d->is_enum_class()) {
                typepart("class");
                print(' ');
            }
            print(d->get_name());
            print_anno_block(d->get_annotations(), " ");
            print(" {");

            ++m_indent;
            for (size_t i = 0, n = d->get_value_count(); i < n; ++i) {
                if (i > 0) {
                    print(',');
                }

                nl();

                ISimple_name const *vname = d->get_value_name(i);
                push_color(C_LITERAL);
                print(vname);
                pop_color();

                if (IExpression const *init = d->get_value_init(i)) {
                    print(" = ");
                    print(init, get_priority(IExpression::OK_ASSIGN));
                }

                print_anno_block(d->get_annotations(i), " ");
            }
            --m_indent;
            nl();
            print("};");
            break;
        }
    case IDeclaration::DK_VARIABLE:
        {
            IDeclaration_variable const *d = cast<IDeclaration_variable>(decl);
            print(d->get_type_name());
            print(' ');
            for (size_t i = 0, n = d->get_variable_count(); i < n; ++i) {
                if (i > 0) {
                    print(", ");
                }

                ISimple_name const *cname = d->get_variable_name(i);
                push_color(C_ENTITY);
                print(cname);
                pop_color();

                if (IExpression const *init = d->get_variable_init(i)) {
                    if (is_constructor_init(d->get_type_name(), init)) {
                        IExpression_call const *call = cast<IExpression_call>(init);
                        int arg_count = call->get_argument_count();
                        if (arg_count == 1 && can_rewite_constructor_init(init)) {
                            // rewrite T v(a) into T v = a;
                            print(" = ");
                            print(
                                call->get_argument(0),
                                get_priority(IExpression::OK_ASSIGN),
                                /*ignore_named=*/true);
                        } else if (arg_count > 0) {
                            bool vertical = 3 < arg_count;
                            print('(');
                            if (vertical) {
                                ++m_indent;
                                nl();
                            }
                            for (int arg_idx = 0; arg_idx < arg_count; arg_idx++) {
                                print(call->get_argument(arg_idx));
                                if (arg_idx < arg_count - 1) {
                                    print(',');
                                    if (vertical) {
                                        nl();
                                    } else {
                                        print(' ');
                                    }
                                }
                            }
                            print(')');
                            if (vertical) {
                                --m_indent;
                            }
                        }
                    } else {
                        print(" = ");
                        print(init, get_priority(IExpression::OK_ASSIGN));
                    }
                }

                print_anno_block(d->get_annotations(i), " ");
            }
            print(';');
            break;
        }
    case IDeclaration::DK_FUNCTION:
        {
            IDeclaration_function const *d = cast<IDeclaration_function>(decl);
            print_mdl_versions(d->get_definition());
            if (d->is_declarative()) {
                keyword("declarative");
                print(' ');
            }
            print(d->get_return_type_name());
            print_anno_block(d->get_return_annotations(), " ");
            print(' ');
            push_color(C_ENTITY);
            print(d->get_name());
            pop_color();
            if (d->is_preset()) {
                print("(*)");
            } else {
                int count = d->get_parameter_count();
                bool vertical = 1 < count;
                if (count == 1) {
                    IParameter const *first = d->get_parameter(0);
                    if (first->get_annotations() != NULL) {
                        vertical = true;
                    }
                }
                print('(');
                if (vertical) {
                    ++m_indent;
                    nl();
                }
                for (int i = 0; i < count; ++i) {
                    IParameter const *parameter = d->get_parameter(i);
                    print(parameter);
                    if (i < count - 1) {
                        print(',');
                        if (vertical) {
                            nl();
                        } else {
                            print(' ');
                        }
                    }
                }
                print(')');
                switch (d->get_qualifier()) {
                case FQ_NONE:
                    break;
                case FQ_VARYING:
                    typepart(" varying");
                    break;
                case FQ_UNIFORM:
                    typepart(" uniform");
                    break;
                }
                if (vertical) {
                    --m_indent;
                }
            }
            print_anno_block(d->get_annotations(), "\n");
            if (IStatement const *body = d->get_body()) {
                if (IStatement_compound const *s = as<IStatement_compound>(body)) {
                    print(s);
                } else if (IStatement_expression const *e = as<IStatement_expression>(body)) {
                    nl();
                    print(" = ");
                    IExpression const *expr = e->get_expression();
                    if (is<IExpression_let>(expr)) {
                        ++m_indent;
                        nl();
                        print(expr);
                        --m_indent;
                    } else {
                        print(expr, get_priority(IExpression::OK_ASSIGN));
                    }
                    print(';');
                }
            } else {
                print(';');
            }
            break;
        }
    case IDeclaration::DK_MODULE:
        {
            IDeclaration_module const *d = cast<IDeclaration_module>(decl);
            keyword("module");
            if (IAnnotation_block const *block = d->get_annotations()) {
                print_anno_block(block, " ");
            }
            print(';');
            break;
        }
    case IDeclaration::DK_NAMESPACE_ALIAS:
        {
            IDeclaration_namespace_alias const *d = cast<IDeclaration_namespace_alias>(decl);
            keyword("using");
            print(' ');
            print(d->get_alias());
            print(" = ");
            print_namespace(d->get_namespace());
            print(';');
            break;
        }
    }
}

// Print qualifier.
void Printer::print(Qualifier qualifier)
{
    switch (qualifier) {
    case FQ_NONE: break;
    case FQ_VARYING: typepart("varying "); break;
    case FQ_UNIFORM: typepart("uniform "); break;
    }
}

// Print parameter.
void Printer::print(IParameter const *parameter)
{
    IType_name const *tname = parameter->get_type_name();
    print(tname);
    print(' ');

    ISimple_name const *pname = parameter->get_name();
    push_color(C_ENTITY);
    print(pname);
    pop_color();

    if (IExpression const *init = parameter->get_init_expr()) {
        print(" = ");
        print(init, get_priority(IExpression::OK_ASSIGN));
    }

    print_anno_block(parameter->get_annotations(), " ");
}

// Print annotation.
void Printer::print(IAnnotation const *anno)
{
    print(anno->get_name());
    print('(');
    for (size_t i = 0, n = anno->get_argument_count(); i < n; ++i) {
        if (i > 0) {
            print(", ");
        }

        IArgument const *arg = anno->get_argument(i);
        print(arg);
    }
    print(')');
}

// Print enable_if annotation.
void Printer::print(IAnnotation_enable_if const *anno)
{
    IExpression const *expr = anno->get_expression();
    if (expr == NULL) {
        print(static_cast<IAnnotation const *>(anno));
        return;
    }

    print(anno->get_name());
    print("(\"");

    // print the expression into a captured color stream
    mi::base::Handle<IOutput_stream_colored> safe_c(m_c_ostr);

    Allocator_builder builder(get_allocator());
    m_c_ostr = mi::base::make_handle(builder.create<Captured_color_stream>(get_allocator()));
    m_type_printer.m_stream = m_c_ostr.get();

    print(expr);

    m_type_printer.m_stream = m_type_printer.m_holder.get();
    m_c_ostr.swap(safe_c);

    Captured_color_stream const *s = static_cast<Captured_color_stream *>(safe_c.get());
    s->replay(m_type_printer.m_holder.get());

    print("\")");
}

// Print annotation block.
void Printer::print(IAnnotation_block const *blk)
{
    push_color(C_ANNOTATION);
    print("[[");
    ++m_indent;
    for (size_t i = 0, n = blk->get_annotation_count(); i < n; ++i) {
        if (i > 0) {
            print(',');
        }
        nl();

        IAnnotation const *anno = blk->get_annotation(i);

        if (anno->get_kind() == IAnnotation::AK_ENABLE_IF) {
            print(cast<IAnnotation_enable_if>(anno));
        } else {
            print(anno);
        }
    }
    --m_indent;
    nl();
    print("]]");
    pop_color();
}

// Print an argument.
void Printer::print(IArgument const *arg, int priority, bool ignore_named)
{
    switch (arg->get_kind()) {
    case IArgument::AK_POSITIONAL:
        break;
    case IArgument::AK_NAMED:
        if (!ignore_named) {
            IArgument_named const *n = cast<IArgument_named>(arg);
            ISimple_name const *name = n->get_parameter_name();
            print(name);
            print(": ");
        }
        break;
    }
    IExpression const *expr = arg->get_argument_expr();
    print(expr, priority);
}

// Print an argument.
void Printer::print(IArgument const *arg)
{
    print(arg, /*priority=*/0);
}

// Print module.
void Printer::print(IModule const *module)
{
    Store<unsigned> store(m_version, impl_cast<Module>(module)->get_version());

    int major, minor;
    module->get_version(major, minor);
    keyword("mdl");
    print(' ');
    push_color(C_LITERAL);
    printf("%d.%d", major, minor);
    pop_color();
    print(';');

    IDeclaration::Kind last_kind = IDeclaration::Kind(-1);
    for (size_t i = 0, n = module->get_declaration_count(); i < n; ++i) {
        IDeclaration const *decl = module->get_declaration(i);
        IDeclaration::Kind kind  = decl->get_kind();

        if (last_kind != kind ||
            (last_kind != IDeclaration::DK_IMPORT
                && last_kind != IDeclaration::DK_NAMESPACE_ALIAS
                && last_kind != IDeclaration::DK_CONSTANT)) {
            nl();
        }
        nl();
        print(decl);

        last_kind = kind;
    }
    nl();

    if (m_show_res_table && module->get_referenced_resources_count() > 0) {
        nl();
        push_color(C_COMMENT);
        print("// Resource table:"); nl();

        for (size_t i = 0, n = module->get_referenced_resources_count(); i < n; ++i) {
            print("// ");

            IType_resource const *t = module->get_referenced_resource_type(i);
            switch (t->get_kind()) {
            case IType_resource::TK_TEXTURE:
                {
                    IType_texture const *tex_tp = cast<IType_texture>(t);

                    switch (tex_tp->get_shape()) {
                    case IType_texture::TS_2D:
                        typepart("texture_2d"); print("       ");
                        break;
                    case IType_texture::TS_3D:
                        typepart("texture_3d"); print("       ");
                        break;
                    case IType_texture::TS_CUBE:
                        typepart("texture_cube"); print( "    ");
                        break;
                    case IType_texture::TS_PTEX:
                        typepart("texture_ptex"); print("     ");
                        break;
                    case IType_texture::TS_BSDF_DATA:
                        typepart("texture_bsdf_data"); print("       ");
                        break;
                    }
                }
                break;
            case IType_texture::TK_LIGHT_PROFILE:
                typepart("light_profile"); print("    ");
                break;
            case IType_texture::TK_BSDF_MEASUREMENT:
                typepart("bsdf_measurement"); print(' ');
                break;
            default:
                MDL_ASSERT(!"unsupported type kind");
                break;
            }

            if (module->get_referenced_resource_exists(i)) {
                print("VALID   ");
            } else {
                push_color(C_ERROR);
                print("INVALID");
                pop_color();
                print(' ');
            }

            push_color(C_LITERAL);
            print('"');
            print(module->get_referenced_resource_url(i));
            print('"');
            pop_color();

            print(' ');
            if (char const *filename = module->get_referenced_resource_file_name(i)) {
                push_color(C_LITERAL);
                print('"');
                print(filename);
                print('"');
                pop_color();
            } else {
                push_color(C_LITERAL);
                print("<UNKNOWN>");
                pop_color();
            }

            nl();
        }
        pop_color();
    }

    if (m_show_func_hashes && module->has_function_hashes()) {
        Module const           *m  = impl_cast<Module>(module);
        Definition_table const &dt = m->get_definition_table();

        class Definition_visitor : public IDefinition_visitor {
        public:
            /// Constructor.
            Definition_visitor(
                Printer       &printer,
                IModule const *mod)
            : m_printer(printer)
            , m_mod(*mod)
            {
            }

            /// Called for every visited definition.
            void visit(Definition const *def) const MDL_FINAL
            {
                if (def->get_kind()!= IDefinition::DK_FUNCTION) {
                    return;
                }

                if (IModule::Function_hash const *hash = m_mod.get_function_hash(def)) {
                    m_printer.print("// ");

                    for (size_t i = 0, n = dimension_of(hash->hash); i < n; ++i) {
                        m_printer.printf("%02x", hash->hash[i]);
                    }
                    m_printer.print(' ');
                    m_printer.print_type(def->get_type(), def->get_symbol());
                    m_printer.nl();
                }
            }

        private:
            Printer &m_printer;
            IModule const &m_mod;
        };

        Definition_visitor visitor(*this, module);

        nl();
        push_color(C_COMMENT);
        print("// Function hash table:");
        nl();

        dt.walk(&visitor);

        pop_color();
    }
}

// Internal print a message
void Printer::print_message(IMessage const *message, IMessage::Severity sev)
{
    bool has_prefix = false;
    char const *fname = message->get_file();
    if (fname != NULL && fname[0] != '\0') {
        print(fname);
        has_prefix = true;
    }
    Position const *pos = message->get_position();
    int line = pos->get_start_line();
    int column = pos->get_start_column();
    if (line > 0) {
        print('(');
        print(line);
        print(',');
        print(column);
        print(')');
        has_prefix = true;
    }
    if (has_prefix) {
        print(": ");
    }

    if (m_color_output) {
        IOutput_stream_colored::Color c = IOutput_stream_colored::DEFAULT;

        switch (sev) {
        case IMessage::MS_ERROR:
            c = IOutput_stream_colored::RED;
            break;
        case IMessage::MS_WARNING:
            c = IOutput_stream_colored::YELLOW;
            break;
        case IMessage::MS_INFO:
            c = IOutput_stream_colored::DEFAULT;
            break;
        }
        m_c_ostr->set_color(c);
    }

    switch (message->get_severity()) {
    case IMessage::MS_ERROR:
        printf("Error %c%03i: ", message->get_class(), message->get_code());
        break;
    case IMessage::MS_WARNING:
        printf("Warning %c%03i: ", message->get_class(), message->get_code());
        break;
    case IMessage::MS_INFO:
        print("Note: ");
        break;
    }
    if (m_color_output) {
        m_c_ostr->reset_color();
    }
    print(message->get_string());
    print("\n");
}

// Print message.
void Printer::print(IMessage const *message, bool include_notes)
{
    IMessage::Severity sev = message->get_severity();

    print_message(message, sev);

    if (include_notes) {
        for (size_t i = 0, n = message->get_note_count(); i < n; ++i) {
            IMessage const *note = message->get_note(i);
            print_message(note, sev);
        }
    }
}

// Print semantic version.
void Printer::print(ISemantic_version const *sem_ver)
{
    printf("%d.%d.%d", sem_ver->get_major(), sem_ver->get_minor(), sem_ver->get_patch());

    char const *prerelease = sem_ver->get_prerelease();
    if (prerelease != NULL && prerelease[0] != '\0') {
        printf("-%s", prerelease);
    }
}

// Print generated code.
void Printer::print(IGenerated_code const *code)
{
    mi::base::Handle<IPrinter_interface const> if_printer(
        code->get_interface<IPrinter_interface>());

    if (if_printer.is_valid_interface()) {
        // the code has its own printer, use it
        if_printer->print(this, code);
        return;
    }

    print("/*** Unsupported code ***/\n");
}

// Print material instance.
void Printer::print(mi::base::IInterface const *iface)
{
    mi::base::Handle<IPrinter_interface const> if_printer(
        iface->get_interface<IPrinter_interface>());

    if (if_printer.is_valid_interface()) {
        // the interface has its own printer, use it
        if_printer->print(this, iface);
    }
}

void Printer::print_position(IStatement const *stmt)
{
    if (m_show_positions && !is<IStatement_declaration>(stmt)) {
        Position const &pos = stmt->access_position();
        print("// ");
        print(pos.get_start_line());
        print('(');
        print(pos.get_start_column());
        print(") - ");
        print(pos.get_end_line());
        print('(');
        print(pos.get_end_column());
        print(')');
        nl();
    }
}

void Printer::print_position(IDeclaration const *decl)
{
    if (m_show_positions) {
        Position const &pos = decl->access_position();
        print("// ");
        print(pos.get_start_line());
        print('(');
        print(pos.get_start_column());
        print(") - ");
        print(pos.get_end_line());
        print('(');
        print(pos.get_end_column());
        print(')');
        nl();
    }
}

void Printer::print_mdl_versions(IDefinition const *idef, bool insert)
{
    if (m_show_mdl_versions && idef != NULL) {
        Definition const *def = impl_cast<Definition>(idef);
        unsigned flags = def->get_version_flags();

        unsigned since = mdl_since_version(flags);
        unsigned rem   = mdl_removed_version(flags);

        if (since != 0 || rem != 0xFFFFFFFF) {
            print(insert ? "/*" : "//");
            switch (since) {
            case IMDL::MDL_VERSION_1_0:
                break;
            case IMDL::MDL_VERSION_1_1:
                print(" Since MDL 1.1");
                break;
            case IMDL::MDL_VERSION_1_2:
                print(" Since MDL 1.2");
                break;
            case IMDL::MDL_VERSION_1_3:
                print(" Since MDL 1.3");
                break;
            case IMDL::MDL_VERSION_1_4:
                print(" Since MDL 1.4");
                break;
            case IMDL::MDL_VERSION_1_5:
                print(" Since MDL 1.5");
                break;
            case IMDL::MDL_VERSION_1_6:
                print(" Since MDL 1.6");
                break;
            case IMDL::MDL_VERSION_1_7:
                print(" Since MDL 1.7");
                break;
            case IMDL::MDL_VERSION_1_8:
                print(" Since MDL 1.8");
                break;
            case IMDL::MDL_VERSION_1_9:
                print(" Since MDL 1.9");
                break;
            case IMDL::MDL_VERSION_1_10:
                print(" Since MDL 1.10");
                break;
            case IMDL::MDL_VERSION_EXP:
                print(" Since MDL experimental");
                break;
            }
            switch (rem) {
            case IMDL::MDL_VERSION_1_0:
                break;
            case IMDL::MDL_VERSION_1_1:
                print(" Removed in MDL 1.1");
                break;
            case IMDL::MDL_VERSION_1_2:
                print(" Removed in MDL 1.2");
                break;
            case IMDL::MDL_VERSION_1_3:
                print(" Removed in MDL 1.3");
                break;
            case IMDL::MDL_VERSION_1_4:
                print(" Removed in MDL 1.4");
                break;
            case IMDL::MDL_VERSION_1_5:
                print(" Removed in MDL 1.5");
                break;
            case IMDL::MDL_VERSION_1_6:
                print(" Removed in MDL 1.6");
                break;
            case IMDL::MDL_VERSION_1_7:
                print(" Removed in MDL 1.7");
                break;
            case IMDL::MDL_VERSION_1_8:
                print(" Removed in MDL 1.8");
                break;
            case IMDL::MDL_VERSION_1_9:
                print(" Removed in MDL 1.9");
                break;
            case IMDL::MDL_VERSION_1_10:
                print(" Removed in MDL 1.10");
                break;
            case IMDL::MDL_VERSION_EXP:
                print(" Removed in MDL experimental");
                break;
            }
            print(insert ? " */" : "\n");
        }
    }
}

// Flush output.
void Printer::flush()
{
    m_type_printer.flush();
}

// Enable color output.
void Printer::enable_color(bool enable)
{
    if (enable) {
        // check if the given output stream supports color
        if (IOutput_stream_colored *cstr =
            m_type_printer.m_stream->get_interface<IOutput_stream_colored>()) {
            // yes
            m_c_ostr = cstr;
        } else {
            // no cannot be enabled
            enable = false;
        }
    }
    m_color_output = enable;
}

// Enable/disable positions output.
void Printer::show_positions(bool enable)
{
    m_show_positions = enable;
}

// Enable/disable all type modifiers.
void Printer::show_extra_modifiers(bool enable)
{
    m_type_printer.show_extra_modifiers(enable);
}

// Enable/disable MDL language levels.
void Printer::show_mdl_versions(bool enable)
{
    m_show_mdl_versions = enable;
}

// Enable/disable MDL resource table comments.
void Printer::show_resource_table(bool enable)
{
    m_show_res_table = enable;
}

// Enable/disable MDL function hash table comments.
void Printer::show_function_hash_table(bool enable)
{
    m_show_func_hashes = enable;
}

// Print namespace.
void Printer::print_namespace(IQualified_name const *name)
{
    if (name->is_absolute()) {
        print("::");
    }
    for (size_t i = 0, n = name->get_component_count(); i < n; ++i) {
        if (i > 0) {
            print("::");
        }
        push_color(C_LITERAL);

        ISymbol const *sym      = name->get_component(i)->get_symbol();
        bool           valid_id =
            sym->get_id() == ISymbol::SYM_DOT ||
            sym->get_id() == ISymbol::SYM_DOTDOT ||
            MDL::valid_mdl_identifier(sym->get_name());

        if (!valid_id) {
            print('"');
            print(sym->get_name());
            print('"');
        }
        else {
            print(sym);
        }
        pop_color();
    }
}

// Prints an annotation block.
void Printer::print_anno_block(
    IAnnotation_block const *block,
    char const              *prefix)
{
    if (block == NULL) {
        return;
    }
    if (prefix != NULL) {
        if (prefix[0] == '\n') {
            nl();
            ++prefix;
        }
        print(prefix);
    }
    print(block);
}

// --------------------------------- exporter ---------------------------------

/// Helper class.
class Sema_printer : public Printer
{
public:
    typedef Printer Base;

    /// Constructor.
    ///
    /// \param alloc        the allocator
    /// \param ostr         the output stream
    /// \param resource_cb  the resource callback
    Sema_printer(
        IAllocator                      *alloc,
        IOutput_stream                  *ostr,
        IMDL_exporter_resource_callback *resource_cb)
    : Base(alloc, ostr)
    , m_alloc(alloc)
    , m_sym_stack(Symbol_stack::container_type(get_allocator()))
    , m_resource_cb(resource_cb)
    , m_module(NULL)
    , m_global(NULL)
    , m_replace_auto(false)
    {
    }

    /// Prints a module.
    void print(IModule const *mod) MDL_FINAL;

    /// Print simple name.
    void print(ISimple_name const *name) MDL_FINAL;

    /// Print qualified name.
    void print(IQualified_name const *name) MDL_FINAL;

    /// Print type name.
    /// \param  name    The name to print.
    void print(IType_name const *name) MDL_FINAL;

    // Print enable_if annotation.
    void print(IAnnotation_enable_if const *anno) MDL_FINAL;

    /// Print annotation block.
    void print_anno_block(
        IAnnotation_block const *anno,
        char const              *prefix) MDL_FINAL;

    /// Print a type with current color.
    ///
    /// \param type  the type to print
    /// \param name  if type is a function type, print name instead of * if != NULL
    void print_type(
        IType const   *type,
        ISymbol const *name = NULL) MDL_FINAL;

    /// Print a type prefix (i.e. only the package name).
    ///
    /// \param e_type  the enum type to print
    void print_type_prefix(IType_enum const *e_type) MDL_FINAL;

    /// Returns true if a variable declaration of kind T v(a); can be rewritten as T v = a;
    virtual bool can_rewite_constructor_init(IExpression const * init) MDL_FINAL;

    /// Prints a resource value.
    ///
    /// \param res   the resource value
    void print_resource(IValue_resource const *res) MDL_FINAL;

    /// Given a definition, prints its scope.
    ///
    /// \param def  the imported definition
    ///
    /// \return false if the scope is empty, true otherwise
    bool print_scope(IDefinition const *def);

    /// Prints a definition name.
    ///
    /// \param def              the definition
    /// \param enforce_simple   if true, only print the name itself, ignore scope
    /// \param no_color_change  do not change color
    /// \param only_package     print only the package, not the name
    void print_def_name(
        IDefinition const *def,
        bool              enforce_simple,
        bool              no_color_change,
        bool              only_package);

    /// Set colors from another color table.
    ///
    /// \param table   a color table
    /// \param enable  true if color output is enabled, false otherwise
    void set_colors(Color_table const &table, bool enable);

private:
    /// The allocator.
    IAllocator *m_alloc;

    typedef stack<ISymbol const *>::Type Symbol_stack;

    /// Temporary used symbol stack.
    Symbol_stack m_sym_stack;

    /// The resource callback.
    IMDL_exporter_resource_callback *m_resource_cb;

    /// The current module.
    Module const *m_module;

    /// The global scope of the current module.
    Scope const *m_global;

    /// If true, replace 'auto' by deduced type.
    bool m_replace_auto;
};

// Prints a module.
void Sema_printer::print(IModule const *mod)
{
    Store<Module const *> m_store(m_module, impl_cast<Module>(mod));
    Store<Scope const *>  s_store(m_global, m_module->get_definition_table().get_global_scope());

    Base::print(mod);
}

// Print simple name.
void Sema_printer::print(ISimple_name const *name)
{
    IDefinition const *def = name->get_definition();

    if (def == NULL) {
        return Base::print(name);
    }

    print_def_name(
        def, /*enforce_simple=*/true, /*no_color_change=*/false, /*only_package=*/false);
}

// Print qualified name.
void Sema_printer::print(IQualified_name const *name)
{
    IDefinition const *def = name->get_definition();

    if (def == NULL) {
        return Base::print(name);
    }

    print_def_name(
        def, /*enforce_simple=*/false, /*no_color_change=*/false, /*only_package=*/false);
}

// Print type name.
void Sema_printer::print(IType_name const *name)
{
    if (m_replace_auto && name->is_auto_type()) {
        if (IType const *deduced = name->get_type()) {
            return print_type(deduced);
        }
        // type was not deduced, should not happen, fall-through
    }
    return Base::print(name);
}

// Print enable_if annotation.
void Sema_printer::print(IAnnotation_enable_if const *anno)
{
    print(anno->get_name());
    Printer::print("(\"");

    // print the expression into a captured color stream
    mi::base::Handle<IOutput_stream_colored> safe_c(m_c_ostr);

    Allocator_builder builder(get_allocator());
    m_c_ostr = mi::base::make_handle(builder.create<Captured_color_stream>(get_allocator()));
    m_type_printer.m_stream = m_c_ostr.get();

    IExpression const *expr = anno->get_expression();
    Printer::print(expr);

    m_type_printer.m_stream = m_type_printer.m_holder.get();
    m_c_ostr.swap(safe_c);

    Captured_color_stream const *s = static_cast<Captured_color_stream *>(safe_c.get());
    s->replay(m_type_printer.m_stream);

    Printer::print("\")");
}

// Prints an annotation block.
void Sema_printer::print_anno_block(
    IAnnotation_block const *block,
    char const              *prefix)
{
    if (block == NULL) {
        return;
    }
    size_t n = block->get_annotation_count();
    if (n == 0) {
        // empty annotation blocks are not valid in MDL
        return;
    }
    return Base::print_anno_block(block, prefix);
}

// Print a type with current color.
void Sema_printer::print_type(IType const *type, ISymbol const *name)
{
    switch (type->get_kind()) {
    case IType::TK_STRUCT:
    case IType::TK_ENUM:
        break;
    default:
        return Base::print_type(type, name);
    }

    Definition_table const &def_tab = m_module->get_definition_table();
    Scope const *scope = def_tab.get_type_scope(type);
    if (scope == NULL) {
        return Base::print_type(type, name);
    }
    Definition const *def = scope->get_owner_definition();
    if (def == NULL) {
        return Base::print_type(type, name);
    }
    print_def_name(
        def, /*enforce_simple=*/false, /*no_color_change=*/true, /*only_package=*/false);
}

// Print a type prefix (i.e. only the package name).
void Sema_printer::print_type_prefix(IType_enum const *e_type)
{
    Definition_table const &def_tab = m_module->get_definition_table();
    Scope const *scope = def_tab.get_type_scope(e_type);
    if (scope == NULL) {
        return Base::print_type_prefix(e_type);
    }
    Definition const *def = scope->get_owner_definition();
    if (def == NULL) {
        return Base::print_type_prefix(e_type);
    }
    print_def_name(
        def, /*enforce_simple=*/false, /*no_color_change=*/true, /*only_package=*/true);
}

// Prints a resource value.
void Sema_printer::print_resource(IValue_resource const *res)
{
    char const *res_name = NULL;
    if (m_resource_cb != NULL) {
        // try to get the resource name from the user callback
        res_name = m_resource_cb->get_resource_name(
            res, m_module->get_mdl_version() >= IMDL::MDL_VERSION_1_3);
    }
    if (res_name == NULL || res_name[0] == '\0') {
        // no valid resource name: use the name from the resource value itself
        res_name = res->get_string_value();
    }

    switch (res->get_kind()) {
    case IValue::VK_TEXTURE:
        {
            IValue_texture const *tex      = cast<IValue_texture>(res);
            IType_texture const  *tex_type = tex->get_type();

            print_type(tex_type);
            Base::print("(\"");
            Base::print_utf8(res_name, /*escape=*/true);
            Base::print("\", ::tex::");

            char const *s = "gamma_default";
            switch (tex->get_gamma_mode()) {
            case IValue_texture::gamma_default:
                s = "gamma_default";
                break;
            case IValue_texture::gamma_linear:
                s = "gamma_linear";
                break;
            case IValue_texture::gamma_srgb:
                s = "gamma_srgb";
                break;
            }
            Base::print(s);

            if (m_module->get_mdl_version() >= IMDL::MDL_VERSION_1_7 &&
                (is_tex_2d(tex_type) || is_tex_3d(tex_type))) {
                // 2D and 3D textures have a selector
                Base::print(", \"");
                Base::print_utf8(tex->get_selector(), /*escape=*/true);
                Base::print("\"");
            }
            Base::print(')');
            break;
        }
    case IValue::VK_LIGHT_PROFILE:
    case IValue::VK_BSDF_MEASUREMENT:
        {
            print_type(res->get_type());
            Base::print("(\"");
            Base::print_utf8(res_name, /*escape=*/true);
            Base::print("\")");
            break;
        }
    default:
        MDL_ASSERT(!"unsupported ressource type");
        break;
    }
}

// Given a definition, prints its scope.
bool Sema_printer::print_scope(IDefinition const *idef)
{
    Store<bool> in_module_name(m_in_module_name, true);
    Definition const *def   = impl_cast<Definition>(idef);
    Scope const      *scope = def->get_def_scope();

    if (def->get_kind() == Definition::DK_CONSTRUCTOR) {
        // constructors live "inside" its type-scope,
        // skip that to create valid MDL syntax.
        if (scope != NULL && scope->get_scope_type() != NULL) {
            scope = scope->get_parent();
        }
    }

    for (; scope != m_global && scope != NULL; scope = scope->get_parent()) {
        if (ISymbol const *sym = scope->get_scope_name()) {
            m_sym_stack.push(sym);
        }
    }

    if (m_sym_stack.empty() &&
        def->get_def_scope() == m_module->get_definition_table().get_global_scope())
    {
        // need "::" in front
        m_sym_stack.push(NULL);
    }

    if (m_sym_stack.empty()) {
        return false;
    }

    bool first = true;
    while (!m_sym_stack.empty()) {
        ISymbol const *sym = m_sym_stack.top();

        if (sym == NULL) {
            Base::print("::");
        } else {
            if (first) {
                first = false;
            } else {
                Base::print("::");
            }
            Base::print(sym);
        }
        m_sym_stack.pop();
    }
    return !first;
}

// Prints a definition name.
void Sema_printer::print_def_name(
    IDefinition const *idef,
    bool              enforce_simple,
    bool              no_color_change,
    bool              only_package)
{
    Definition const *def = impl_cast<Definition>(idef);

    if (!no_color_change) {
        switch (def->get_kind()) {
        case Definition::DK_ERROR:         ///< This is an error definition.
            push_color(ISyntax_coloring::C_ERROR);
            break;
        case Definition::DK_CONSTANT:      ///< This is a constant entity.
        case Definition::DK_ENUM_VALUE:    ///< This is an enum value.
        case Definition::DK_ARRAY_SIZE:    ///< This is a constant array size.
            push_color(ISyntax_coloring::C_LITERAL);
            break;
        case Definition::DK_ANNOTATION:    ///< This is an annotation.
            push_color(ISyntax_coloring::C_ANNOTATION);
            break;
        case Definition::DK_TYPE:          ///< This is a type.
        case Definition::DK_CONSTRUCTOR:   ///< This is a constructor.
            push_color(ISyntax_coloring::C_TYPE);
            break;
        case Definition::DK_FUNCTION:      ///< This is a function.
        case Definition::DK_VARIABLE:      ///< This is a variable.
        case Definition::DK_MEMBER:        ///< This is a field member.
        case Definition::DK_PARAMETER:     ///< This is a parameter.
        case Definition::DK_OPERATOR:      ///< This is an operator.
        case Definition::DK_STRUCT_CATEGORY:      ///< This is a struct category.
        case Definition::DK_NAMESPACE:     ///< This is a namespace.
            push_color(ISyntax_coloring::C_ENTITY);
            break;
        }
    }

    if (!enforce_simple && def->get_kind() == Definition::DK_MEMBER) {
        // members are always "simple"
        enforce_simple = true;
    }

    bool has_scope = false;

    if (!enforce_simple) {
        if (def->get_semantics() == Definition::DS_CONV_OPERATOR) {
            // the name of the conversion operator is ALWAYS its return type
            IType_function const *f_type = cast<IType_function>(def->get_type());
            IType const *ret_type = f_type->get_return_type();
            return print_type(ret_type, NULL);
        }
        has_scope = print_scope(def);
    }

    if (has_scope) {
        Base::print("::");
    }
    if (!only_package) {
        Base::print(def->get_sym());
    }

    if (!no_color_change) {
        pop_color();
    }
}

// Set colors.
void Sema_printer::set_colors(Color_table const &table, bool enable)
{
    enable_color(enable);
    m_color_table = table;
}

// Returns true if a variable declaration of kind T v(a); can be rewritten as T v = a;
bool Sema_printer::can_rewite_constructor_init(IExpression const * init)
{
    if (!is<IExpression_call>(init)) {
        return false;
    }

    IExpression_call const      *call = cast<IExpression_call>(init);
    IExpression_reference const *ref  = as<IExpression_reference>(call->get_reference());

    if (ref == NULL) {
        return false;
    }

    IDefinition const *def = ref->get_definition();
    if (def == NULL) {
        return false;
    }

    IDefinition::Semantics sema = def->get_semantics();
    return sema == IDefinition::DS_COPY_CONSTRUCTOR ||
        sema == IDefinition::DS_CONV_CONSTRUCTOR ||
        sema == IDefinition::DS_MATRIX_DIAG_CONSTRUCTOR;
}

// Constructor.
MDL_exporter::MDL_exporter(IAllocator *alloc)
: Base(alloc)
, m_builder(alloc)
, m_color_output(false)
{
}

// Destructor.
MDL_exporter::~MDL_exporter()
{
}

// Export a module.
void MDL_exporter::export_module(
    IOutput_stream                  *ostr,
    IModule const                   *module,
    IMDL_exporter_resource_callback *resource_cb)
{
    if (module == NULL) {
        return;
    }
    if (!module->is_valid()) {
        return;
    }

    mi::base::Handle<Sema_printer> printer(
        m_builder.create<Sema_printer>(m_builder.get_allocator(), ostr, resource_cb));

    printer->set_colors(Base::m_color_table, m_color_output);
    printer->print(module);
}

// Enable color output.
void MDL_exporter::enable_color(bool enable)
{
    m_color_output = enable;
}

}  // mdl
}  // mi
