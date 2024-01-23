/******************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDLTLC_PPRINT_H
#define MDLTLC_PPRINT_H 1

#include <iostream>

#include <mdl/compiler/compilercore/compilercore_memory_arena.h>

namespace pp {

class Pretty_print {
  public:
    static const int LARGE_LINE_WIDTH = 1000000;

    enum Flags: unsigned {
        PRINT_TYPES = 0x01,
        PRINT_ATTRIBUTE_TYPES = 0x02,
        PRINT_RETURN_TYPES = 0x04,
        PRINT_ALL_TYPES = PRINT_TYPES
            | PRINT_ATTRIBUTE_TYPES
            | PRINT_RETURN_TYPES
    };


  public:
    Pretty_print(mi::mdl::Memory_arena &arena,
                 std::ostream &out,
                 int line_width = 120)
        : m_arena(arena)
        , m_out(out)
        , m_margin(line_width)
        , m_indent_level(0)
        , m_current_col(0)
        , m_pending_whitespace(0)
        , m_indent_spaces(4)
        , m_flags(0)
    {
    }

    void set_flags(unsigned flags);
    void remove_flags(unsigned flags);
    unsigned flags();

    void string(char const *s);
    void string_with_nl(char const *s);
    void chr(char c);
    void escaped_string(char const *s);
    void integer(int i);
    void floating_point(double d);
    void lparen() { string("("); }
    void rparen() { string(")"); }
    void lbrace() { string("{"); }
    void rbrace() { string("}"); }
    void lbracket() { string("["); }
    void rbracket() { string("]"); }
    void comma() { string(","); }
    void colon() { string(":"); }
    void semicolon() { string(";"); }
    void dot() { string("."); }
    void oper(char const *s, bool spaces = false) {
        if (spaces)
            space();
        string(s);
        if (spaces)
            space();
    }
    void nl();
    void space();
    void softbreak();

    void with_parens(std::function< void(Pretty_print &) > wrapped);
    void with_brackets(std::function< void(Pretty_print &) > wrapped);
    void with_braces(std::function< void(Pretty_print &) > wrapped);
    void with_indent(std::function< void(Pretty_print &) > wrapped);
    void without_indent(std::function< void(Pretty_print &) > wrapped);

    Pretty_print &operator++() {
        m_indent_level += 1;
        return *this;
    }

    Pretty_print &operator--() {
        m_indent_level -= 1;
        return *this;
    }

  private:
    void clear();
    void indent();

    mi::mdl::Memory_arena &m_arena;

    // Output stream.
    std::ostream &m_out;

    // Linewidth limit.
    int m_margin;

    // Current indentation level.
    int m_indent_level;

    // Current column position.
    int m_current_col;

    // Whitespace that needs to be printed, but hasn't yet (avoids
    // trailing whitespace).
    int m_pending_whitespace;

    // How many spaces to indent by for each level of indentation.
    int m_indent_spaces;

    // Printing flags.
    unsigned m_flags;
};

} // namespace pp

#endif // MDLTLC_PPRINT_H
