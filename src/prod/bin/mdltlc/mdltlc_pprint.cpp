/******************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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
#include <cstring>

#include "mdltlc_pprint.h"

namespace pp {

void Pretty_print::set_flags(unsigned flags) {
    m_flags |= flags;
}

void Pretty_print::remove_flags(unsigned flags) {
    m_flags &= ~flags;
}

unsigned Pretty_print::flags() {
    return m_flags;
}

void Pretty_print::string(char const *s) {
    int l = strlen(s);
    clear();
    m_current_col += l;
    m_out << s;
}

void Pretty_print::string_with_nl(char const *s) {
    clear();
    while (*s) {
        if (*s == '\n') {
            nl();
        } else {
            clear();
            m_current_col += 1;
            m_out << *s;
        }
        s++;
    }
}

void Pretty_print::chr(char c) {
    clear();
    m_current_col += 1;
    m_out << c;
}

void Pretty_print::escaped_string(char const *s) {
    clear();
    while (*s) {
        switch (*s) {
        case '\'':
        case '\"':
        case '\\':
            m_current_col += 2;
            m_out << '\\' << *s;
            break;
        default:
            m_current_col += 1;
            m_out << *s;
            break;
        }
        s++;
    }
}

void Pretty_print::integer(int i) {
    std::stringstream s;
    s << i;
    clear();
    m_current_col += s.str().size();
    m_out << s.str();
}

void Pretty_print::floating_point(double d) {
    std::stringstream s;
    s << d;
    clear();
    m_current_col += s.str().size();
    m_out << s.str();
}

void Pretty_print::nl() {
    m_out << "\n";
    m_pending_whitespace = 0;
    m_current_col = 0;
    indent();
}

void Pretty_print::space() {
    if (m_current_col + m_pending_whitespace >= m_margin) {
        nl();
    } else {
        m_pending_whitespace++;
    }
}

void Pretty_print::softbreak() {
    if (m_current_col + m_pending_whitespace >= m_margin) {
        nl();
    }
}

void Pretty_print::with_parens(std::function< void(Pretty_print &) > wrapped) {
    lparen();
    wrapped(*this);
    rparen();
}

void Pretty_print::with_braces(std::function< void(Pretty_print &) > wrapped) {
    lbrace();
    wrapped(*this);
    rbrace();
}

void Pretty_print::with_brackets(std::function< void(Pretty_print &) > wrapped) {
    lbracket();
    wrapped(*this);
    rbracket();
}

void Pretty_print::with_indent(std::function< void(Pretty_print &) > wrapped) {
    ++(*this);
    wrapped(*this);
    --(*this);
}

void Pretty_print::without_indent(std::function< void(Pretty_print &) > wrapped) {
    int old_indent_level = m_indent_level;
    m_indent_level = 0;
    m_current_col = 0;
    m_pending_whitespace = 0;
    wrapped(*this);
    m_indent_level = old_indent_level;
}

void Pretty_print::indent() {
    m_pending_whitespace += m_indent_level * m_indent_spaces;
}

void Pretty_print::clear() {
    if (m_pending_whitespace > 0) {
        for (int i = 0; i < m_pending_whitespace; i++) {
            m_current_col++;
            m_out << ' ';
        }
        m_pending_whitespace = 0;
    }
}

} // namespace pp
