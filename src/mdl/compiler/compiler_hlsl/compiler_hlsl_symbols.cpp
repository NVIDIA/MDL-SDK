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

#include <cstring>
#include <cstdio>
#include "compiler_hlsl_symbols.h"
#include "compiler_hlsl_assert.h"

#undef FMT_SIZE_T
#if defined(MI_ARCH_64BIT) || defined(MI_PLATFORM_MACOSX)
#ifdef MI_PLATFORM_WINDOWS
#  define FMT_SIZE_T        "llu"
#else
#  define FMT_SIZE_T        "zu"
#endif
#else
#  define FMT_SIZE_T        "u"
#endif

namespace mi {
namespace mdl {
namespace hlsl {

// Constructor for an operator symbol.
Symbol::Symbol(char const *name)
: m_id(Symbol::SYM_OPERATOR)
, m_name(name)
{
}

// Constructor for a syntax element symbol.
Symbol::Symbol(size_t id, char const *name)
: m_id(id)
, m_name(name)
{
}

// ------------------------------- symbol table -------------------------------

namespace {
// create all predefined symbols
#define OPERATOR_SYM(sym_name, id, name)   Symbol sym_name(name);
#define PREDEF_SYM(sym_name, id, name)     Symbol sym_name(Symbol::id, name);

#include "compiler_hlsl_predefined_symbols.h"
}  // anonymous

// Constructor.
Symbol_table::Symbol_table(Memory_arena &arena)
: m_builder(arena)
, m_string_arena(arena.get_allocator())
, m_symbol_map(256, Symbol_map::hasher(), Symbol_map::key_equal(), &arena)
, m_symbols(256, NULL, &arena)
, m_next_id(Symbol::SYM_USER)
, m_anon_id(0)
{
    create_predefined();
}

// Get or create a new Symbol for the given name.
Symbol *Symbol_table::lookup_symbol(char const *name) const
{
    HLSL_ASSERT(name != NULL);
    Symbol_map::const_iterator it = m_symbol_map.find(name);
    if (it == m_symbol_map.end()) {
        return NULL;
    }
    return it->second;
}

// Get or create a new Symbol for the given name.
Symbol *Symbol_table::get_symbol(char const *name)
{
    if (Symbol *ret = lookup_symbol(name))
        return ret;

    name = internalize(name);
    return enter_symbol(name, m_next_id++);
}

// Return the symbol for a given id.
Symbol *Symbol_table::get_symbol_for_id(size_t id) const
{
    if (id >= m_symbols.size())
        return NULL;
    return m_symbols[id];
}

// Return the error symbol.
Symbol *Symbol_table::get_error_symbol()
{
    return &sym_error;
}

// Return a predefined symbol.
Symbol *Symbol_table::get_predefined_symbol(Symbol::Predefined_id id)
{
    switch (id) {
    case Symbol::SYM_OPERATOR:                     return NULL; // a set of symbols
    case Symbol::SYM_USER:                         return NULL; // first non-predefined
    case Symbol::SYM_USER_TYPE_NAME:               return NULL; // a set of symbols
    case Symbol::SYM_KEYWORD:                      return NULL; // a set of symbols
    case Symbol::SYM_RESERVED:                     return NULL; // a set of symbols

#define PREDEF_SYM(sym_name, id, name) case Symbol::id: return &sym_name;
#define KEYWORD(k)
#define RESERVED(w)
#include "compiler_hlsl_predefined_symbols.h"
    }
    return NULL;
}

// Get or create a new Symbol for the given user type name.
Symbol *Symbol_table::get_user_type_symbol(char const *name)
{
    if (Symbol *ret = lookup_symbol(name))
        return ret;

    name = internalize(name);
    return enter_symbol(name, Symbol::SYM_USER_TYPE_NAME);
}

// Create a new anonymous symbol.
Symbol *Symbol_table::get_anonymous_symbol()
{
    char buf[32];

    snprintf(buf, sizeof(buf), "{anon_%" FMT_SIZE_T "}", m_anon_id++);
    buf[sizeof(buf) - 1] = '\0';

    char const *name = internalize(buf);
    return enter_symbol(name, Symbol::SYM_USER_TYPE_NAME);
}

// Create all predefined symbols
void Symbol_table::create_predefined()
{
    // enter all predefined names, ignore the operators
#define PREDEF_SYM(sym_name, id, name) enter_predefined(&sym_name);
#include "compiler_hlsl_predefined_symbols.h"
}

// Create a predefined symbol with a given Id.
Symbol *Symbol_table::enter_symbol(char const *ident, size_t id)
{
    Symbol *res = m_builder.create<Symbol>(id, ident);
    m_symbol_map[ident] = res;

    size_t size = m_symbols.size();
    if (id >= size) {
        size = ((size + 1) + 0x3F) & ~0x3F;
        m_symbols.resize(size, NULL);
    }
    m_symbols[id] = res;
    return res;
}

// Enter a predefined symbol into the table.
void Symbol_table::enter_predefined(Symbol *sym)
{
    size_t id = sym->get_id();
    m_symbol_map[sym->get_name()] = sym;

    size_t size = m_symbols.size();
    if (id >= size) {
        size = ((size + 1) + 0x3F) & ~0x3F;
        m_symbols.resize(size, NULL);
    }
    m_symbols[id] = sym;
}

// Create a save copy of a string by putting it into the memory arena.
char const *Symbol_table::internalize(char const *s)
{
    size_t l = strlen(s);
    void *p = m_string_arena.allocate(l + 1, 1);

    if (p != NULL) {
        memcpy(p, s, l + 1);
    }
    return reinterpret_cast<const char *>(p);
}

}  // hlsl
}  // mdl
}  // mi
