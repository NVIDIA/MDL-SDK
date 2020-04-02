/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "compilercore_symbols.h"

#include <cstring>

#include <mi/mdl/mdl_symbols.h>
#include <mi/base/interface_implement.h>
#include <mi/base/iallocator.h>

#include <string>
#include <algorithm>

#include "compilercore_memory_arena.h"
#include "compilercore_assert.h"
#include "compilercore_serializer.h"

namespace mi {
namespace mdl {

namespace {
// create all predefined symbols
#define OPERATOR_SYM(sym_name, id, name)   Symbol sym_name(name);
#define PREDEF_SYM(sym_name, id, name)     Symbol sym_name(ISymbol::id, name);

#include "compilercore_predefined_symbols.h"
};

Symbol_table::Symbol_table(Memory_arena &arena)
: m_builder(arena)
, m_string_arena(arena.get_allocator())
, m_symbol_map(256, Symbol_map::hasher(), Symbol_map::key_equal(), &arena)
, m_symbols(256, NULL, &arena)
, m_next_id(ISymbol::SYM_FREE)
{
    create_predefined();
}

// Create all predefined symbols
void Symbol_table::create_predefined()
{
    // enter all predefined names, ignore the operators
#define PREDEF_SYM(sym_name, id, name) enter_predefined(&sym_name);
#include "compilercore_predefined_symbols.h"
}

// Register all predefined symbols in the serializer.
void Symbol_table::register_predefined(Factory_serializer &serializer) const
{
    // register all predefined symbols. When this happens, the symbol
    // tag set must be empty, check that.
    Tag_t t, check = Tag_t(0);

#define OPERATOR_SYM(sym_name, id, name) \
    t = serializer.register_symbol(&sym_name); ++check; MDL_ASSERT(t == check);
#define PREDEF_SYM(sym_name, id, name) \
    t = serializer.register_symbol(&sym_name); ++check; MDL_ASSERT(t == check);
#include "compilercore_predefined_symbols.h"
}

// Register all predefined symbols in the deserializer.
void Symbol_table::register_predefined(Factory_deserializer &deserializer) const
{
    // register all predefined symbols. When this happens, the symbol
    // tag set must be empty, check that.
    Tag_t t = Tag_t(0);

#define OPERATOR_SYM(sym_name, id, name) \
     ++t; deserializer.register_symbol(t, &sym_name);
#define PREDEF_SYM(sym_name, id, name) \
     ++t; deserializer.register_symbol(t, &sym_name);
#include "compilercore_predefined_symbols.h"
}

// Create a save copy of a string by putting it into the memory arena.
char const *Symbol_table::internalize(char const *s)
{
    size_t l = strlen(s);
    void *p = m_string_arena.allocate(l + 1, 1);

    if (p != NULL) {
        memcpy(p, s, l + 1);
    }
    return reinterpret_cast<char const *>(p);
}

// Create a symbol.
ISymbol const *Symbol_table::create_symbol(char const *name)
{
    return get_symbol(name);
}

// Create a symbol for the given user type name.
ISymbol const *Symbol_table::create_user_type_symbol(char const *name)
{
    return get_user_type_symbol(name);
}

// Create a shared symbol (no unique ID).
ISymbol const *Symbol_table::create_shared_symbol(char const *name)
{
    return get_shared_symbol(name);
}

// Get or create a new Symbol for the given name.
ISymbol const *Symbol_table::lookup_symbol(char const *name) const
{
    Symbol_map::const_iterator it = m_symbol_map.find(name);
    if (it == m_symbol_map.end()) {
        return NULL;
    }
    return it->second;
}

// Get an existing operator Symbol for the given name.
ISymbol const *Symbol_table::lookup_operator_symbol(
    char const *op_name,
    size_t     n_params) const
{
    size_t l = strlen(op_name);

    if (n_params == 1) {
#define OPERATOR_SYM_UNARY(sym_name, id, name) \
    if (l == sizeof(name) - 1 && strcmp(op_name, name) == 0) return &sym_name;
#include "compilercore_predefined_symbols.h"
    } else if (n_params == 2) {
#define OPERATOR_SYM_BINARY(sym_name, id, name) \
    if (l == sizeof(name) - 1 && strcmp(op_name, name) == 0) return &sym_name;
#include "compilercore_predefined_symbols.h"
    } else if (n_params == 3) {
#define OPERATOR_SYM_TERNARY(sym_name, id, name) \
    if (l == sizeof(name) - 1 && strcmp(op_name, name) == 0) return &sym_name;
#include "compilercore_predefined_symbols.h"
    } else {
#define OPERATOR_SYM_VARIADIC(sym_name, id, name) \
    if (l == sizeof(name) - 1 && strcmp(op_name, name) == 0) return &sym_name;
#include "compilercore_predefined_symbols.h"
    }
    return NULL;
}

// Get or create a new Symbol for the given name.
ISymbol const *Symbol_table::get_symbol(char const *name)
{
    if (ISymbol const *ret = lookup_symbol(name))
        return ret;

    name = internalize(name);
    return enter_symbol(name, m_next_id++);
}

// Return the symbol for a given id.
ISymbol const *Symbol_table::get_symbol_for_id(size_t id) const
{
    if (id >= m_symbols.size())
        return NULL;
    return m_symbols[id];
}

// Return the error symbol.
ISymbol const *Symbol_table::get_error_symbol() const
{
    return &sym_error;
}

// Find the equal symbol of this symbol table if it exists.
ISymbol const *Symbol_table::find_equal_symbol(ISymbol const *other) const
{
    Symbol_map::const_iterator it = m_symbol_map.find(other->get_name());
    if (it != m_symbol_map.end())
        return it->second;
    return NULL;
}

// Return the symbol for an operator.
ISymbol const *Symbol_table::get_operator_symbol(IExpression::Operator op) const
{
    switch (op) {
        // handlke all operators
#define OPERATOR_SYM(sym_name, id, name) case IExpression::id: return &sym_name;
#include "compilercore_predefined_symbols.h"
    }
    MDL_ASSERT(!"Unknown operator kind");
    return NULL;
}

// Return a predefined symbol.
ISymbol const *Symbol_table::get_predefined_symbol(ISymbol::Predefined_id id)
{
    switch (id) {
    case ISymbol::SYM_OPERATOR:    return NULL; // a set of symbols
    case ISymbol::SYM_SHARED_NAME: return NULL; // a set of symbols
    case ISymbol::SYM_FREE:        return NULL; // first non-predefined

#define PREDEF_SYM(sym_name, id, name) case ISymbol::id: return &sym_name;
#include "compilercore_predefined_symbols.h"
    }
    return NULL;
}

// Get or create a new Symbol for the given user type name.
ISymbol const *Symbol_table::get_user_type_symbol(char const *name)
{
    return get_shared_symbol(name);
}

// Get or create a new shared Symbol for the given name.
ISymbol const *Symbol_table::get_shared_symbol(char const *name)
{
    if (ISymbol const *ret = lookup_symbol(name))
        return ret;

    name = internalize(name);
    return enter_symbol(name, ISymbol::SYM_SHARED_NAME);
}

// Create a predefined symbol with a given Id.
ISymbol const *Symbol_table::enter_symbol(char const *ident, size_t id)
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



namespace {

/// Helper class to compare symbols.
struct Symbol_less {
    bool operator() (Symbol const *s, Symbol const *t)
    {
        size_t s_id = s->get_id();
        size_t t_id = t->get_id();

        if (s_id == t_id) {
            return strcmp(s->get_name(), t->get_name()) < 0;
        }
        return s_id < t_id;
    }
};

}  // anonymous

// Serialize the symbol table.
void Symbol_table::serialize(Factory_serializer &serializer) const
{
    register_predefined(serializer);

    serializer.write_section_tag(Serializer::ST_SYMBOL_TABLE);

    // remember the symbol table itself for its users
    Tag_t sym_tag = serializer.register_symbol_table(this);
    serializer.write_encoded_tag(sym_tag);

    DOUT(("symtable %u\n", unsigned(sym_tag)));
    INC_SCOPE();

    // first step: count the number of entires.
    // Note that this cannot be retrieved from m_next_id, because a symbol map
    // can contain type names, which share all ONE ID.
    size_t num_symbols = 0;
    for (Symbol_map::const_iterator it(m_symbol_map.begin()), end(m_symbol_map.end());
         it != end;
         ++it)
    {
        Symbol const *sym = it->second;

        if (!sym->is_predefined())
            ++num_symbols;
    }

#ifdef NO_MDL_SERIALIZATION_SORT
    typedef Symbol_map Symbols;
    Symbols const &symbols = m_symbol_map;
#else
    // sort them
    typedef vector<Symbol const *>::Type Symbols;

    Symbols symbols(serializer.get_allocator());
    symbols.reserve(num_symbols);

    for (Symbol_map::const_iterator it(m_symbol_map.begin()), end(m_symbol_map.end());
        it != end;
        ++it)
    {
        Symbol const *sym = it->second;

        if (sym->is_predefined())
            continue;

        symbols.push_back(sym);
    }

    std::sort(symbols.begin(), symbols.end(), Symbol_less());
#endif

    // now do it
    serializer.write_encoded_tag(num_symbols);
    for (Symbols::const_iterator it(symbols.begin()), end(symbols.end());
        it != end;
        ++it)
    {
#ifdef NO_MDL_SERIALIZATION_SORT
        Symbol const *sym = it->second;
#else
        Symbol const *sym = *it;
#endif

        if (sym->is_predefined())
            continue;

        Tag_t sym_tag = serializer.register_symbol(sym);

        serializer.write_encoded_tag(sym_tag);
        serializer.write_encoded_tag(sym->get_id());
        serializer.write_cstring(sym->get_name());

        DOUT(("symbol %u (%u %s)\n", unsigned(sym_tag), unsigned(sym->get_id()), sym->get_name()));
    }

    // no need to serialize m_symbols, will be automatically recreated
    serializer.write_encoded_tag(m_next_id);
    DEC_SCOPE();
}

// Deserialize the symbol table.
void Symbol_table::deserialize(Factory_deserializer &deserializer)
{
    register_predefined(deserializer);

    Serializer::Serializer_tags t;

    t = deserializer.read_section_tag();
    MDL_ASSERT(t == Serializer::ST_SYMBOL_TABLE);

    // get the tag of this symbol table itself for its users
    Tag_t sym_tag = deserializer.read_encoded_tag();
    deserializer.register_symbol_table(sym_tag, this);

    DOUT(("symtable %u (%p)\n", unsigned(sym_tag), this));
    INC_SCOPE();

    // read symbols
    size_t num_symbols = deserializer.read_encoded_tag();
    for (size_t i = 0; i < num_symbols; ++i) {
        Tag_t sym_tag = deserializer.read_encoded_tag();
        size_t id     = deserializer.read_encoded_tag();
        char const *s = deserializer.read_cstring();

        DOUT(("symbol %u (%u %s)\n", unsigned(sym_tag), unsigned(id), s));

        s = internalize(s);
        ISymbol const *sym = enter_symbol(s, id);

        deserializer.register_symbol(sym_tag, sym);
    }

    // no need to deserialize m_symbols, will be automatically recreated
    m_next_id = deserializer.read_encoded_tag();
    DEC_SCOPE();
}

// ----------------------------- Symbol -----------------------------

// Get the name of the symbol.
char const *Symbol::get_name() const
{
    return m_name;
}

// Get the id of the symbol.
size_t Symbol::get_id() const
{
    return m_id;
}

// Returns true if this symbol is predefined.
bool Symbol::is_predefined() const
{
    return m_id < ISymbol::SYM_SHARED_NAME;
}

// Constructor for an operator symbol.
Symbol::Symbol(char const *name)
: Base()
, m_id(ISymbol::SYM_OPERATOR)
, m_name(name)
{
}

// Constructor for a syntax element symbol.
Symbol::Symbol(size_t id, char const *name)
: Base()
, m_id(id)
, m_name(name)
{
}

}  // mdl
}  // mi
