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

#include "pch.h"

#include "mdltlc_symbols.h"

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

// ------------------------------- symbols ------------------------------------

// Constructor for a syntax element symbol.
Symbol::Symbol(char const *name)
    : m_name(name)
{
}

// ------------------------------- symbol table -------------------------------

// Constructor.
Symbol_table::Symbol_table(mi::mdl::Memory_arena &arena)
  : m_builder(arena)
  , m_string_arena(arena.get_allocator())
  , m_symbol_map(256, Symbol_map::hasher(), Symbol_map::key_equal(), &arena)
  , m_symbols(256, NULL, &arena)
{
}

// Get or create a new Symbol for the given name.
Symbol *Symbol_table::lookup_symbol(char const *name) const
{
    MDL_ASSERT(name != NULL);
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
    return enter_symbol(name);
}

Symbol *Symbol_table::enter_symbol(char const *ident)
{
    Symbol *res = m_builder.create<Symbol>(ident);
    m_symbol_map[ident] = res;

    return res;
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
