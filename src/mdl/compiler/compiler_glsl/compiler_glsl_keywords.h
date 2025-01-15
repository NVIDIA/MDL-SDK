/***************************************************************************************************
 * Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************************************/

#ifndef MDL_COMPILER_GLSL_KEYWORDS_H
#define MDL_COMPILER_GLSL_KEYWORDS_H 1

#include <mdl/compiler/compilercore/compilercore_memory_arena.h>
#include "compiler_glsl_version.h"

namespace mi {
namespace mdl {
namespace glsl {

/// Computes a hash for a C-string.
unsigned glsl_cstring_hash(char const* data, size_t len);

//-------------------------------------------------------------------------------------------
// KeywordMap  -- maps strings to integers (identifiers to keyword kinds)
//-------------------------------------------------------------------------------------------
class GLSLKeywordMap {
private:
    /// An Element in the keyword map.
    class Elem {
    public:
        char const *key;
        size_t len;
        int val;
        bool is_future_keyword;
        Elem *next;

        /// Constructor.
        Elem(char const *key, size_t len, int val, bool future_keyword)
        : key(key)
        , len(len)
        , val(val)
        , is_future_keyword(future_keyword)
        , next(NULL)
        {}
    };

    Arena_builder builder;
    Elem *tab[128];

public:
    /// Constructor.
    GLSLKeywordMap(Memory_arena &arena, GLSLang_context &ctx)
    : builder(arena)
    {
        memset(tab, 0, 128 * sizeof(Elem*));

        ctx.register_keywords(this);
    }

    /// Enter a keyword.
    void set(char const *key, int val, bool future_keyword = false)
    {
        size_t len = strlen(key);
        Elem *e = builder.create<Elem>(key, len, val, future_keyword);
        unsigned k = glsl_cstring_hash(key, len) & 127;
        e->next = tab[k]; tab[k] = e;
    }

    /// Get a keyword value.
    int get(int len, char const *key, int defaultVal) const
    {
        const Elem *e = tab[glsl_cstring_hash(key, len) & 127];
        while (e != NULL && (len != e->len || strncmp(e->key, key, len) != 0)) {
            e = e->next;
        }
        return e == NULL ? defaultVal : e->val;
    }

    /// Notify the the GLSLang version has changes.
    void glsl_version_changed(GLSLang_context &glslang_ctx);

    /// Check if the given identifier is a keyword or a reserved word.
    bool keyword_or_reserved(char const *s, size_t len) const;

private:
    /// Set the token value of a keyword depending on the keyword state.
    void set_keyword(char const *keyword, int val, GLSL_keyword_state state);

    /// Fill the map initially.
    void init();
};

}  // glsl
}  // mdl
}  // mi

#endif
