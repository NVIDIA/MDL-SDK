/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/mdl/mdl_values.h>
#include <mi/mdl/mdl_symbols.h>

namespace mi {
namespace mdl {

// Check if two coordinate spaces are equal.
bool equal_coordinate_space(
    IValue const *a,
    IValue const *b,
    const char   *internal_space)
{
    if (strcmp(internal_space, "*") == 0) {
        // magic constant '*' means all 3 spaces are equal
        return true;
    }
    IValue_enum const *ae = as<IValue_enum>(a);
    IValue_enum const *be = as<IValue_enum>(b);
    if (is<IValue_enum>(ae) && is<IValue_enum>(b)) {
        IType_enum const *at = ae->get_type();
        IType_enum const *bt = be->get_type();
        if (at == bt) {
            ISymbol const *sym = at->get_symbol();
            if (strcmp(sym->get_name(), "::state::coordinate_space") == 0) {
                int av = ae->get_value();
                int bv = be->get_value();
                if (av == bv)
                    return true;
                int coordinate_internal = -1;
                int coordinate_target = -1;
                int count = at->get_value_count();
                for (int i = 0; i < count; ++i) {
                    ISymbol const*es;
                    int ec;
                    at->get_value(i,es,ec);
                    char const *name = es->get_name();
                    if (strcmp(name, "coordinate_internal") == 0)
                        coordinate_internal = ec;
                    else if (strcmp(name, internal_space) == 0)
                        coordinate_target = ec;
                }
                if (coordinate_internal < 0 || coordinate_target < 0)
                    return false;
                if (av == coordinate_internal)
                    av = coordinate_target;
                if (bv == coordinate_internal)
                    bv = coordinate_target;
                return av == bv;
            }
        }
    }
    return false;
}

} // mdl
} // mi
