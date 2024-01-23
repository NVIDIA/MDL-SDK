/******************************************************************************
 * Copyright (c) 2010-2024, NVIDIA CORPORATION. All rights reserved.
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

/// \file
/// \brief regression test public/mi/base components
///
/// See \ref mi_base_iinterface
///

#include "pch.h"
#include <base/system/test/i_test_auto_case.h>

#include <mi/base/atom.h>

using mi::Uint32;

MI_TEST_AUTO_FUNCTION( test_atom )
{
    mi::base::Atom32 a;
    mi::base::Atom32 a0(Uint32(0));
    mi::base::Atom32 a1(Uint32(1));
    mi::base::Atom32 a2(Uint32(2));

    MI_CHECK_EQUAL( Uint32(a),  Uint32(0));
    MI_CHECK_EQUAL( Uint32(a0), Uint32(0));
    MI_CHECK_EQUAL( Uint32(a1), Uint32(1));
    MI_CHECK_EQUAL( Uint32(a2), Uint32(2));

    a = Uint32(1);
    MI_CHECK_EQUAL( Uint32(a),  Uint32(1));

    mi::base::Atom32 b(1);
    b = (a = a0);
    MI_CHECK_EQUAL( Uint32(a),  Uint32(0));
    MI_CHECK_EQUAL( Uint32(b),  Uint32(0));

    b = ( a += 2);
    MI_CHECK_EQUAL( Uint32(a),  Uint32(2));
    MI_CHECK_EQUAL( Uint32(b),  Uint32(2));

    b = ( a -= 2);
    MI_CHECK_EQUAL( Uint32(a),  Uint32(0));
    MI_CHECK_EQUAL( Uint32(b),  Uint32(0));

    MI_CHECK_EQUAL( Uint32(a++),  Uint32(0));
    MI_CHECK_EQUAL( Uint32(a),  Uint32(1));

    MI_CHECK_EQUAL( Uint32(a--),  Uint32(1));
    MI_CHECK_EQUAL( Uint32(a),  Uint32(0));

    MI_CHECK_EQUAL( Uint32(++a),  Uint32(1));
    MI_CHECK_EQUAL( Uint32(a),  Uint32(1));

    MI_CHECK_EQUAL( Uint32(--a),  Uint32(0));
    MI_CHECK_EQUAL( Uint32(a),  Uint32(0));

    MI_CHECK_EQUAL( Uint32(a.swap(42)),  Uint32(0));
    MI_CHECK_EQUAL( Uint32(a),  Uint32(42));
}

