/******************************************************************************
 * Copyright (c) 2007-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief  a cast that reinterprets bit patterns in a warning-free way
///
///     binary_cast<Target, Source>()
///             Cast a constant value reference of type 'Source' to type
///             'Target' by re-interpreting the binary representation. This
///             cast is equivalent to C++'s reinterpret_cast<T>().
///
///     Code that deals with serialized representations of integral machine
///     types requires type-casting to switch between interpretations of a
///     given memory area from, say, a floating point value to an octet stream
///     and back. The MSG module, for example, provides a serializer for
///     'float' that is implemented by obtaining the address of the value,
///     casting that 'float*' to 'Uint32*', and then dereferencing that pointer
///     to obtain the raw binary representation of the value:
///
///       void set_float(Uint8///buffer, float float_value)
///       {
///           Uint32 value = *(Uint32*)&float_value;
///           buffer[0] = (Uint8)((value >> 24) & 0xff);
///           buffer[1] = (Uint8)((value >> 16) & 0xff);
///           buffer[2] = (Uint8)((value >> 8) & 0xff);
///           buffer[3] = (Uint8)(value & 0xff);
///       }
///
///     This code is perfectly legal, but still gcc 4.1.1 will issue a warning
///     when compiling the cast:
///
///       dereferencing type-punned pointer will break strict-aliasing rules
///
///     Unfortunately, the warning is substantial because the optimizer is
///     likely to break that code. In section 3.10, "Options That Control
///     Optimization", the GCC manual says about this subject:
///
///      | [Strict aliasing] activates optimizations based on the type of
///      | expressions. In particular, an object of one type is assumed never
///      | to reside at the same address as an object of a different type,
///      | unless the types are almost the same. For example, an `unsigned int'
///      | can alias an `int', but not a `void*' or a `double'. [...] Pay
///      | special attention to code like this:
///      |
///      | union a_union
///      | {
///      |    int i;
///      |    double d;
///      | };
///      |
///      | int f()
///      | {
///      |    a_union t;
///      |    t.d = 3.0;
///      |    return t.i;
///      | }
///      |
///      | The practice of reading from a different union member than the one
///      | most recently written to (called "type-punning") is common. Even
///      | with `-fstrict-aliasing', type-punning is allowed, provided the
///      | memory is accessed through the union type. So, the code above will
///      | work as expected. However, this code might not:
///      |
///      | int f()
///      | {
///      |   a_union t;
///      |   int* ip;
///      |   t.d = 3.0;
///      |   ip = &t.i;
///      |   return *ip;
///      | }
///
///     The function template binary_cast() defined in this module encapsulates
///     the union idiom transparently. To the user, the cast looks exactly like
///     a C++ reinterpret_cast().


#ifndef BASE_SYSTEM_STLEXT_BINARY_CAST_H
#define BASE_SYSTEM_STLEXT_BINARY_CAST_H

#include <mi/base/assert.h>

namespace MI { namespace STLEXT {

namespace                        // helper class for the function defined below
{
    template <class Target, class Source>
    union Binary_cast
    {
        Source  source;
        Target  target;
    };
}

//
// Cast an immutable 'Source' value to an immutable 'Target' value. Use this
// function as follows:
//
//     float  fval( 0.0f );
//     Uint32 uval( binary_cast<Uint32>(fval) );
//

template <class Target, class Source>
inline Target binary_cast(Source const & val)
{
    mi_static_assert(sizeof(Source) == sizeof(Target));
    Binary_cast<Target, Source> val_;
    val_.source = val;
    return val_.target;
}

}} // MI::STLEXT

#endif // BASE_SYSTEM_STLEXT_BINARY_CAST_H
