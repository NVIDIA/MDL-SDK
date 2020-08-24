/******************************************************************************
 * Copyright (c) 2014-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/mdl/mdl_types.h>

#include "compilercore_mangle.h"
#include "compilercore_def_table.h"
#include "compilercore_function_instance.h"
#include "compilercore_tools.h"
#include "compilercore_wchar_support.h"

#include <cstdio>

namespace mi {
namespace mdl {

// --------------------------- Punycode ---------------------------

/// This class implements a modified Punycode encoder. Originally taken from
///
/// punycode-sample.c 2.0.0 (2004-Mar-21-Sun)
/// http://www.nicemice.net/idn/
/// Adam M. Costello
/// http://www.nicemice.net/amc/
///
/// This is ANSI C code (C89) implementing Punycode 1.0.x.
///
/// This single file contains three sections (an interface, an
/// implementation, and a wrapper for testing) that would normally belong
/// in three separate files (punycode.h, punycode.c, punycode-test.c), but
/// here they are bundled into one file (punycode-sample.c) for convenient
/// testing.  Anyone wishing to reuse this code will probably want to split
/// it apart.
///
/// Identifiers that contain non-ASCII characters are encoded using the Punycode algorithm
/// specified in RFC 3492, with the modifications that '_' is used as the encoding delimiter,
/// and uppercase letters A through J are used in place of digits 0 through 9 in the encoding
/// character set. (Taken from Swift name mangling to avoid digits in front of a generated
/// string)
class Punycode_encoder_swift {
public:
    enum punycode_status {
        punycode_success    = 0,
        punycode_bad_input  = 1, /* Input is invalid.                       */
        punycode_big_output = 2, /* Output would exceed the space provided. */
        punycode_overflow   = 3  /* Wider integers needed to process input. */
    };

    /// Constructor.
    Punycode_encoder_swift(
        IAllocator *alloc)
    : m_alloc(alloc)
    {
    }

    /// Check if given string needs encoding.
    static bool needs_encoding(
        char const *s, size_t len)
    {
        for (size_t i = 0; i < len; ++i) {
            if (!is_basic(s[i]))
                return true;
        }
        return false;
    }

    /// Encode an UTF-8 string.
    string encode(char const *utf8) {
        u32string tmp(m_alloc);

        utf8_to_utf32(tmp, utf8);

        string res(m_alloc);
        punycode_encode(tmp, res);
        return res;
    }

    /// Decode an UTF-8 string.
    string decode(char const *s) {
        u32string tmp(m_alloc);
        punycode_decode(strlen(s), s, tmp);

        string res(m_alloc);
        utf32_to_utf8(res, tmp.c_str());
        return res;
    }

private:
    static unsigned const maxint = 0xffffffff;

    enum {
        base = 36,
        tmin = 1,
        tmax = 26,
        skew = 38,
        damp = 700,
        initial_bias = 72,
        initial_n = 0x80,
        delimiter = '_'
    };

    ///< tests whether cp is a basic code point, in out version an character allowed
    static bool is_basic(unsigned cp) {
        if ('a' <= cp && cp <= 'z') return true;
        if ('A' <= cp && cp <= 'Z') return true;
        if ('0' <= cp && cp <= '9') return true;
        if ('_' == cp) return true;
        return false;
    }

    /// tests whether cp is a delimiter
    static bool is_delim(unsigned cp) { return cp == delimiter; }

    /// returns the basic code point whose value
    /// (when used for representing integers) is d, which needs to be in
    /// the range 0 to base-1
    static char encode_digit(unsigned d) {
        //  0..25 map to ASCII a..z
        // 26..35 map to ASCII A..J
        return d + ('A' - 26) + (('a' - 'A' + 26) * (d < 26));
    }

    /// returns the numeric value of a basic code point (for use in representing integers)
    /// in the range 0 to  base-1, or base if cp does not represent a value.
    static unsigned decode_digit(unsigned cp) {
        return  (cp - 'A') < 10 ? cp - 'A' + 26 : cp - 'a' < 26 ? cp - 'a' : base;
    }

    /// Bias adaptation function
    static unsigned adapt(
        unsigned delta,
        unsigned numpoints,
        int firsttime)
    {
        unsigned k;

        delta = firsttime ? delta / damp : delta >> 1;
        /* delta >> 1 is a faster way of doing delta / 2 */
        delta += delta / numpoints;

        for (k = 0; delta > ((base - tmin) * tmax) / 2; k += base) {
            delta /= base - tmin;
        }

        return k + (base - tmin + 1) * delta / (delta + skew);
    }

    /// Main encode function
    static enum punycode_status punycode_encode(
        u32string const &input,
        string          &output)
    {
        unsigned input_length, n, delta, h, b, bias, j, m, q, k, t;
        size_t out, max_out;

        /* The Punycode spec assumes that the input length is the same type */
        /* of integer as a code point, so we need to convert the size_t to  */
        /* a punycode_uint, which could overflow.                           */

        if (input.length() > maxint)
            return punycode_overflow;
        input_length = (unsigned)input.length();

        /* Initialize the state: */

        n = initial_n;
        delta = 0;
        out = 0;
        max_out = 64*1024;    //< 64k identifiers will probably be too long anyway
        bias = initial_bias;

        /* Handle the basic code points: */

        for (j = 0; j < input_length; ++j) {
            if (is_basic(input[j])) {
                if (max_out - out < 2)
                    return punycode_big_output;
                output.append((char)input[j]);
                ++out;
            }
            /* else if (input[j] < n) return punycode_bad_input; */
            /* (not needed for Punycode with unsigned code points) */
        }

        h = b = (unsigned)out;
        /* cannot overflow because out <= input_length <= maxint */

        /* h is the number of code points that have been handled, b is the  */
        /* number of basic code points, and out is the number of ASCII code */
        /* points that have been output.                                    */

        if (b > 0) {
            output.append(char(delimiter));
            ++out;
        }

        /* Main encoding loop: */

        while (h < input_length) {
            /* All non-basic code points < n have been     */
            /* handled already.  Find the next larger one: */

            for (m = maxint, j = 0; j < input_length; ++j) {
                /* if (basic(input[j])) continue; */
                /* (not needed for Punycode) */
                if (input[j] >= n && input[j] < m)
                    m = input[j];
            }

            /* Increase delta enough to advance the decoder's    */
            /* <n,i> state to <m,0>, but guard against overflow: */

            if (m - n > (maxint - delta) / (h + 1))
                return punycode_overflow;
            delta += (m - n) * (h + 1);
            n = m;

            for (j = 0; j < input_length; ++j) {
                /* Punycode does not need to check whether input[j] is basic: */
                if (input[j] < n /* || basic(input[j]) */) {
                    if (++delta == 0)
                        return punycode_overflow;
                }

                if (input[j] == n) {
                    /* Represent delta as a generalized variable-length integer: */

                    for (q = delta, k = base; ; k += base) {
                        if (out >= max_out)
                            return punycode_big_output;
                        t = k <= bias /* + tmin */ ? tmin :     /* +tmin not needed */
                            k >= bias + tmax ? tmax : k - bias;
                        if (q < t)
                            break;
                        output.append(encode_digit(t + (q - t) % (base - t)));
                        ++out;
                        q = (q - t) / (base - t);
                    }

                    output.append(encode_digit(q));
                    ++out;
                    bias = adapt(delta, h + 1, h == b);
                    delta = 0;
                    ++h;
                }
            }

            ++delta, ++n;
        }

        return punycode_success;
    }

    /// Main decode function
    enum punycode_status punycode_decode(
        size_t input_length,
        const char input[],
        u32string &output)
    {
        unsigned n, out, i, max_out, bias, oldi, w, k, digit, t;
        size_t b, j, in;

        /* Initialize the state: */

        n = initial_n;
        out = i = 0;
        max_out = maxint;
        bias = initial_bias;

        /* Handle the basic code points:  Let b be the number of input code */
        /* points before the last delimiter, or 0 if there is none, then    */
        /* copy the first b code points to the output.                      */

        for (b = j = 0; j < input_length; ++j)
            if (is_delim(input[j]))
                b = j;

        if (b > max_out)
            return punycode_big_output;

        for (j = 0; j < b; ++j) {
            if (!is_basic(input[j]))
                return punycode_bad_input;
            output.append(unsigned(input[j]));
            ++out;
        }

        /* Main decoding loop:  Start just after the last delimiter if any  */
        /* basic code points were copied; start at the beginning otherwise. */

        for (in = b > 0 ? b + 1 : 0; in < input_length; ++out) {

            /* in is the index of the next ASCII code point to be consumed, */
            /* and out is the number of code points in the output array.    */

            /* Decode a generalized variable-length integer into delta,  */
            /* which gets added to i.  The overflow checking is easier   */
            /* if we increase i as we go, then subtract off its starting */
            /* value at the end to obtain delta.                         */

            for (oldi = i, w = 1, k = base; ; k += base) {
                if (in >= input_length)
                    return punycode_bad_input;
                digit = decode_digit(input[in++]);
                if (digit >= base)
                    return punycode_bad_input;
                if (digit > (maxint - i) / w)
                    return punycode_overflow;
                i += digit * w;
                t = k <= bias /* + tmin */ ? tmin :     /* +tmin not needed */
                    k >= bias + tmax ? tmax : k - bias;
                if (digit < t)
                    break;
                if (w > maxint / (base - t))
                    return punycode_overflow;
                w *= (base - t);
            }

            bias = adapt(i - oldi, out + 1, oldi == 0);

            /* i was supposed to wrap around from out+1 to 0,   */
            /* incrementing n each time, so we'll fix that now: */

            if (i / (out + 1) > maxint - n)
                return punycode_overflow;
            n += i / (out + 1);
            i %= (out + 1);

            /* Insert n at position i of the output: */

            /* not needed for Punycode: */
            /* if (basic(n)) return punycode_bad_input; */
            if (out >= max_out)
                return punycode_big_output;

            /* cannot overflow because out <= old value of *output_length */
            if (out + 1 > output.size())
                output.resize(out + 1);
            memmove(&output[i + 1], &output[i], (out - i) * sizeof(output[0]));
            output[i++] = n;
        }

        output.resize(size_t(out));
        /* cannot overflow because out <= old value of *output_length */
        return punycode_success;
    }

private:
    IAllocator *m_alloc;
};

// --------------------------- MDL mangler ---------------------------

// Constructor.
MDL_name_mangler::MDL_name_mangler(IAllocator *alloc, string &out)
: m_alloc(alloc)
, m_out(out)
, m_ignore_uniform_varying(false)
{
}

// Mangle a function declaration.
bool MDL_name_mangler::mangle_function_decl(
    char const              *prefix,
    IDefinition const       *def,
    Function_instance const *inst)
{
  // If we get here, mangle the decl name!
  m_out.append("_Z");
  mangle_function_encoding(prefix, def, inst);
  return true;
}

// Mangle an MDL entity (function/non-function).
bool MDL_name_mangler::mangle(
    char const        *prefix,
    IDefinition const *def)
{
    // <mangled-name> ::= _Z <encoding>
    //            ::= <data name>
    //            ::= <special-name>
    IType const *type = def->get_type();

    if (is<IType_function>(type))
        return mangle_function_decl(prefix, def, NULL);

    m_out.append("_Z");
    mangle_name(prefix, def->get_symbol()->get_name(), NULL);
    return true;
}

// Mangle an MDL function instance.
bool MDL_name_mangler::mangle(
    char const              *prefix,
    Function_instance const &inst)
{
    // <mangled-name> ::= _Z <encoding>
    //            ::= <data name>
    //            ::= <special-name>
    if (!inst.is_instantiated())
        return mangle_function_decl(prefix, inst.get_def(), NULL);
    return mangle_function_decl(prefix, inst.get_def(), &inst);
}

bool MDL_name_mangler::mangle_name(
    char const *prefix,
    char const *name)
{
    // <mangled-name> ::= _Z <encoding>
    m_out.append("_Z");
    mangle_name(prefix, name, NULL);
    return true;
}

// Ignore uniform/varying modifiers.
void MDL_name_mangler::ignore_uniform_varying()
{
    m_ignore_uniform_varying = true;
}

// <special-name> ::= TV <type>  # virtual table
// <special-name> ::= TI <type>  # typeinfo structure
// <special-name> ::= GV <object name>	# Guard variable for one-time
//                                      # initialization

void MDL_name_mangler::mangle_function_encoding(
    char const              *prefix,
    IDefinition const       *def,
    Function_instance const *inst)
{
    // <encoding> ::= <function name> <bare-function-type>
    mangle_name(prefix, def->get_symbol()->get_name(), inst);

    // Whether the mangling of a function type includes the return type depends
    // on the context and the nature of the function. The rules for deciding
    // whether the return type is included are:
    //
    //   1. Template functions (names or types) have return types encoded, with
    //   the exceptions listed below.
    //   2. Function types not appearing as part of a function name mangling,
    //   e.g. parameters, pointer types, etc., have return type encoded, with the
    //   exceptions listed below.
    //   3. Non-template function names do not have return types encoded.
    //
    // The exceptions mentioned in (1) and (2) above, for which the return
    // type is never included, are
    //   1. Constructors.
    //   2. Destructors.
    //   3. Conversion operator functions, e.g. operator int.
    bool mangleReturnType = inst != NULL;
    mangle_bare_function_type(
        cast<IType_function>(def->get_type()), mangleReturnType, inst);
}

void MDL_name_mangler::mangle_name(
    char const              *prefix,
    char const              *name,
    Function_instance const *inst)
{
    //  <name> ::= <nested-name>
    //         ::= <unscoped-name>
    //         ::= <unscoped-template-name> <template-args>
    //         ::= <local-name>     # See Scope Encoding below
    //
    //  <unscoped-name> ::= <unqualified-name>
    //                  ::= St <unqualified-name>   # ::std::
    if (prefix != NULL) {
        if (prefix[0] == ':' && prefix[1] == ':')
            prefix += 2;
        if (prefix[0] == '\0')
            prefix = NULL;
    }
    if (prefix == NULL)
        mangle_unqualified_name(name, inst);
    else
        mangle_nested_name(prefix, name, inst);
}

void MDL_name_mangler::mangle_type_name(ISymbol const *sym)
{
    // <nested-name> ::= N [<CV-qualifiers>] <prefix> <unqualified-name> E
    //               ::= N [<CV-qualifiers>] <template-prefix> <template-args> E
    m_out.append(1, 'N');
    mangle_prefix(sym->get_name());
    m_out.append(1, 'E');
}

void MDL_name_mangler::mangle_unqualified_name(
    char const              *name,
    Function_instance const *inst)
{
    //  <unqualified-name> ::= <operator-name>
    //                     ::= <ctor-dtor-name>
    //                     ::= <source-name>
    mangle_source_name(name);

    if (inst != NULL)
        mangle_template_argument_list(*inst);
}

// <operator-name> ::= cv <type>	# (cast)

void MDL_name_mangler::mangle_source_name(char const *name)
{
    // <source-name> ::= <positive length number> <identifier>
    // <number> ::= [n] <non-negative decimal integer>
    // <identifier> ::= <unqualified source code identifier>
    size_t l = strlen(name);

    char buf[16];

    if (Punycode_encoder_swift::needs_encoding(name, l)) {
        string s = Punycode_encoder_swift(get_allocator()).encode(name);

        snprintf(buf, sizeof(buf), "0%u", unsigned(s.length()));
        m_out.append(buf);
        m_out.append(s);
    } else {
        snprintf(buf, sizeof(buf), "%u", unsigned(l));
        m_out.append(buf);
        m_out.append(name);
    }
}

void MDL_name_mangler::mangle_nested_name(
    char const              *prefix,
    char const              *name,
    Function_instance const *inst)
{
    // <nested-name> ::= N [<CV-qualifiers>] <prefix> <unqualified-name> E
    //               ::= N [<CV-qualifiers>] <template-prefix> <template-args> E
    m_out.append(1, 'N');
    mangle_prefix(prefix);
    mangle_unqualified_name(name, inst);
    m_out.append(1, 'E');
}

void MDL_name_mangler::mangle_prefix(char const *prefix)
{
    //  <prefix> ::= <prefix> <unqualified-name>
    //           ::= <template-prefix> <template-args>
    //           ::= <template-param>
    //           ::= # empty
    //           ::= <substitution>
    if (prefix == NULL)
        return;

    // skip leading "::"
    if (prefix[0] == ':' && prefix[1] == ':')
        prefix += 2;

    // handle mdl paths
    for (;prefix[0] != '\0';) {
        size_t l = 0;
        for (; prefix[l] != '\0'; ++l) {
            if (prefix[l] == ':' && prefix[l + 1] == ':')
                break;
        }

        char buf[16];
        if (Punycode_encoder_swift::needs_encoding(prefix, l)) {
            string p(prefix, prefix + l, get_allocator());
            string s = Punycode_encoder_swift(get_allocator()).encode(p.c_str());

            snprintf(buf, sizeof(buf), "0%u", unsigned(s.length()));
            m_out.append(buf);
            m_out.append(s);
        } else {
            snprintf(buf, sizeof(buf), "%u", unsigned(l));
            m_out.append(buf);
            m_out.append(prefix, l);
        }

        prefix += l;
        if (prefix[0] == ':' && prefix[1] == ':')
            prefix += 2;
    }
}

// <operator-name> ::= nw     # new
//              ::= na        # new[]
//              ::= dl        # delete
//              ::= da        # delete[]
//              ::= ps        # + (unary)
//              ::= pl        # +
//              ::= ng        # - (unary)
//              ::= mi        # -
//              ::= ad        # & (unary)
//              ::= an        # &
//              ::= de        # * (unary)
//              ::= ml        # *
//              ::= co        # ~
//              ::= dv        # /
//              ::= rm        # %
//              ::= or        # |
//              ::= eo        # ^
//              ::= aS        # =
//              ::= pL        # +=
//              ::= mI        # -=
//              ::= mL        # *=
//              ::= dV        # /=
//              ::= rM        # %=
//              ::= aN        # &=
//              ::= oR        # |=
//              ::= eO        # ^=
//              ::= ls        # <<
//              ::= rs        # >>
//              ::= lS        # <<=
//              ::= rS        # >>=
//              ::= eq        # ==
//              ::= ne        # !=
//              ::= lt        # <
//              ::= gt        # >
//              ::= le        # <=
//              ::= ge        # >=
//              ::= nt        # !
//              ::= aa        # &&
//              ::= oo        # ||
//              ::= pp        # ++
//              ::= mm        # --
//              ::= cm        # ,
//              ::= pm        # ->*
//              ::= pt        # ->
//              ::= cl        # ()
//              ::= ix        # []
//              ::= qu        # ?

void MDL_name_mangler::mangle_type_qualifiers(IType::Modifiers mod) {
    // <CV-qualifiers> ::= [r] [V] [K] 	# restrict (C99), volatile, const

    // In cases where multiple order-insensitive qualifiers are present, they should be ordered
    // 'K' (closest to the base type), 'V', 'r', and 'U' (farthest from the base type), with the
    // 'U' qualifiers in alphabetical order by the vendor name (with alphabetically earlier names
    // closer to the base type). For example, int* volatile const restrict _far p has mangled
    // type name U4_farrVKPi.

    if (!m_ignore_uniform_varying) {
        if (mod & IType::MK_VARYING) {
            m_out.append(1, 'U');
            mangle_source_name("varying");
        }
        if (mod & IType::MK_UNIFORM) {
            m_out.append(1, 'U');
            mangle_source_name("uniform");
        }
    }
    if (mod & IType::MK_CONST)
        m_out.append(1, 'K');
}

void MDL_name_mangler::mangle_type(IType const *type)
{
    IType::Modifiers mod = type->get_type_modifiers();
    type = type->skip_type_alias();

    //  <type> ::= <CV-qualifiers> <type>
    mangle_type_qualifiers(mod);

    //         ::= <class-enum-type>
    if (IType_struct const *s_tp = as<IType_struct>(type))
        mangle_type(s_tp);
    //         ::= <class-enum-type>
    else if (IType_enum const *e_tp = as<IType_enum>(type))
        mangle_type(e_tp);
    //         ::= <builtin-type>
    else if (IType_atomic const *a_tp = as<IType_atomic>(type))
        mangle_type(a_tp);
    //         ::= <function-type>
    else if (IType_function const *f_tp = as<IType_function>(type))
        mangle_type(f_tp);
    //         ::= <array-type>
    else if (IType_array const *a_tp = as<IType_array>(type))
        mangle_type(a_tp);
    else if (is<IType_string>(type))
        m_out.append("PKc"); // encode as char const *
    //         ::= <pointer-to-member-type>
    //         ::= <template-param>
    //         ::= <template-template-param> <template-args>
    //         ::= <substitution> # See Compression below
    //         ::= P <type>   # pointer-to
    //         ::= R <type>   # reference-to
    //         ::= O <type>   # rvalue reference-to (C++0x)
    //         ::= C <type>   # complex pair (C 2000)
    //         ::= G <type>   # imaginary (C 2000)
    //         ::= U <source-name> <type>     # vendor extended type qualifier
    else if (IType_vector const *v_tp = as<IType_vector>(type))
        mangle_type(v_tp);
    else if (IType_matrix const *m_tp = as<IType_matrix>(type))
        mangle_type(m_tp);
    else if (IType_reference const *r_tp = as<IType_reference>(type))
        mangle_type(r_tp);
    else if (IType_color const *c_tp = as<IType_color>(type))
        mangle_type(c_tp);
    else
        MDL_ASSERT(!"Cannot mangle unknown type");
}

void MDL_name_mangler::mangle_type(IType_atomic const *type)
{
    //  <builtin-type> ::= v  # void
    //                 ::= w  # wchar_t
    //                 ::= b  # bool
    //                 ::= c  # char
    //                 ::= a  # signed char
    //                 ::= h  # unsigned char
    //                 ::= s  # short
    //                 ::= t  # unsigned short
    //                 ::= i  # int
    //                 ::= j  # unsigned int
    //                 ::= l  # long
    //                 ::= m  # unsigned long
    //                 ::= x  # long long, __int64
    //                 ::= y  # unsigned long long, __int64
    //                 ::= n  # __int128
    //                 ::= o  # unsigned __int128
    //                 ::= f  # float
    //                 ::= d  # double
    //                 ::= e  # long double, __float80
    //                 ::= g  # __float128
    //                 ::= Dd # IEEE 754r decimal floating point (64 bits)
    //                 ::= De # IEEE 754r decimal floating point (128 bits)
    //                 ::= Df # IEEE 754r decimal floating point (32 bits)
    //                 ::= Dh # IEEE 754r half-precision floating point (16 bits)
    //                 ::= Di # char32_t
    //                 ::= Ds # char16_t
    //                 ::= u <source-name>    # vendor extended type
    switch (type->get_kind()) {
    case IType::TK_BOOL:   m_out.append(1, 'b'); break;
    case IType::TK_INT:    m_out.append(1, 'i'); break;
    case IType::TK_FLOAT:  m_out.append(1, 'f'); break;
    case IType::TK_DOUBLE: m_out.append(1, 'd'); break;

    default:
        MDL_ASSERT(!"unexpected atomic type kind");
    }
}

void MDL_name_mangler::mangle_type(IType_function const *type)
{
    // <function-type> ::= F [Y] <bare-function-type> E
    m_out.append(1, 'F');
    mangle_bare_function_type(type, /*MangleReturnType=*/true, NULL);
    m_out.append(1, 'E');
}

void MDL_name_mangler::mangle_bare_function_type(
    IType_function const    *type,
    bool                    MangleReturnType,
    Function_instance const *inst)
{
    // <bare-function-type> ::= <signature type>+
    if (MangleReturnType) {
        IType const *ret_tp = type->get_return_type();
        int arr_size = -1;
        if (IType_array const *a_ret_tp = as<IType_array>(ret_tp)) {
            // check if the array size was instantiated
            if (inst != NULL) {
                arr_size = inst->instantiate_type_size(a_ret_tp);
            }
            if (arr_size >= 0) {
                mangle_array_type(a_ret_tp->get_element_type(), arr_size);
            } else {
                mangle_type(a_ret_tp);
            }
        } else {
            mangle_type(ret_tp);
        }
    }

    int n_params = type->get_parameter_count();
    if (n_params == 0) {
        m_out.append(1, 'v');
        return;
    }

    for (int i = 0; i < n_params; ++i) {
        IType const   *param_type;
        ISymbol const *dummy;

        type->get_parameter(i, param_type, dummy);

        int arr_size = -1;
        // check if the array size was instantiated
        if (IType_array const *a_param_type = as<IType_array>(param_type)) {
            if (inst != NULL) {
                arr_size = inst->instantiate_type_size(a_param_type);
            }
            if (arr_size >= 0) {
                mangle_array_type(a_param_type->get_element_type(), arr_size);
            } else {
                mangle_type(a_param_type);
            }
        } else {
            mangle_type(param_type);
        }
    }

    // <builtin-type>      ::= z  # ellipsis
}

void MDL_name_mangler::mangle_type(IType_struct const *type)
{
    //  <class-enum-type> ::= <name>
    mangle_type_name(type->get_symbol());
}

void MDL_name_mangler::mangle_type(IType_enum const *type)
{
    //  <class-enum-type> ::= <name>
    mangle_type_name(type->get_symbol());
}

void MDL_name_mangler::mangle_type(IType_array const *type)
{
    // <array-type> ::= A <positive dimension number> _ <element type>
    //              ::= A <deferred size> _ <element type>
    m_out.append(1, 'A');
    if (type->is_immediate_sized()) {
        mangle_array_type(type->get_element_type(), type->get_size());
    } else {
        IType_array_size const *size = type->get_deferred_size();
        mangle_deferred_size(size);

        m_out.append(1, '_');
        mangle_type(type->get_element_type());
    }
}

void MDL_name_mangler::mangle_array_type(
    IType const *type,
    int         size)
{
    // <array-type> ::= A <positive dimension number> _ <element type>
    //              ::= A <deferred size> _ <element type>
    m_out.append(1, 'A');
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", size);
    m_out.append(buf);

    m_out.append(1, '_');
    mangle_type(type);
}

void MDL_name_mangler::mangle_type(IType_vector const *type)
{
    // ::= u <source-name>	# vendor extended type
    m_out.append(1, 'u');

    IType_atomic const *e_tp = type->get_element_type();
    int                size  = type->get_size();

    char buffer[32];
    switch (e_tp->get_kind()) {
    case IType::TK_BOOL:
        snprintf(buffer, sizeof(buffer), "bool%d", size);
        break;
    case IType::TK_INT:
        snprintf(buffer, sizeof(buffer), "int%d", size);
        break;
    case IType::TK_FLOAT:
        snprintf(buffer, sizeof(buffer), "float%d", size);
        break;
    case IType::TK_DOUBLE:
        snprintf(buffer, sizeof(buffer), "double%d", size);
        break;
    default:
        MDL_ASSERT(!"unsupported vector type");
        snprintf(buffer, sizeof(buffer), "bad%d", size);
    }
    buffer[sizeof(buffer) - 1] = '\0';
    mangle_source_name(buffer);
}

void MDL_name_mangler::mangle_type(IType_matrix const *type)
{
    // ::= u <source-name>	# vendor extended type
    m_out.append(1, 'u');

    char buf[32];
    int cols = type->get_columns();
    IType_vector const *v_type = type->get_element_type();
    int rows = v_type->get_size();
    snprintf(buf, sizeof(buf), "%dx%d", cols, rows);
    buf[sizeof(buf) - 1] = '\0';

    IType_atomic const *e_tp = v_type->get_element_type();
    char buffer[38];
    switch (e_tp->get_kind()) {
    case IType::TK_FLOAT:
        snprintf(buffer, sizeof(buffer), "float%s", buf);
        break;
    case IType::TK_DOUBLE:
        snprintf(buffer, sizeof(buffer), "double%s", buf);
        break;
    default:
        MDL_ASSERT(!"unsupported vector type");
        snprintf(buffer, sizeof(buffer), "bad%s", buf);
    }
    buffer[sizeof(buffer) - 1] = '\0';
    mangle_source_name(buffer);

}

void MDL_name_mangler::mangle_type(IType_color const *type)
{
    // ::= u <source-name>	# vendor extended type
    m_out.append(1, 'u');
    mangle_source_name("color");
}

void MDL_name_mangler::mangle_type(IType_reference const *type)
{
    // ::= u <source-name>	# vendor extended type
    m_out.append(1, 'u');

    switch (type->get_kind()) {
    case IType::TK_BSDF:
        mangle_source_name("bsdf");
        break;
    case IType::TK_HAIR_BSDF:
        mangle_source_name("hair_bsdf");
        break;
    case IType::TK_VDF:
        mangle_source_name("vdf");
        break;
    case IType::TK_EDF:
        mangle_source_name("edf");
        break;
    case IType::TK_LIGHT_PROFILE:
        mangle_source_name("light_profile");
        break;
    case IType::TK_TEXTURE:
        {
            IType_texture const *t_type = cast<IType_texture>(type);
            switch (t_type->get_shape()) {
            case IType_texture::TS_2D:
                mangle_source_name("texture_2d");
                break;
            case IType_texture::TS_3D:
                mangle_source_name("texture_3d");
                break;
            case IType_texture::TS_CUBE:
                mangle_source_name("texture_cube");
                break;
            case IType_texture::TS_PTEX:
                mangle_source_name("texture_ptex");
                break;
            case IType_texture::TS_BSDF_DATA:
                mangle_source_name("texture_bsdf_data");
            }
        }
        break;
    case IType::TK_BSDF_MEASUREMENT:
        mangle_source_name("bsdf_measurement");
        break;
    default:
        MDL_ASSERT(!"unsupported resource type");
        mangle_source_name("bad");
    }
}

// <pointer-to-member-type> ::= M <class type> <member type>
// <template-param> ::= T_    # first template parameter
//                  ::= T <parameter-2 non-negative number> _

void MDL_name_mangler::mangle_deferred_size(IType_array_size const *size)
{
// <deferred size> ::= D <mangled-name>
    m_out.append(1, 'D');
    char const *name = size->get_name()->get_name();
    if (char const *p = strrchr(name, ':')) {
        string prefix(name, p - 1 - name, m_alloc);
        name = p + 1;

        mangle_name(prefix.c_str(), name, NULL);
    } else {
        mangle_name(/*prefix=*/NULL, name, NULL);
    }
}

// <ctor-dtor-name> ::= C1  # complete object constructor
//                  ::= C2  # base object constructor
//                  ::= C3  # complete object allocating constructor
//
// <ctor-dtor-name> ::= D0  # deleting destructor
//                  ::= D1  # complete object destructor
//                  ::= D2  # base object destructor
//

void MDL_name_mangler::mangle_template_argument_list(
    Function_instance const &inst)
{
    // <template-args> ::= I <template-arg>+ E
    m_out.append(1, 'I');

    Function_instance::Array_instances const &ais(inst.get_array_instances());
    for (size_t i = 0, n = ais.size(); i < n; ++i) {
         Array_instance const &ai = ais[i];

        mangle_template_argument(ai);
    }
    m_out.append(1, 'E');
}

void MDL_name_mangler::mangle_template_argument(Array_instance const &ai)
{
    // <template-arg> ::= <type>              # type or template
    //                ::= X <expression> E    # expression
    //                ::= <expr-primary>      # simple expressions
    //                ::= I <template-arg>* E # argument pack
    //                ::= sp <expression>     # pack expansion of (C++0x)
    int size = ai.get_immediate_size();

    //  <expr-primary> ::= L <type> <value number> E # integer literal

    m_out.append(1, 'L');
    m_out.append(1, 'i');

    char buffer[32];
    snprintf(buffer, 32, "%u", size);
    m_out.append(buffer);

    m_out.append(1, 'E');
}

// Parse "<len><name>" from strptr and append the name to the generated string.
bool MDL_name_mangler::demangle_name(char const *&strptr, char const *endptr, string &ref_str)
{
    char *num_end = NULL;
    unsigned long len = strtoul(strptr, &num_end, 10);
    if (num_end == NULL || strptr == num_end || num_end + len > endptr)
        return false;

    bool is_punycode = *strptr == '0';
    if (is_punycode) {
        string s(num_end, num_end + len, get_allocator());

        ref_str += Punycode_encoder_swift(get_allocator()).decode(s.c_str());
    } else {
        ref_str.append(num_end, 0, len);
    }
    strptr = num_end + len;

    return true;
}

// Parse the sequence id for a substitution.
unsigned long MDL_name_mangler::parse_sequence_id(char const *&strptr, char const *endptr)
{
    // sequence id in base 64 (digits + upper case letters)
    // and _ being first entry, 0_ second.

    if (*strptr == '_') {
        ++strptr;
        return 0;
    }

    char *num_end = NULL;
    unsigned long seqid = strtoul(strptr, &num_end, 36) + 1;
    if (num_end == NULL || num_end == strptr || strptr >= endptr)
        return ~0;
    strptr = num_end;
    if (*strptr != '_')
        return ~0;
    ++strptr;
    return seqid;
}

// Demangle a C++ name to an MDL function name for use with resolvers.
bool MDL_name_mangler::demangle(char const *mangled_name, size_t len)
{
    if (len < 3 || mangled_name[0] != '_' || mangled_name[1] != 'Z')
        return false;

    char const *ptr = mangled_name + 2;
    char const *endptr = mangled_name + len;

    Memory_arena arena(m_alloc);
    Arena_vector<char const *>::Type substitutions(&arena);
    Arena_vector<bool>::Type subst_skips(&arena);

    // got qualified name?
    if (*ptr == 'N') {
        ++ptr;

        // skip any const modifier
        if (*ptr == 'K')
            ++ptr;

        string name("::", m_alloc);
        while (ptr < endptr) {
            if (!demangle_name(ptr, endptr, name))
                return false;

            // unexpected end of string?
            if (ptr >= endptr)
                return false;

            // end of qualified name?
            if (*ptr == 'E')
                break;

            // note: the fully-qualified function name is not added to the substitution table,
            //       so only add the current name prefix, if the end has not been reached
            substitutions.push_back(Arena_strdup(arena, name.c_str()));
            subst_skips.push_back(false);

            name += "::";
        }
        ++ptr;  // skip 'E'
        m_out += name;
    } else {
        if (!demangle_name(ptr, endptr, m_out))
            return false;
    }

    // for functions, handle parameter types
    if (ptr < endptr)
    {
        m_out += '(';
        if (*ptr != 'v')  // not a void function?
        {
            bool first = true;
            while (ptr < endptr)
            {
                // we need to skip pointer parameters which have been added
                // to map array returns
                bool skip_parameter = false;

                // skip reference and const qualifiers and pointers
                char const *qualifier_start = ptr;
                int num_qualifiers = 0;
                int skip_from_qualifier = 0x7fffffff;
                while (ptr < endptr && (*ptr == 'R' || *ptr == 'K' || *ptr == 'P') || *ptr == 'U') {
                    if (*ptr == 'P') {
                        skip_parameter = true;
                        skip_from_qualifier = num_qualifiers;
                    } else if (*ptr == 'U') {
                        ++ptr;
                        string qualifier(m_alloc);
                        if (!demangle_name(ptr, endptr, qualifier))
                            return false;
                        m_out += qualifier;
                        m_out += ' ';
                        --ptr;
                        skip_from_qualifier = num_qualifiers;
                    }
                    ++ptr, ++num_qualifiers;
                }

                // unexpected end of string?
                if (ptr >= endptr)
                    return false;

                string name(m_alloc);
                char const *add_subst = NULL;
                bool add_self_subst = false;

                switch (*ptr)
                {
                // handle builtin type
                #define CASE_BUILTIN(ch, type_name)  \
                    case ch:                         \
                        name = type_name;            \
                        ++ptr;                       \
                        add_subst = type_name;       \
                        break;

                CASE_BUILTIN('b', "bool");
                CASE_BUILTIN('d', "double");
                CASE_BUILTIN('f', "float");
                CASE_BUILTIN('i', "int");

                #undef CASE_BUILTIN

                case '0': case '1': case '2': case '3': case '4':
                case '5': case '6': case '7': case '8': case '9':
                {
                    // Plain name with size
                    if (!demangle_name(ptr, endptr, name))
                        return false;

                    add_subst = Arena_strdup(arena, name.c_str());
                    add_self_subst = true;
                    break;
                }

                case 'N':
                {
                    ++ptr;

                    // check for a leading substitution
                    if (*ptr == 'S') {
                        ++ptr;
                        unsigned long seqid = parse_sequence_id(ptr, endptr);
                        if (seqid == ~0 || seqid >= substitutions.size())
                            return false;
                        name += substitutions[seqid];
                    }
                    name += "::";

                    while (ptr < endptr) {
                        if (!demangle_name(ptr, endptr, name))
                            return false;

                        // unexpected end of string?
                        if (ptr >= endptr)
                            return false;

                        // end of qualified name?
                        if (*ptr == 'E')
                            break;

                        substitutions.push_back(Arena_strdup(arena, name.c_str()));
                        subst_skips.push_back(false);

                        name += "::";
                    }
                    ++ptr;  // skip 'E'

                    add_subst = Arena_strdup(arena, name.c_str());
                    add_self_subst = true;
                    break;
                }

                case 'c':
                {
                    ++ptr;

                    // map "char const *" to string
                    if (num_qualifiers == 2 &&
                        qualifier_start[0] == 'P' &&
                        qualifier_start[1] == 'K')
                    {
                        name = "string";

                        add_subst = "string";
                        add_self_subst = true;
                        skip_parameter = false;
                        break;
                    }

                    return false;
                }

                case 'S':
                {
                    // substitution
                    ++ptr;
                    unsigned long seqid = parse_sequence_id(ptr, endptr);
                    if (seqid == ~0 || seqid >= substitutions.size())
                        return false;

                    name = substitutions[seqid];
                    add_subst = substitutions[seqid];
                    skip_parameter |= subst_skips[seqid];
                    break;
                }

                case 'D':
                {
                    ++ptr;

                    // GNU extension: vector types:
                    // <type>                  ::= <vector-type>
                    // <vector-type>           ::= Dv <positive dimension number> _
                    //                                    <extended element type>
                    //                         ::= Dv [<dimension expression>] _ <element type>
                    // <extended element type> ::= <element type>
                    //                         ::= p # AltiVec vector pixel
                    //                         ::= b # AltiVec vector bool
                    if (*ptr == 'v') {
                        ++ptr;
                        char const *num_start = ptr;
                        char *num_end = NULL;

                        // make sure the dimension is a number
                        strtoul(ptr, &num_end, 10);
                        if (num_end == NULL || num_end == ptr || num_end >= endptr)
                            return false;
                        ptr = num_end;
                        if (*ptr != '_')
                            return false;
                        ++ptr;

                        switch (*ptr) {
                        case 'b': name = "bool"; ++ptr; break;
                        case 'd': name = "double"; ++ptr; break;
                        case 'f': name = "float"; ++ptr; break;
                        case 'i': name = "int"; ++ptr; break;
                        default:
                            return false;
                        }
                        name.append(num_start, num_end - num_start);
                        add_subst = Arena_strdup(arena, name.c_str());
                        add_self_subst = true;
                        break;
                    }
                    return false;
                }

                case 'u':
                {
                    ++ptr;
                    if (!demangle_name(ptr, endptr, name))
                        return false;
                    add_subst = Arena_strdup(arena, name.c_str());
                    add_self_subst = true;
                    break;
                }

                default:
                    return false;
                }

                // append name as parameter, if we shouldn't skip it
                if (!skip_parameter) {
                    if (first) first = false;
                    else m_out += ',';
                    m_out += name;
                }

                // the substitution table would normally also contain entries for all
                // qualifiers, so "RK6float3" would result in
                //   ["float3", "float3 const", "float3 const&"]
                // being added. But for MDL names we ignore the qualifiers and just
                // add the name multiple times.
                int num_add_subst = num_qualifiers + (add_self_subst ? 1 : 0);
                for (int i = 0; i < num_add_subst; ++i) {
                    substitutions.push_back(add_subst);
                    subst_skips.push_back(i >= skip_from_qualifier);
                }
            }
        }
        m_out += ')';
    }

    return true;
}


// --------------------------- DAG mangler ---------------------------

// Check if the name for the given definition must get a signature suffix.
bool DAG_mangler::need_signature_suffix(IDefinition const *def) const
{
    IDefinition::Kind kind = def->get_kind();
    if (kind == IDefinition::DK_ANNOTATION) {
        // annotations are not stored in the neuray DB
        return true;
    }
    if (IType_function const *fun_type = as<IType_function>(def->get_type())) {
        if (def->get_semantics() == IDefinition::DS_UNKNOWN) {
            // a user defined function
            IType const *ret_type = fun_type->get_return_type();

            // do not add a suffix for material creator function, these are not
            // overloaded
            return !is_material_type(ret_type);
        }
        return true;
    }
    return false;
}

// Convert a definition.
string DAG_mangler::mangle(IDefinition const *idef, const char *module_name, bool with_signature_suffix)
{
    Definition const *def = impl_cast<Definition>(idef);

    string result(get_allocator());
    string symbol_name(def->get_sym()->get_name(), get_allocator());

#if 0
    Scope const *scope = def->get_def_scope();
    while (ISymbol const *scope_name = scope->get_scope_name()) {
        // a named scope, add its name
        symbol_name = scope_name->get_name() + ("::" + symbol_name);
        scope = scope->get_parent();
    }
#endif

    if (module_name != NULL && module_name[0]) {
        if (module_name[0] != ':')
            result = "::";
        result += module_name;
        result += "::";
    }
    result += symbol_name;

    // if this entity is removed at some version, add a marker
    unsigned version_flags = def->get_version_flags();
    unsigned removed_ver = mdl_removed_version(version_flags);
    if (removed_ver != 0xFFFFFFFF) {
        // was removed at some version, we report the version *until* it exists
        switch (IMDL::MDL_version(removed_ver)) {
        case IMDL::MDL_VERSION_1_0:
            result += "$0.9";
            break;
        case IMDL::MDL_VERSION_1_1:
            result += "$1.0";
            break;
        case IMDL::MDL_VERSION_1_2:
            result += "$1.1";
            break;
        case IMDL::MDL_VERSION_1_3:
            result += "$1.2";
            break;
        case IMDL::MDL_VERSION_1_4:
            result += "$1.3";
            break;
        case IMDL::MDL_VERSION_1_5:
            result += "$1.4";
            break;
        case IMDL::MDL_VERSION_1_6:
            result += "$1.5";
            break;
        case IMDL::MDL_VERSION_1_7:
            result += "$1.6";
            break;
        }
    }

    if ((symbol_name == "operator.") || (symbol_name == "operator[]"))
        return result;
    if (with_signature_suffix && need_signature_suffix(def)) {
        result += "(";
        IType_function const *fun_type = cast<IType_function>(def->get_type());
        int count = fun_type->get_parameter_count();
        for (int i = 0; i < count; ++i) {
            if (i > 0)
                result += ",";

            IType const   *p_type = NULL;
            ISymbol const *p_sym = NULL;
            fun_type->get_parameter(i, p_type, p_sym);

            result += mangle_parameter_type(p_type);
        }
        result += ")";
    }
    return result;
}

// Mangle a definition.
string DAG_mangler::mangle(IDefinition const *def, IModule const *owner, bool with_signature_suffix)
{
    const char* module_name = NULL;
    if (owner && !owner->is_builtins())
        module_name = owner->get_name();

    return mangle(def, module_name, with_signature_suffix);
}

// Mangle a type.
string DAG_mangler::mangle(IType const *type, char const *module_name)
{
    string result(get_allocator());

    ISymbol const *sym;
    if (IType_struct const *s_type = as<IType_struct>(type)) {
        sym = s_type->get_symbol();
    } else if (IType_enum const *e_type = as<IType_enum>(type)) {
        sym = e_type->get_symbol();
    } else {
        MDL_ASSERT(!"Can only mangle user type names");
        return result;
    }

    // user defined types already have a full qualified name
    result = sym->get_name();
    return result;
}

// Mangle a type.
string DAG_mangler::mangle(IType const *type, IModule const *owner)
{
    return mangle(type, owner->is_builtins() ? NULL : owner->get_name());
}

// Mangle a parameter type.
string DAG_mangler::mangle_parameter_type(IType const *type)
{
    type = type->skip_type_alias();

    if (IType_array const *a_tp = as<IType_array>(type)) {
        IType const *e_tp = a_tp->get_element_type();
        m_printer.print(e_tp->skip_type_alias());
        m_printer.print("[");
        if (a_tp->is_immediate_sized()) {
            size_t size = a_tp->get_size();
            m_printer.print(size);
        } else {
            IType_array_size const *size = a_tp->get_deferred_size();
            m_printer.print(size->get_size_symbol());
        }
        m_printer.print("]");
    } else {
        m_printer.print(type);
    }

    return m_printer.get_line();
}

}  // mdl
}  // mi
