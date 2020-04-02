/***************************************************************************************************
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
 **************************************************************************************************/
/// \file
/// \brief poor man's mpl

#ifndef BASE_SYSTEM_STLEXT_MPL_H
#define BASE_SYSTEM_STLEXT_MPL_H

namespace MI
{
namespace STLEXT
{

// For safety I'd decided to put everything in its own namespace such that it does not interfere.
namespace mpl
{

//--------------------------------------------------------------------------------------------------

// dunno
template <typename T, T N>
struct integral_c
{
    static const T value = N;
// agurt, 08/mar/03: SGI MIPSpro C++ workaround, have to #ifdef because some
// other compilers (e.g. MSVC) are not particulary happy about it
#if MI_BOOST_WORKAROUND_FOR_EDG_VERSION_less_than_or_equal_238
    typedef struct integral_c type;
#else
    typedef integral_c type;
#endif
    typedef T value_type;
    //typedef integral_c_tag tag;  // <-- probably not used in our code?

// have to #ifdef here: some compilers don't like the 'N + 1' form (MSVC),
// while some other don't like 'value + 1' (Borland), and some don't like
// either
#if MI_BOOST_WORKAROUND_FOR_EDG_VERSION_less_than_or_equal_243
 private:
    const static T next_value = static_cast<T>(N + 1);
    const static T prior_value = static_cast<T>(N - 1);
 public:
    typedef integral_c< T, next_value > next;
    typedef integral_c<T, prior_value > prior;
/* another one
#elif MI_BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x561)) \
    || MI_BOOST_WORKAROUND(__IBMCPP__, BOOST_TESTED_AT(502)) \
    || MI_BOOST_WORKAROUND(__HP_aCC, BOOST_TESTED_AT(53800))
    typedef integral_c< T, static_cast<T>(N + 1) > next;
    typedef integral_c< T, static_cast<T>(N - 1) > prior;
*/
#else
    typedef integral_c< T, static_cast<T>(value + 1) > next;
    typedef integral_c< T, static_cast<T>(value - 1) > prior;
#endif

    // enables uniform function call syntax for families of overloaded
    // functions that return objects of both arithmetic ('int', 'long',
    // 'double', etc.) and wrapped integral types (for an example, see
    // "mpl/example/power.cpp")
    operator T() const { return static_cast<T>(this->value); }
};

// Since we haven't defined this yet it should always be initialized here.
#if !defined(MI_BOOST_NO_INCLASS_MEMBER_INITIALIZATION)
template<typename T, T N>
T const integral_c<T, N>::value;
#endif

// 'bool' constant doesn't have 'next'/'prior' members
template< bool C >
struct integral_c<bool, C>
{
    const static bool value = C;
    //typedef integral_c_tag tag;  // <-- probably not used in our code?
    typedef integral_c type;
    typedef bool value_type;
    operator bool() const { return this->value; }
};


// //-----------------------------------------------------------------------------------------------

// // The if_c helper.
// template<bool C, typename T1, typename T2>
// struct if_c
// {
//     typedef T1 type;
// };

// // The if_c helper specialization.
// template<typename T1, typename T2>
// struct if_c<false, T1, T2>
// {
//     typedef T2 type;
// };

// // The if_ type.
// template<typename T1, typename T2,typename T3>
// struct if_
// {
//  private:
//     typedef if_c<static_cast<bool>(T1::value), T2, T3> almost_type_;
//  public:
//     typedef typename almost_type_::type type;
// };

//==================================================================================================

/// The boolean function of OR-concatenating its parameters. Helper struct to get the result of
/// an OR-concatenation of all template parameters. This is the declaration with default values.
template <bool b1, bool b2, bool b3 = false, bool b4 = false, bool b5 = false, bool b6 = false>
struct bool_func_or;

/// The boolean function of OR-concatenating its parameters. Helper struct to get the result of
/// an OR-concatenation of all template parameters. This is the default implementation.
template <bool b1, bool b2, bool b3, bool b4, bool b5, bool b6>
struct bool_func_or
{
    const static bool value = true;
};

/// The boolean function of OR-concatenating its parameters. Helper struct to get the result of
/// an OR-concatenation of all template parameters. This is the specialization when all parameters
/// are \c false.
template <>
struct bool_func_or<false, false, false, false, false, false>
{
    const static bool value = false;
};

}

}
}

#endif
