/***************************************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The definition of the \c Any class.

#ifndef BASE_SYSTEM_STLEXT_ANY_H
#define BASE_SYSTEM_STLEXT_ANY_H

#include <boost/any.hpp>

namespace MI {
namespace STLEXT {

/// The class \c Any is a class whose instances can hold instances of any type. Hence it allows
/// the creation of containers holding values of any types in a typesafe manner. A typical
/// example of creating such a container would be
/// \code
///   typedef std::list<STLEXT::Any> many;
///   int i=9; string txt="Text"; float f=12.5; STLEXT::Any empty;
///   many myList;
///   myList.push_back(i);
///   myList.push_back(txt);
///   myList.push_back(f);
///   myList.push_back(empty);
/// \endcode
/// The retrieval of the type information of the elements can be done by the usage of the member
/// \c Any::type(). Assuming a \c list<STLEXT::Any>::const_iterator \c it into the container \c many
/// \code
///   if (it->type() == typeid(int))
///     cout << "has type int" << endl;
/// \endcode
/// In addition to that there is the member \c Any::empty() which finds out whether the \c Any
/// instance holds a type or not
/// \code
///   Any empty_any;
///   assert(empty_any.empty());
///   assert(empty_any.type() == typeid(void));
/// \endcode
typedef ::boost::any Any;

/// The retrieval of the values in a typesafe manner works by using the \c any_cast.
/// Assuming that we have a \c list<STLEXT::Any>::const_iterator \c it into the created container
/// \c many that we can access the elements as
/// \code
///   if (STLEXT::any_cast<int>(&(*it)))
///     cout << "int " << *STLEXT::any_cast<int>(&(*it)) << endl;
/// \endcode
/// or
/// \code
///   if (it->type() == typeid(int))
///     cout << "int " << *STLEXT::any_cast<int>(&(*it)) << endl;
/// \endcode
/// Both variants are identical.
/// For a more elaborated example look at the unit test file \c test_any.cpp.
using ::boost::any_cast;

using ::boost::bad_any_cast;
}
}
#endif
