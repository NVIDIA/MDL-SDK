/******************************************************************************
 * Copyright (c) 2006-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Useful base classes to express common C++ concepts.
///
///    - Non_copyable          Derive from this class to make an object non-copyable.
///
///    - Abstract_interface    Derive from this class to make sure your
///                            interface classes have a virtual destructor.
///
///    - Derived_from<T, B>    Ensure that type T is derived from B.
///
///    The class "noncopyable" is a useful little helper class which defines a
///    private copy constructor and assignment operator. Thus, derived classes
///    cannot be copied.
///
///    The class "Abstract_interface" is supposed to be a common base class
///    for abstract interfaces; especially classes with pure virtual
///    functions.

#ifndef BASE_SYSTEM_STLEXT_CONCEPTS_H
#define BASE_SYSTEM_STLEXT_CONCEPTS_H

#include "i_stlext_no_unused_variable_warning.h"

namespace MI { namespace STLEXT {

// Derive from this class to make your class non-copyable. The idea is that
// the declaration
//
//   class Foo : private Non_copyable
//   {
//     ...
//   };
//
// is more readable than the equivalent
//
//   class Foo
//   {
//   public:
//     ...
//   private:
//     Foo(Foo const &);
//     Foo & operator= (Foo const &);
//   };
//
// would be, particularly for long and complex classes. It makes no difference
// whether you derive publicly or privately from "Non_copyable". Private
// inheritance should be preferred, though, because "Non_copyable" is not
// supposed to be act as a public base class.

class Non_copyable
{
private:
    Non_copyable(Non_copyable const &);
    Non_copyable& operator= (Non_copyable const &);

protected:
    Non_copyable()  { }
    ~Non_copyable() { }
};


// Classes that have virtual functions should usually have a virtual
// destructor too, and some C++ compilers produce warnings when your classes
// don't follow this guideline. To avoid trouble with unexpectedly non-virtual
// destructors (and to make the compiler shut up), simply derive from
// "Abstract_interface".
//
// The class is non-copyable because the generated default copy semantics
// practically never work for classes with virtual functions. If you want to
// pass your class by value nonetheless for some reason, define an explicit
// copy constructor or assignment operator for it; don't rely on the defaults.

struct Abstract_interface : private Non_copyable
{
    virtual ~Abstract_interface() { }
};


// Check whether type T is really derived from B. Actually, Derived_from
// doesn't check derivation, but conversion, but that's often a better
// constaint.

template<class T, class B>
struct Derived_from
{
    static void constraints(T* p) { B* pb = p; no_unused_variable_warning_please(pb); }
    Derived_from() { void(*p)(T*) = constraints; no_unused_variable_warning_please(p); }
};

}} // MI::STLEXT

#endif  // BASE_SYSTEM_STLEXT_CONCEPTS_H
