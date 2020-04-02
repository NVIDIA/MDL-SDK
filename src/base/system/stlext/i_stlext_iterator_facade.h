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
/// \brief The definition of the iterator facade.

#ifndef BASE_SYSTEM_STLEXT_ITERATOR_FACADE_H
#define BASE_SYSTEM_STLEXT_ITERATOR_FACADE_H

#include <base/system/stlext/i_stlext_type_traits.h>

namespace MI
{
namespace STLEXT
{
namespace detail
{

/// Generate the associated types for an iterator_facade with the given parameters. Currently
/// there is only the removal of const from ValueParam taking place, the rest is disabled.
template <
    typename ValueParam,
    typename CategoryOrTraversal,
    typename Reference,
    typename Difference>
struct iterator_facade_types
{
    // map given CategoryOrTraversal onto boost's new iterator types - unsupported
    /*
    typedef typename facade_iterator_category<
        CategoryOrTraversal, ValueParam, Reference
    >::type iterator_category;
    */

    /// \p remove_const from the ValueParam. Reasoning: The C++ standard requires an iterator's
    /// \p value_type not be const-qualified, so \p iterator_facade strips the const from its
    /// \p Value parameter in order to produce the iterator's \p value_type. How? Simply by
    /// using this very type_traits function.
    typedef typename remove_const<ValueParam>::type value_type;

    // True iff the user has explicitly disabled writability of this iterator. Pass the
    // iterator_facade's Value parameter and its nested ::reference type.
    // Currently unsupported.
    /*
    typedef typename mpl::eval_if<
        detail::iterator_writability_disabled<ValueParam,Reference>,
        add_pointer<const value_type>,
        add_pointer<value_type>
    >::type pointer;
    */
};

}


//==================================================================================================

/// Helper class for granting access to the iterator core interface.
/// The simple core interface is used by iterator_facade. The core interface of a user/library
/// defined iterator type should not be made public so that it does not clutter the public
/// interface. Instead iterator_core_access should be made friend so that iterator_facade can
/// access the core interface through iterator_core_access.
class iterator_core_access
{
public:
    /// Provide access to the underlying type of \p f.
    /// \return reference to the underlying type
    template <class Facade>
    static typename Facade::reference dereference(
        Facade & f)                                     ///< the facade we are using
    {
        return f.dereference();
    }
    /// Set iterator \p f to the next element.
    template <class Facade>
    static void increment(
        Facade& f)                                      ///< the facade we are using
    {
        f.increment();
    }
    /// Set iterator \p f to the previous element.
    template <class Facade>
    static void decrement(
        Facade& f)                                      ///< the facade we are using
    {
        f.decrement();
    }
    /// Compare two iterators \p f1 and \p f2 for equality.
    /// \return true if \p f1 is equals to \p f2 or false otherwise.
    template <class Facade1, class Facade2>
    static bool equal(Facade1 const& f1, Facade2 const& f2) {
        return f1.equal(f2);
    }
    /// Proceed iterator \p f by \p n.
    template <class Facade>
    static void advance(Facade& f, typename Facade::difference_type n) {
        f.advance(n);
    }
    /// Compute distance between iterator \p f by \p n.
    /// \return Computed distance.
    template <class Facade1, class Facade2>
    static typename Facade1::difference_type distance_from(const Facade1& f1, const Facade2& f2) {
        return -f1.distance_to(f2);
    }

  private:
    /// Make creation undefined. Since objects of this class are useless noone should create one.
    iterator_core_access();
};


//==================================================================================================

/// The base class. Instantiate this template class to get a facade and inherit from it to
/// offer even more convenient wrapping. Eg the iterator adaptors offer such convenience.
/// Writing standard-conforming iterators is tricky, but the need comes up often. In order to
/// ease the implementation of new iterators, this file provides the iterator_facade class template,
/// which implements many useful defaults and compile-time checks.
/// Requirement: static_cast<Derived*>(iterator_facade*) shall be well-formed.
template <
    typename Derived,
    typename Value,
    typename CategoryOrTraversal,
    typename Reference,
    typename Difference = ptrdiff_t>
struct iterator_facade
{
private:
    /// Get access to the "derived class". Curiously Recurring Template interface at work.
    Derived& derived() { return *static_cast<Derived*>(this); }
    /// Get const access to the "base class". Curiously Recurring Template interface at work.
    Derived const& derived() const { return *static_cast<Derived const*>(this); }

    /// Correct associated type. Eg, remove const from \p Value if required.
    typedef detail::iterator_facade_types<
        Value, CategoryOrTraversal, Reference, Difference
    > associated_types;

public:
    typedef typename associated_types::value_type value_type;
    typedef Reference reference;
    typedef Difference difference_type;
/*
    // currently unsupported
    typedef typename associated_types::pointer pointer;
    typedef typename associated_types::iterator_category iterator_category;
*/
    typedef Value* pointer;
    typedef const Value* const_pointer;
    typedef CategoryOrTraversal iterator_category;

    /// Retrieve access to underlying type. This relies on \p iterator_core_access definition.
    /// \return reference to underlying type
    reference operator*() const {
        return iterator_core_access::dereference(this->derived());
    }

    /// Retrieve access to underlying type.
    /// \return pointer to underlying type
    const_pointer operator->() const {
        return &this->operator*();
    }

    /// Progress to next element. This relies on \p iterator_core_access definitions.
    /// \return reference to "derived class"
    Derived& operator++() {
        iterator_core_access::increment(this->derived());
        return this->derived();
    }

    /// Progress to next element. This relies on \p iterator_core_access definitions.
    /// \return reference to "derived class"
    Derived& operator--() {
        iterator_core_access::decrement(this->derived());
        return this->derived();
    }

    /// Progress to next element. This relies on \p iterator_core_access definitions.
    /// \return reference to "derived class"
    Derived& operator+=(difference_type n) {
        iterator_core_access::advance(this->derived(), n);
        return this->derived();
    }

    /// Progress to next element. This relies on \p iterator_core_access definitions.
    /// \return reference to "derived class"
    Derived& operator-=(difference_type n) {
        iterator_core_access::advance(this->derived(), -n);
        return this->derived();
    }

    /// Compare with another iterator \p other. This relies on \p iterator_core_access definitions.
    /// \return true if \p this and \p other are equal, otherwise false
    bool operator==(const iterator_facade& other) const {
        return iterator_core_access::equal(this->derived(), (&other)->derived());
    }
    /// Compare with another iterator \p other. This relies on \p iterator_core_access definitions.
    /// \return true if \p this and \p other are not equal, otherwise false
    bool operator!=(const iterator_facade& other) const {
        return !iterator_core_access::equal(this->derived(), (&other)->derived());
    }
};

}
}

#endif
