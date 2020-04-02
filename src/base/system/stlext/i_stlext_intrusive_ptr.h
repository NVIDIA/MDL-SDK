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
/// \brief An intrusive, reference-counting smart pointer.

#ifndef BASE_SYSTEM_STLEXT_INTRUSIVE_PTR_H
#define BASE_SYSTEM_STLEXT_INTRUSIVE_PTR_H

#include <algorithm>        // std::swap()
#include "i_stlext_concepts.h"                  // Abstract_interface
#include "i_stlext_atomic_counter.h"            // Atomic_counter

namespace MI { namespace STLEXT {

/**
 * \brief Standard base class for reference counted objects.
 *
 * Derive from this class to trivially create an object suitable for use with
 * \c Intrusive_ptr.
 */

class Ref_counted : protected Abstract_interface
{
    Atomic_counter      m_refcount;

public:
    /// \brief Upon construction, a reference counted object has a refcount of 1.
    Ref_counted() : m_refcount(1u)  { }

    /// \brief Query the current reference count.
    size_t ref_count()       { return m_refcount; }

    /// \brief Increment the reference count.
    void pin()    { ++m_refcount; }

    /// \brief Decrement the reference count and return \c true if the object
    /// was destroyed.
    bool unpin()  { return (--m_refcount == 0u) ? delete this, true : false; }

    //@{
    /// \brief Free function required by \c Intrusive_ptr.
    friend inline void increment_ref_count(Ref_counted * obj)
    {
        obj->pin();
    }
    friend inline void decrement_ref_count_and_release(Ref_counted * obj)
    {
        obj->unpin();
    }
    //@}
};

/**
 * \brief Tag value to pass to \c Intrusive_ptr's constructor.
 *
 * Reference counted objects are typically constructed with a reference count
 * of 1. Intrusive pointers increment that count on construction and decrement
 * it on destruction. Consequently, this means that the reference count would
 * never actually reach zero -- the object would never be destroyed! For the
 * proper semantics, exactly one \c Intrusive_ptr must take ownership by \em
 * not incrementing the reference count at construction. This is accomplished
 * by passing this tag value to the constructor.
 */

enum take_ownership_tag { take_ownership };

/**
 * \brief An intrusive, reference-counting smart pointer.
 *
 * Reference-counting smart pointers such as \c boost::shared_ptr by nature introduce
 * a certain amount of run-time overhead compared to an ordinary pointer. One
 * particularly expensive operation is creating a shared pointer, because it
 * requires the creation of a "shared object state" on the heap.
 *
 * In cases where this overhead is too expensive, \c Intrusive_ptr might be an
 * alternative. Unlike \c boost::shared_ptr, intrusive pointers cannot store arbitrary
 * types \c T, because \c Intrusive_ptr requires the \em object to provide the
 * reference counter. That reference counter is accessed through the following
 * a free-function API:
 *
 * <pre>
 *   void increment_ref_count(Ref_counted * obj);
 *   void decrement_ref_count_and_release(Ref_counted * obj);
 * <post>
 *
 * Because of the use of free functions, objects stored in an \c intrusive
 * poiter don't need to be derived from a particular base class. Nonetheless,
 * this module provides the generic base class \c Ref_counted to make use of
 * intrusive pointers more convenient. Anyhow, other classes might be useful,
 * i.e. to implement other types of reference counters.
 *
 * How an object stored in \c Intrusive_ptr is going to be deleted is
 * determined by the \c decrement_ref_count_and_release() function. In other
 * words, the object's \em type determines the deleter. As a consequence,
 * intrusive pointers don't support casting, like boost::shared_ptr does.
 *
 * On the other hand, intrusive pointers have a capability that shared pointers
 * do not: since the reference count resides in the object, it is possible to
 * promote an ordinary \c T* to \c Intrusive_ptr<T>.
 *
 * \note Managing object with \c Intrusive_ptr that do not have a virtual
 *       destructor is almost certainly a mistake.
 */

template <class T>
class Intrusive_ptr
{
public:
    /// \brief Expose our parameter type \c T.
    typedef T value_type;

    /// \brief An ordinary pointer to \c T.
    typedef value_type * pointer;

    /// \brief An ordinary pointer to constant \c T.
    typedef value_type const * const_pointer;

    /// \brief A reference to \c T values.
    typedef value_type & reference;

    /// \brief A reference to constant \c T values.
    typedef value_type const & const_reference;

    /// \brief Construct an intrusive smart pointer from a raw pointer.
    explicit Intrusive_ptr(pointer ptr = 0) : m_ptr(ptr)
    {
      if (m_ptr) increment_ref_count(m_ptr);
    }

    /// \brief Construct an intrusive smart pointer from a raw pointer without
    /// incrementing the reference count.
    Intrusive_ptr(pointer ptr, take_ownership_tag) : m_ptr(ptr) { }

    /// \brief Copy-construct from another instance.
    Intrusive_ptr(Intrusive_ptr const & ptr) : m_ptr(ptr.m_ptr)
    {
        if (m_ptr) increment_ref_count(m_ptr);
    }

    /// \brief Decrement the reference count and delete the pointee if it
    /// reaches zero.
    ~Intrusive_ptr() { if (m_ptr) decrement_ref_count_and_release(m_ptr); }

    /// \brief Assign this instance with another object.
    Intrusive_ptr & operator= (Intrusive_ptr const & rhs)
    {
        Intrusive_ptr<T>(rhs).swap(*this);
        return *this;
    }

    /// \brief Test whether an \c Intrusive_ptr is valid, i.e. not the null pointer.
    bool is_valid() const { return m_ptr != 0; }

private:
    typedef bool (Intrusive_ptr::*unspecified_bool_type)() const;

public:
    /// \brief Like a normal pointer, \c Intrusive_ptr can be used as a boolean.
    ///
    /// An \c Intrusive_ptr object compares to true when \c
    /// Intrusive_ptr::is_valid() returns true, i.e. when the pointer is not
    /// null.
    operator unspecified_bool_type() const { return is_valid() ? &Intrusive_ptr<T>::is_valid : 0; }

    /// \brief Reset an \c Intrusive_ptr to a new object.
    void swap(Intrusive_ptr & other) { std::swap(m_ptr, other.m_ptr); }

    /// \brief Reset an existing object, dropping any former contents.
    void reset(pointer ptr = 0) { Intrusive_ptr(ptr).swap(*this); }

    /// \brief Overloaded version of \c reset() that takes ownership of the class.
    void reset(pointer ptr, take_ownership_tag) { Intrusive_ptr(ptr, take_ownership).swap(*this); }

    /// \brief Dereference the pointer.
    //@{
    reference operator *  () const { return *m_ptr; }
    pointer   operator -> () const { return m_ptr; }
    //@}

    /// \brief Access the underlying \c T* pointer.
    ///
    /// Using ordinary pointers in the presence of an intrusive pointer is
    /// potentially dangerous and should be avoided.
    pointer get() const { return m_ptr; }

    //@{
    /// \brief Compare two intrusive pointers.
    bool operator== (Intrusive_ptr const & rhs) const { return m_ptr == rhs.m_ptr; }
    bool operator!= (Intrusive_ptr const & rhs) const { return m_ptr != rhs.m_ptr; }
    bool operator<  (Intrusive_ptr const & rhs) const { return m_ptr <  rhs.m_ptr; }
    bool operator<= (Intrusive_ptr const & rhs) const { return m_ptr <= rhs.m_ptr; }
    bool operator>  (Intrusive_ptr const & rhs) const { return m_ptr >  rhs.m_ptr; }
    bool operator>= (Intrusive_ptr const & rhs) const { return m_ptr >= rhs.m_ptr; }
    //@}

private:
    pointer m_ptr;
};

}} // MI::STLEXT

namespace std
{
    /// \brief Overload standard \c std::swap().
    template<class T>
    inline void swap(MI::STLEXT::Intrusive_ptr<T> & a, MI::STLEXT::Intrusive_ptr<T> & b)
    {
        a.swap(b);
    }
}

#endif // BASE_SYSTEM_STLEXT_INTRUSIVE_PTR_H
