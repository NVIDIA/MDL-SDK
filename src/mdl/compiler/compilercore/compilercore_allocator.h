/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILERCORE_ALLOCATOR_H
#define MDL_COMPILERCORE_ALLOCATOR_H 1

#define USE_OWN_STRING

#include "compilercore_cc_conf.h"

#include <mi/base/iallocator.h>
#include <mi/base/handle.h>
#include <mi/base/types.h>
#include <mi/base/uuid.h>
#include <mi/base/atom.h>
#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>

#ifdef USE_OWN_STRING
#include "compilercore_string.h"
#else
#include <string>
#endif

#if MDL_STD_HAS_UNORDERED
#include <unordered_map>
#include <unordered_set>
#else
// pre C++11 need boost headers
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#endif

#include <vector>
#include <set>
#include <list>
#include <deque>
#include <queue>
#include <stack>
#include <map>
#include <typeinfo>
#include <utility>

#include "compilercore_hash_ptr.h"
#include "compilercore_assert.h"

namespace mi {
namespace mdl {

#ifndef USE_OWN_STRING

// hash functor for simple_string
template<typename String>
struct string_hash : public std::hash<String> {
};
#endif

///
/// An extended version of the mi::base::IAllocator interface with extra
/// debug support for tracking reference counted objects.
///
class IDebugAllocator : public
    mi::base::Interface_declare<0xdb3f3f76,0x0e707,0x480c,0x9b,0x08,0x2f,0x59,0x22,0xa2,0x00,0xc0,
        mi::base::IAllocator>
{
public:
    /// Allocates a memory block for a class instance.
    ///
    /// \param cls_name  the class name of the object or NULL
    /// \param size      the size of the class object
    virtual void *objalloc(char const *cls_name, Size size) = 0;

    /// Marks the given object as reference counted and set the initial count.
    ///
    /// \param obj       the object (address)
    /// \param initial   the initial reference count
    virtual void mark_ref_counted(void const *obj, Uint32 initial) = 0;

    /// Increments the reference count of an reference counted object.
    ///
    /// \param obj       the object (address)
    virtual void inc_ref_count(void const *obj) = 0;

    /// Decrements the reference count of an reference counted object.
    ///
    /// \param obj       the object (address)
    virtual void dec_ref_count(void const *obj) = 0;
};

typedef mi::base::IAllocator IAllocator;

///
/// Implementation of the allocator builder.
///
class Allocator_builder
{
public:
    explicit Allocator_builder(IAllocator *alloc)
    : m_alloc(alloc, mi::base::DUP_INTERFACE)
    {
        // MDL_ASSERT(M_COMP, alloc != NULL);
    }

    template<typename T>
    T *create() {
        if (T *p = allocate<T>()) {
            return new(p) T;
        }
        return NULL;
    }

    template<typename T, typename A1>
    T *create(A1 a1) {
        if (T *p = allocate<T>()) {
            return new(p) T(a1);
        }
        return NULL;
    }

    template<typename T, typename A1, typename A2>
    T *create(A1 a1, A2 a2) {
        if (T *p = allocate<T>()) {
            return new(p) T(a1, a2);
        }
        return NULL;
    }

    template<typename T, typename A1, typename A2, typename A3>
    T *create(A1 a1, A2 a2, A3 a3) {
        if (T *p = allocate<T>()) {
            return new(p) T(a1, a2, a3);
        }
        return NULL;
    }

    template<typename T, typename A1, typename A2, typename A3, typename A4>
    T *create(A1 a1, A2 a2, A3 a3, A4 a4) {
        if (T *p = allocate<T>()) {
            return new(p) T(a1, a2, a3, a4);
        }
        return NULL;
    }

    template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5>
    T *create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5) {
        if (T *p = allocate<T>()) {
            return new(p) T(a1, a2, a3, a4, a5);
        }
        return NULL;
    }

    template<
        typename T, typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6>
    T *create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6) {
        if (T *p = allocate<T>()) {
            return new(p) T(a1, a2, a3, a4, a5, a6);
        }
        return NULL;
    }

    template<
        typename T, typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6, typename A7>
    T *create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7) {
        if (T *p = allocate<T>()) {
            return new(p) T(a1, a2, a3, a4, a5, a6, a7);
        }
        return NULL;
    }

    template<
        typename T, typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6, typename A7,
        typename A8>
    T *create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8) {
        if (T *p = allocate<T>()) {
            return new(p) T(a1, a2, a3, a4, a5, a6, a7, a8);
        }
        return NULL;
    }

    template<
        typename T, typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6, typename A7,
        typename A8, typename A9>
    T *create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9) {
        if (T *p = allocate<T>()) {
            return new(p) T(a1, a2, a3, a4, a5, a6, a7, a8, a9);
        }
        return NULL;
    }

    template<
        typename T, typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6, typename A7,
        typename A8, typename A9, typename A10>
    T *create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10) {
        if (T *p = allocate<T>()) {
            return new(p) T(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
        }
        return NULL;
    }

    template<
        typename T, typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6, typename A7,
        typename A8, typename A9, typename A10, typename A11>
    T *create(
        A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10,
        A11 a11)
    {
        if (T *p = allocate<T>()) {
            return new(p) T(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
        }
        return NULL;
    }

    template<
        typename T, typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6, typename A7,
        typename A8, typename A9, typename A10, typename A11,
        typename A12>
    T *create(
        A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10,
        A11 a11, A12 a12)
    {
        if (T *p = allocate<T>()) {
            return new(p) T(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12);
        }
        return NULL;
    }

    template<
        typename T, typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6, typename A7,
        typename A8, typename A9, typename A10, typename A11,
        typename A12, typename A13>
    T *create(
        A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10,
        A11 a11, A12 a12, A13 a13)
    {
        if (T *p = allocate<T>()) {
            return new(p) T(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13);
        }
        return NULL;
    }

    template<
        typename T, typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6, typename A7,
        typename A8, typename A9, typename A10, typename A11,
        typename A12, typename A13, typename A14>
    T *create(
        A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10,
        A11 a11, A12 a12, A13 a13, A14 a14)
    {
        if (T *p = allocate<T>()) {
            return new(p) T(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14);
        }
        return NULL;
    }

    template<
        typename T, typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6, typename A7,
        typename A8, typename A9, typename A10, typename A11,
        typename A12, typename A13, typename A14, typename A15>
    T *create(
        A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10,
        A11 a11, A12 a12, A13 a13, A14 a14, A15 a15)
    {
        if (T *p = allocate<T>()) {
            return new(p) T(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15);
        }
        return NULL;
    }

    template<
        typename T, typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6, typename A7,
        typename A8, typename A9, typename A10, typename A11,
        typename A12, typename A13, typename A14, typename A15,
        typename A16>
    T *create(
        A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10,
        A11 a11, A12 a12, A13 a13, A14 a14, A15 a15, A16 a16)
    {
        if (T *p = allocate<T>()) {
            return new(p) T(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16);
        }
        return NULL;
    }

    template<
        typename T, typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6, typename A7,
        typename A8, typename A9, typename A10, typename A11,
        typename A12, typename A13, typename A14, typename A15,
        typename A16, typename A17>
    T *create(
        A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10,
        A11 a11, A12 a12, A13 a13, A14 a14, A15 a15, A16 a16, A17 a17)
    {
        if (T *p = allocate<T>()) {
            return new(p) T(
                a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17);
        }
        return NULL;
    }

    template<
        typename T, typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6, typename A7,
        typename A8, typename A9, typename A10, typename A11,
        typename A12, typename A13, typename A14, typename A15,
        typename A16, typename A17, typename A18>
    T *create(
        A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10,
        A11 a11, A12 a12, A13 a13, A14 a14, A15 a15, A16 a16, A17 a17, A18 a18)
    {
        if (T *p = allocate<T>()) {
            return new(p) T(
                a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18);
        }
        return NULL;
    }

    template<
        typename T, typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6, typename A7,
        typename A8, typename A9, typename A10, typename A11,
        typename A12, typename A13, typename A14, typename A15,
        typename A16, typename A17, typename A18, typename A19>
        T *create(
            A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10,
            A11 a11, A12 a12, A13 a13, A14 a14, A15 a15, A16 a16, A17 a17, A18 a18,
            A19 a19)
    {
        if (T *p = allocate<T>()) {
            return new(p) T(
                a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17,
                a18, a19);
        }
        return NULL;
    }

    /// Destroy an object
    template<typename T>
    void destroy(T const *obj) {
        obj->~T();
        m_alloc->free((void *)obj);
    }

    /// allocate size bytes.
    void *malloc(Size size) { return m_alloc->malloc(size); }

    /// allocate n elements of type T
    template<typename T>
    T *alloc(Size n_elems) { return (T *)m_alloc->malloc(sizeof(T) * n_elems); }

    /// free a memory block
    template<typename T>
    void free(T const *memory) { m_alloc->free((void *)memory); }

public:
    /// Get the allocator, not retained.
    ///
    /// \note Does NOT increase the reference count of the returned
    ///       module, do NOT decrease it just because of this call.
    inline IAllocator *get_allocator() const { return m_alloc.get(); }

protected:
    template<typename T>
    T *allocate() {
#ifdef DEBUG
        if (IDebugAllocator *dbg_alloc = m_alloc->get_interface<IDebugAllocator>()) {
#ifdef NO_RTTI
            char const *type_name = "<unknown>";
#else
            char const *type_name = typeid(T).name();
#endif
            T *p = reinterpret_cast<T*>(dbg_alloc->objalloc(type_name, sizeof(T)));
            dbg_alloc->release();
            return p;
        }
#endif
        return reinterpret_cast<T*>(m_alloc->malloc(sizeof(T)));
    }

protected:
    /// The allocator.
    mi::base::Handle<IAllocator> m_alloc;
};

/// Mixin class template for deriving interface implementations.
///
/// #mi::mdl::Allocator_interface_implement is a mixin class template that allows you to derive
/// interface class implementations easily. It provides you with the full implementation
/// of reference counting and the #mi::base::IInterface::get_interface(const Uuid&)
/// method. It requires that you used interfaces derived from the corresponding mixin 
/// class template #mi::base::Interface_declare.
///
/// #mi::mdl::Allocator_interface_implement is derived from the interface \c I.
///
/// \tparam I The interface class that this class implements.
template <class I>
class Allocator_interface_implement : public I
{
public:
    /// Constructor.
    ///
    /// \param alloc     The used allocator.
    /// \param initial   The initial reference count (defaults to 1).
    Allocator_interface_implement(IAllocator *alloc, Uint32 initial = 1)
    : m_refcnt(initial), m_alloc(alloc)
    {
        if (alloc) {
            alloc->retain();
#ifdef DEBUG
            if (IDebugAllocator *dbg_alloc = alloc->get_interface<IDebugAllocator>()) {
                dbg_alloc->mark_ref_counted(this, m_refcnt);
                dbg_alloc->release();
            }
#endif
        }
    }

    /// Copy constructor.
    ///
    /// Initializes the reference count to 1.
    Allocator_interface_implement( const Allocator_interface_implement<I>& other)
    : m_refcnt(1), m_alloc(other.m_alloc)
    {
        if (m_alloc) {
            m_alloc->retain();
#ifdef DEBUG
            if (IDebugAllocator *dbg_alloc = m_alloc->get_interface<IDebugAllocator>()) {
                dbg_alloc->mark_ref_counted(this, m_refcnt);
                dbg_alloc->release();
            }
#endif
        }
    }

    /// Get the allocator, not retained.
    ///
    /// \note Does NOT increase the reference count of the returned
    ///       module, do NOT decrease it just because of this call.
    IAllocator *get_allocator() const
    {
        return m_alloc;
    }

    /// Assignment operator.
    ///
    /// The reference count of \c *this and \p other remain unchanged.
    Allocator_interface_implement<I>& operator=( const Allocator_interface_implement<I>& other)
    {
        // Note: no call of operator= on m_refcount
        return *this; 
    }

    /// Increments the reference count.
    ///
    /// Increments the reference count of the object referenced through this interface
    /// and returns the new reference count. The operation is thread-safe.
    Uint32 retain() const MDL_OVERRIDE
    {
#ifdef DEBUG
        if (m_alloc) {
            if (IDebugAllocator *dbg_alloc = m_alloc->get_interface<IDebugAllocator>()) {
                dbg_alloc->inc_ref_count(this);
                dbg_alloc->release();
            }
        }
#endif
        return ++m_refcnt;
    }

    /// Decrements the reference count.
    ///
    /// Decrements the reference count of the object referenced through this interface
    /// and returns the new reference count. If the reference count dropped to
    /// zero, the object will be deleted. The operation is thread-safe.
    Uint32 release() const MDL_OVERRIDE
    {
#ifdef DEBUG
        if (m_alloc) {
            if (IDebugAllocator *dbg_alloc = m_alloc->get_interface<IDebugAllocator>()) {
                dbg_alloc->dec_ref_count(this);
                dbg_alloc->release();
            }
        }
#endif
        Uint32 cnt = --m_refcnt;
        if (!cnt) {
            if (IAllocator *alloc = m_alloc) {
                // we have an allocator, free there
                this->~Allocator_interface_implement();
                alloc->free((void *)this);
                alloc->release();
            } else {
                // assume it was created by new and delete it
                delete this;
            }
        }
        return cnt;
    }

    /// Acquires a const interface.
    ///
    /// If this interface is derived from or is the interface with the passed
    /// \p interface_id, then return a non-\c NULL \c const #mi::base::IInterface* that
    /// can be casted via \c static_cast to an interface pointer of the interface type
    /// corresponding to the passed \p interface_id. Otherwise return \c NULL.
    ///
    /// In the case of a non-\c NULL return value, the caller receives ownership of the 
    /// new interface pointer, whose reference count has been retained once. The caller 
    /// must release the returned interface pointer at the end to prevent a memory leak.
    mi::base::IInterface const *get_interface(
        mi::base::Uuid const &interface_id) const MDL_OVERRIDE
    {
        return I::get_interface_static(this, interface_id);
    }

    /// Acquires a mutable interface.
    ///
    /// If this interface is derived from or is the interface with the passed
    /// \p interface_id, then return a non-\c NULL #mi::base::IInterface* that
    /// can be casted via \c static_cast to an interface pointer of the interface type
    /// corresponding to the passed \p interface_id. Otherwise return \c NULL.
    ///
    /// In the case of a non-\c NULL return value, the caller receives ownership of the 
    /// new interface pointer, whose reference count has been retained once. The caller 
    /// must release the returned interface pointer at the end to prevent a memory leak.
    mi::base::IInterface *get_interface(
        mi::base::Uuid const &interface_id) MDL_OVERRIDE
    {
        return I::get_interface_static(this, interface_id);
    }

    /// Returns the interface ID of the most derived interface.
    mi::base::Uuid get_iid() const MDL_OVERRIDE
    {
        return typename I::IID();
    }

protected:
    virtual ~Allocator_interface_implement() {}

private:
    mutable mi::base::Atom32 m_refcnt;
    IAllocator               *m_alloc;
};


/// A standards-compliant allocator using an IAllocator.
template<typename T>
class Mi_allocator
{
public:
    typedef size_t      size_type;
    typedef ptrdiff_t   difference_type;
    typedef T           value_type;
    typedef T*          pointer;
    typedef T const*    const_pointer;
    typedef T&          reference;
    typedef T const&    const_reference;

    template<typename U> 
    struct rebind { typedef Mi_allocator<U> other; };

    /// Creates an STL allocator from an IAllocator.
    /// Non-explicit to simplify use.
    /*implicit*/ Mi_allocator(IAllocator *alloc) : m_alloc(alloc, mi::base::DUP_INTERFACE)
    {
    }

#ifdef MI_ENABLE_MISTD
    /// Default constructor, required by the native STL (and destroys the whole concept).
    Mi_allocator()
    : m_alloc()
    {
        // best is to abort here ...
    }
#endif

    /// Copy constructor.
    Mi_allocator(Mi_allocator<T> const &other)
    : m_alloc(other.m_alloc)
    {
    }

#if MDL_RVALUE_REFERENCES
    /// Move constructor.
    Mi_allocator(Mi_allocator<T> &&other)
    : m_alloc(other.m_alloc)
    {
        // Do NOT move the allocator here. Doing this will make the deallocate() function fail.
        // Instead, make a real copy. This is legal, as a move constructor must leave the object
        // in a valid state for destruction, which is given. However, at least the MSVC STL
        // will use a "moved" allocator to destruct existing objects which will fail.
    }

    /// Assignment operator.
    Mi_allocator<T> &operator=(Mi_allocator<T> const &other)
    {
        m_alloc = other.m_alloc;
        return *this;
    }

    /// Move assignment operator.
    Mi_allocator<T> &operator=(Mi_allocator<T> &&other)
    {
        // Do NOT move the allocator here. Doing this will make the deallocate() function fail.
        // Instead, make a real assignment. This is legal, as a move assignment must leave the
        // object in a valid state for destruction, which is given. However, at least the MSVC STL
        // will use a "moved" allocator to destruct existing objects which will fail.
        m_alloc = other.m_alloc;
        return *this;
    }
#endif

    /// Creates an STL allocator to the argument's IAllocator.
    template<typename U>
    Mi_allocator(Mi_allocator<U> const& arg) : m_alloc(arg.m_alloc)
    {
    }

    /// The largest value that can meaningfully passed to allocate.
    size_type max_size() const { return 0xffffffff; }

    /// Memory is allocated for \c count objects of type \c T but objects are not constructed.
    pointer allocate( size_type count, std::allocator<void>::const_pointer /*hint*/ = 0 ) const
    {
#ifdef DEBUG
        if (count == 1) {
            if (IDebugAllocator *dbg_alloc = m_alloc->get_interface<IDebugAllocator>()) {
#ifdef NO_RTTI
                char const *type_name = "<unknown>";
#else
                char const *type_name = typeid(T).name();
#endif
                T *p = reinterpret_cast<T*>(dbg_alloc->objalloc(type_name, sizeof(T)));

                dbg_alloc->release();
                return p;
            }
        }
#endif
        return reinterpret_cast<T*>(m_alloc->malloc(count * sizeof(T)));
    }

    /// Deallocates memory allocated by allocate.
    void deallocate(pointer block, size_type count) const throw()
    {
        m_alloc->free(reinterpret_cast<void *>(block));
    }

    /// Constructs an element of \c T at the given pointer.
    void construct(pointer element, T const& arg)
    {
        new (element) T(arg);
    }

#ifdef MI_ENABLE_MISTD
    template <typename U, typename... Args>
    void construct(U *p, Args&&... args) 
    {
        ::new((void *)p) U(std::forward<Args>(args)...);
    }
#endif

    /// Destroys an element of \c T at the given pointer.
    void destroy(pointer element)
    {
        element->~T();
    }

    /// Returns the address of the given reference.
    pointer address(reference element) const
    {
        return &element;
    }

    /// Returns the address of the given reference.
    const_pointer address(const_reference element) const
    {
        return &element;
    }

    /// The memory arena for this allocator.
    mi::base::Handle<IAllocator> m_alloc;
};

/// A specialization of the pooled allocator for the void type.
template<>
class Mi_allocator<void>
{
public:
    typedef size_t      size_type;
    typedef ptrdiff_t   difference_type;

    typedef void        value_type;
    typedef void*       pointer;
    typedef void const* const_pointer;

    template<typename U> 
    struct rebind { typedef Mi_allocator<U> other; };

    /// Creates a pooled allocator to the given pool.
    /// Native STL requires default constructor to exist.
    Mi_allocator(IAllocator *alloc = 0) : m_alloc(alloc, mi::base::DUP_INTERFACE) {}

    /// The memory arena for this allocator.
    mi::base::Handle<IAllocator> m_alloc;
};

/// Returns true if objects allocated from one memory arena can be deallocated from the other.
template<typename T, typename U>
bool operator==(Mi_allocator<T> const& left, Mi_allocator<U> const& right)
{
    return left.m_alloc == right.m_alloc;
}

/// Returns true if objects allocated from one memory arena cannot be deallocated from the other.
template<typename T, typename U>
bool operator!=(Mi_allocator<T> const& left, Mi_allocator<U> const& right)
{
    return left.m_alloc != right.m_alloc;
}

/// A VLA using an allocator.
/// \tparam T  element type, must be a POD.
/// \tparam A  the allocator type
template<typename T, typename A = IAllocator>
class VLA {
public:
    typedef T       *iterator;
    typedef T const *const_iterator;

    /// Construct a new VLA of size n, allocated with the allocator alloc.
    VLA(A *alloc, size_t n)
    : m_alloc(alloc)
    , m_data(reinterpret_cast<T*>(alloc->malloc(sizeof(T) * n)))
    , m_size(n)
    {
    }

    /// Free the VLA.
    ~VLA() { m_alloc->free(m_data); }

    /// Access the VLA array by index.
    T &operator[](size_t index) {
        MDL_ASSERT(index < m_size && "index out of bounds");
        return m_data[index];
    }

    /// Access the VLA array by index.
    T const &operator[](size_t index) const {
        MDL_ASSERT(index < m_size && "index out of bounds");
        return m_data[index];
    }

    /// Access the VLA array.
    T *data() { return m_data; }

    /// Access the VLA array.
    T const *data() const { return m_data; }

    /// Return the size of the VLA.
    size_t size() const { return m_size; }

    /// Get the allocator.
    A *get_allocator() const { return m_alloc; }

    /// Return iterator to beginning.
    iterator begin() { return data(); }

    /// Return iterator to beginning.
    const_iterator begin() const { return data(); }

    /// Return iterator to end.
    iterator end() { return data() + m_size; }

    /// Return iterator to end.
    const_iterator end() const { return data() + m_size; }

private:
    A * const    m_alloc;
    T * const    m_data;
    size_t const m_size;
};

/// A VLA using space inside the object or using an allocator if the requested size is too large.
/// \tparam T  element type, must be a POD.
/// \tparam N  preallocated size
/// \tparam A  the allocator type
template<typename T, size_t N, typename A = IAllocator>
class Small_VLA {
public:
    typedef T       *iterator;
    typedef T const *const_iterator;

    /// Construct a new VLA of size n, using the internal space or allocated with the
    /// allocator alloc if the size is too large.
    Small_VLA(A *alloc, size_t n)
    : m_alloc(alloc)
    , m_data(NULL)
    , m_size(n)
    {
        if (m_size > N) {
            m_data = reinterpret_cast<T*>(m_alloc->malloc(sizeof(T) * m_size));
        } else {
            m_data = m_static;
        }
    }

    /// Free the VLA.
    ~Small_VLA()
    {
        if (m_data != m_static)
            m_alloc->free(m_data);
    }

    /// Access the VLA array by index.
    T &operator[](size_t index) {
        MDL_ASSERT(index < m_size && "index out of bounds");
        return m_data[index];
    }

    /// Access the VLA array by index.
    T const &operator[](size_t index) const {
        MDL_ASSERT(index < m_size && "index out of bounds");
        return m_data[index];
    }

    /// Access the VLA array.
    T *data() { return m_data; }

    /// Access the VLA array.
    T const *data() const { return m_data; }

    /// Return the size of the VLA.
    size_t size() const { return m_size; }

    /// Get the allocator.
    A *get_allocator() { return m_alloc; }

    /// Return iterator to beginning.
    iterator begin() { return data(); }

    /// Return iterator to beginning.
    const_iterator begin() const { return data(); }

    /// Return iterator to end.
    iterator end() { return data() + m_size; }

    /// Return iterator to end.
    const_iterator end() const { return data() + m_size; }

private:
    A            *m_alloc;
    T            *m_data;
    T             m_static[N];
    size_t const  m_size;
};

/// Returns the dimension of an array.
template<typename T, typename A>
inline size_t dimension_of(VLA<T,A> const &vla) { return vla.size(); }

template<typename T, typename A>
inline size_t dimension_of(std::vector<T,A> const &vec) { return vec.size(); }

#ifdef USE_OWN_STRING
/// A string using a IAllocator.
typedef simple_string<
    char,
    char_traits<char>,
    Mi_allocator<char> > string;

/// A wide string using a IAllocator.
typedef simple_string<
    wchar_t,
    char_traits<wchar_t>,
    Mi_allocator<wchar_t> > wstring;

/// A u32 string using a IAllocator.
typedef simple_string<
    unsigned,
    char_traits<unsigned>,
    Mi_allocator<unsigned> > u32string;

#else
/// A string using a IAllocator.
typedef std::basic_string<
    char,
    std::char_traits<char>,
    Mi_allocator<char> > string;

/// A wide string using a IAllocator.
typedef std::basic_string<
    wchar_t,
    std::char_traits<wchar_t>,
    Mi_allocator<wchar_t> > wstring;

/// A u32 string using a IAllocator.
typedef std::basic_string<
    unsigned,
    std::char_traits<unsigned>,
    Mi_allocator<unsigned> > u32string;

#endif

inline bool has_dynamic_memory_consumption(string const &) { return true; }
inline size_t dynamic_memory_consumption (const string& s)
{
    // we may have a local storage optimization in the string class,
    // i.e. short strings are stored inside the string object (no heap allocation)
    // try to detect this case, if we have the string stored inside the object,
    // then the address must be in the range [&s, &s + sizeof(s)]
    if (!s.empty()) {
        string::value_type const* ps = &s[0];
        string::value_type const* pobj =
            reinterpret_cast<string::value_type const*>(
            static_cast<const void*>(&s));
        if (ps < pobj || ps >= (pobj + sizeof(string))) {
            return s.capacity() + 1;  // + trailing '\0'
        }
    }
    return 0;
}

/// A vector using IAllocator.
template<typename T>
struct vector
{
    typedef std::vector<T, Mi_allocator<T> > Type;
};

/// A hashmap from Key * to Tp, using a IAllocator.
#if MDL_STD_HAS_UNORDERED
template <
    typename Key,
    typename Tp,
    typename HashFcn = Hash_ptr<Key>,
    typename EqualKey = Equal_ptr<Key>
>
struct ptr_hash_map {
    typedef std::unordered_map<
        Key *, Tp, HashFcn, EqualKey,
        Mi_allocator<typename std::unordered_map<Key *, Tp, HashFcn, EqualKey>::value_type>
    > Type;
};
#else
template <
    typename Key,
    typename Tp,
    typename HashFcn = Hash_ptr<Key>,
    typename EqualKey = Equal_ptr<Key>
>
struct ptr_hash_map {
    typedef boost::unordered_map<
        Key *, Tp, HashFcn, EqualKey, 
        Mi_allocator<typename boost::unordered_map<Key *, Tp, HashFcn, EqualKey>::value_type>
    > Type;
};
#endif

/// A hash functor for handles.
template <typename T>
class Hash_handle {
public:
    size_t operator()(base::Handle<T> const &p) const {
        size_t t = p.get() - (T const *)0;
        return ((t) / (sizeof(size_t) * 2)) ^ (t >> 16);
    }
};

/// An Equal functor for pointers.
template <typename T>
struct Equal_handle {
    inline unsigned operator()(base::Handle<T> const &a, base::Handle<T> const &b) const {
        return a == b;
    }
};

/// A hashmap from base::Handle<Key> to Tp, using a IAllocator.
#if MDL_STD_HAS_UNORDERED
template <
    typename Key,
    typename Tp,
    typename HashFcn = Hash_handle<Key>,
    typename EqualKey = Equal_handle<Key>
>
struct handle_hash_map {
    typedef std::unordered_map<
        base::Handle<Key>, Tp, HashFcn, EqualKey,
        Mi_allocator<
            typename std::unordered_map<base::Handle<Key>, Tp, HashFcn, EqualKey>::value_type>
    > Type;
};
#else
template <
    typename Key,
    typename Tp,
    typename HashFcn = Hash_handle<Key>,
    typename EqualKey = Equal_handle<Key>
>
struct handle_hash_map {
    typedef boost::unordered_map<
        base::Handle<Key>, Tp, HashFcn, EqualKey,
        Mi_allocator<
            typename boost::unordered_map<base::Handle<Key>, Tp, HashFcn, EqualKey>::value_type>
    > Type;
};
#endif

/// A hashmap using a IAllocator.
#if MDL_STD_HAS_UNORDERED
template <
    typename Key,
    typename Tp,
    typename HashFcn = std::hash<Key>,
    typename EqualKey = std::equal_to<Key>
>
struct hash_map {
    typedef std::unordered_map<
        Key, Tp, HashFcn, EqualKey,
        Mi_allocator<typename std::unordered_map<Key, Tp, HashFcn, EqualKey>::value_type>
    > Type;
};
#else
template <
    typename Key,
    typename Tp,
    typename HashFcn = boost::hash<Key>,
    typename EqualKey = std::equal_to<Key>
>
struct hash_map {
    typedef boost::unordered_map<
        Key, Tp, HashFcn, EqualKey, 
        Mi_allocator<typename boost::unordered_map<Key, Tp, HashFcn, EqualKey>::value_type>
    > Type;
};
#endif

/// A hash set using a IAllocator.
#if MDL_STD_HAS_UNORDERED
template <
    typename Key,
    typename HashFcn = std::hash<Key>,
    typename EqualKey = std::equal_to<Key>
>
struct hash_set {
    typedef std::unordered_set<
        Key, HashFcn, EqualKey, Mi_allocator<Key> > Type;
};
#else
template <
    typename Key,
    typename HashFcn = boost::hash<Key>,
    typename EqualKey = std::equal_to<Key>
>
struct hash_set {
    typedef boost::unordered_set<
        Key, HashFcn, EqualKey, Mi_allocator<Key> > Type;
};
#endif

/// A hash set of pointers using a IAllocator.
#if MDL_STD_HAS_UNORDERED
template <
    typename Key,
    typename HashFcn = Hash_ptr<Key>,
    typename EqualKey = Equal_ptr<Key>
>
struct ptr_hash_set {
    typedef std::unordered_set<
        Key *, HashFcn, EqualKey, Mi_allocator<Key *> > Type;
};
#else
template <
    typename Key,
    typename HashFcn = Hash_ptr<Key>,
    typename EqualKey = Equal_ptr<Key>
>
struct ptr_hash_set {
    typedef boost::unordered_set<
        Key *, HashFcn, EqualKey, Mi_allocator<Key *> > Type;
};
#endif

/// A hash set of base::Handle<Key> using a IAllocator.
#if MDL_STD_HAS_UNORDERED
template <
    typename Key,
    typename HashFcn = Hash_handle<Key>,
    typename EqualKey = Equal_handle<Key>
>
struct handle_hash_set {
    typedef std::unordered_set<
        base::Handle<Key>, HashFcn, EqualKey, Mi_allocator<base::Handle<Key> > > Type;
};
#else
template <
    typename Key,
    typename HashFcn = Hash_handle<Key>,
    typename EqualKey = Equal_handle<Key>
>
struct handle_hash_set {
    typedef boost::unordered_set<
        base::Handle<Key>, HashFcn, EqualKey, Mi_allocator<base::Handle<Key> > > Type;
};
#endif

/// A set using a IAllocator.
template <
    typename Key,
    typename Compare = std::less<Key>
>
struct set {
    typedef std::set<Key, Compare, Mi_allocator<Key> > Type;
};

/// A list using a IAllocator.
template <typename Tp>
struct list {
    typedef std::list<Tp, Mi_allocator<Tp> > Type;
};

/// A deque using a IAllocator.
template <typename Tp>
struct deque {
    typedef std::deque<Tp, Mi_allocator<Tp> > Type;
};

/// A queue using a IAllocator.
template <
    typename Tp,
    typename Sequence = std::deque<Tp, Mi_allocator<Tp> >
>
struct queue {
    typedef std::queue<Tp, Sequence> Type;
};

/// A stack using a IAllocator.
template <
    typename Tp,
    typename Sequence = std::deque<Tp, Mi_allocator<Tp> >
>
struct stack {
    typedef std::stack<Tp, Sequence> Type;
};

/// A map using a IAllocator.
template<
    typename Key,
    typename Tp,
    typename Compare = std::less<Key>
>
struct map {
    typedef std::map<
        Key, Tp, Compare, Mi_allocator<std::pair<const Key, Tp> > > Type;
};

/// A map from Key * to Tp, using a IAllocator.
template <
    typename Key,
    typename Tp,
    typename LessFcn = std::less<Key *>
>
struct ptr_map {
    typedef std::map<
        Key *, Tp, LessFcn,
        Mi_allocator<typename std::map<Key *, Tp, LessFcn>::value_type>
    > Type;
};

}  // mdl
}  // mi

#endif // MDL_COMPILERCORE_ALLOCATOR_H
