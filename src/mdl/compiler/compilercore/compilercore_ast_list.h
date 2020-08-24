/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILERCORE_AST_LIST_H
#define MDL_COMPILERCORE_AST_LIST_H 1

#include <cstddef>
#include "compilercore_cc_conf.h"

namespace mi {
namespace mdl {

/// The iterator class for our intrinsic lists.
template <typename T>
class Ast_list_iterator {
    typedef Ast_list_iterator<T> Self;
public:
    Ast_list_iterator &operator++() {
        if (this->m_ptr != NULL) this->m_ptr = m_ptr->m_next;
        return *this;
    }
    T &operator*() { return *this->m_ptr; }
    T *operator->() { return this->m_ptr; }
    T const &operator*() const { return *this->m_ptr; }
    T const *operator->() const { return this->m_ptr; }
    operator T *() { return this->m_ptr; }
    operator T const *() const { return this->m_ptr; }

public:
    /// Default constructor, constructs the end iterator.
    Ast_list_iterator() :m_ptr(NULL)
    {
    }

    /// Constructor from a pointer.
    explicit Ast_list_iterator(T *ptr) : m_ptr(ptr)
    {}

    bool operator==(Self const &o) const { return this->m_ptr == o.m_ptr; }
    bool operator!=(Self const &o) const { return !operator==(o); }

private:
    /// The pointer.
    T *m_ptr;
};

/// A list of AST entities.
template <typename T>
class Ast_list : public Interface_owned
{
    typedef Interface_owned Base;
public:
    typedef Ast_list_iterator<T>       iterator;
    typedef Ast_list_iterator<T const> const_iterator;

    /// Add an element at the list's end.
    ///
    /// \param id  the element to be added
    void push(T *id) {
        id->m_prev = m_last;
        if (m_last == NULL) { m_first = id; }
        else                { m_last->m_next = id; }
        m_last     = id;
        id->m_next = NULL;
    }

    /// Add an element at the list's begin.
    ///
    /// \param id  the element to be added
    void push_front(T *id) {
        id->m_prev = NULL;
        id->m_next = m_first;
        m_first    = id;
        if (m_last == NULL) { m_last = id; }
    }

    /// Get the first layout qualifier id.
    iterator begin() { return iterator(m_first); }

    /// Get the end iterator.
    iterator end() { return iterator(); }

    /// Get the first layout qualifier id.
    const_iterator begin() const { return const_iterator(m_first); }

    /// Get the end iterator.
    const_iterator end() const { return const_iterator(); }

    /// Check if the list is empty.
    bool empty() const { return m_first == NULL; }

    /// Return the length of this list.
    size_t size() const {
        size_t l = 0;
        for (const_iterator it(this->begin()), end(this->end()); it != end; ++it)
            ++l;
        return l;
    }

    /// Return true if there is exactly one element in the list.
    bool single_element() const {
        size_t l = 0;
        for (const_iterator it(this->begin()), end(this->end()); it != end && l < 2; ++it)
            ++l;
        return l == 1;
    }

    /// Access first element.
    T *front() { return m_first; }

    /// Access first element.
    T const *front() const { return m_first; }

    /// Access last element.
    T *back() { return m_last; }

    /// Access last element.
    T const *back() const { return m_last; }

    /// Delete first element.
    void pop_front() {
        if (m_first != NULL) {
            T *e = m_first;
            m_first = e->m_next;

            if (m_first != NULL) {
                m_first->m_prev = NULL;
            } else {
                m_last = NULL;
            }
            e->m_next = e->m_prev = NULL;
        }
    }

    /// Delete last element.
    void pop_back() {
        if (m_last != NULL) {
            T *e   = m_last;
            m_last = e->m_prev;

            if (m_last != NULL) {
                m_last->m_next = NULL;
            } else {
                m_first = NULL;
            }
            e->m_next = e->m_prev = NULL;
        }
    }

    /// Removes from the list a single element.
    iterator erase(iterator pos) {
        iterator res = pos;

        if (m_first == pos) {
            m_first = pos->m_next;
            if (m_first == NULL) {
                m_last = NULL;
            }
        } else {
            pos->m_prev->m_next = pos->m_next;
        }

        if (m_last == pos) {
            m_last = pos->m_prev;
            if (m_last == NULL) {
                m_first = NULL;
            }
        } else {
            pos->m_next->m_prev = pos->m_prev;
        }
        pos->m_prev= pos->m_next= NULL;

        return ++res;
    }

public:
    /// Default constructor, constructs an empty list.
    Ast_list()
    : Base()
    , m_first(NULL)
    , m_last(NULL)
    {
    }

private:
    // non copyable
    Ast_list(Ast_list const &) MDL_DELETED_FUNCTION;
    Ast_list &operator=(Ast_list const &) MDL_DELETED_FUNCTION;

private:
    /// Points to the first list element.
    T *m_first;

    /// Points to the last list element.
    T *m_last;
};

/// A list element.
template <typename T>
class Ast_list_element : public Interface_owned {
    typedef Interface_owned Base;
    friend class Ast_list<T>;
    friend class Ast_list_iterator<T>;
    friend class Ast_list_iterator<T const>;
public:
    /// Constructor.
    Ast_list_element()
    : m_prev(NULL)
    , m_next(NULL)
    {}

private:
    /// Pointer to the previous element in the list.
    T *m_prev;

    /// Pointer to the next element in the list.
    T *m_next;
};

}  // mdl
}  // mi

#endif
