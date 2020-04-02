/******************************************************************************
 * Copyright (c) 2004-2020, NVIDIA CORPORATION. All rights reserved.
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
 /// \brief Implements doubly-linked lists.
 ///
 ///        Double linked lists implement sequences of varying length.
 ///        They are symmetric in the sense that removing any element
 ///        has the same time complexity.
 ///        The main advantage of doubly linked lists over singly linked lists
 ///        is that all removals can be performed in constant time.
 ///
 ///        This is an example usage of non-intrusive doubly linked lists:
 ///
 ///        void example_dlist(void)
 ///        {
 ///            using namespace MI::CONT;
 ///
 ///            typedef Dlist<int> Dlist_int;
 ///
 ///            Dlist_int list;
 ///        
 ///            for(int n = 0; n < 10; n++)
 ///               list.append(n);
 ///
 ///            Dlist_int::Iterator i(&list);
 ///
 ///            for(i.to_last(); !i.at_end(); i.to_previous())
 ///               printf("%d\n",*i);
 ///        }

#ifndef BASE_LIB_CONT_DLIST_H
#define BASE_LIB_CONT_DLIST_H

#include <base/lib/log/i_log_assert.h>
#include <base/lib/mem/i_mem_allocatable.h>

namespace MI {

namespace CONT {

// Link pointers that need to be embedded in every element that can be
// part of this list.
template<
    class T>    // Type of list elements.
class Dlist_link : public MEM::Allocatable {

public:

    // Link pointers should be private, and Dlist should be a friend class.
    // Unfortunately, this is not possible because Dlist is a template
    // that requires a Dlist_link as a parameter, and therefore can not
    // be declared a friend before the definition of Dlist_link is complete.

    // Pointer to next element.
    T *m_next;

#ifdef ENABLE_ASSERT
    // Pointer to list;
    void *m_cont;
#endif

    // Pointer to previous element.
    T *m_previous;

    // Constructor.
    Dlist_link()
    {
        m_next = 0;
        m_previous = 0;
#ifdef ENABLE_ASSERT
        m_cont = 0;
#endif
    }

    // Destructor.
    ~Dlist_link()
    {
        // Currently neuray cannot shutdown correctly and some lists
        // might not be cleared and then we hit this assertion.
        // Look in dbnr_scope.cpp:~Info_container(). Scopes are registering
        // their infos but when the database is shutdown the scopes might not
        // have remove the infos from the list.
        // Set_link does not have this assert either.. Look at the comment
        // in cont_tree.h
        // ASSERT(M_CONT,m_cont == 0);
    }

private:

    // Copy constructor.
    Dlist_link(
                const Dlist_link&);	// The link to copy.

public:

    // Copy assignment.
    //
    // Dummy implementation to satisfy requirements of std::vector. It was not needed with
    // STLport, let's hope it is only required to exist, but not used with the native STL.
    Dlist_link &operator=(
                const Dlist_link&)	// The link to assign.
    { ASSERT(M_CONT, false); return *this; }

};

// Intrusive list, where links are embedded in nodes carrying the data.
 template<class T,Dlist_link<T> T::*link> class Dlist_intr : public MEM::Allocatable
{

public:

    // Constructor.
    Dlist_intr();

    // Destructor.
    ~Dlist_intr();

    //=========================================================================
    // Iterator.
    class Iterator : public MEM::Allocatable
    {

    public:

        // Constructor.
        // list is the list to iterate over.
        Iterator(
                const Dlist_intr<T,link> *list); // The list to iterate over.

        // Destructor.
        ~Iterator();

        // Set iterator to the first element of the list if it exists.
        void to_first();

        // Set iterator to the last element of the list if it exists.
        void to_last();

        // Set iterator to the next element of the list if it exists.
        void to_next();

        // Set iterator to the previous element of the list if it exists.
        void to_previous();

        // Set iterator to element of the list.
        void to_element(
                        T *element);    // The element to set the iterator to.

        // Return true if iterator is exhausted, false otherwise.
        bool at_end() const;

        // Return a reference to the current element.
        T &operator*() const;

        // Apply member selection to the current element.
        T *operator->() const;

    private:

        // Pointer to the list to iterate over.
        const Dlist_intr<T,link> *m_list;

        // Pointer to the current node in the list.
        T *m_current;

    };
    //=========================================================================

    // Return count of elements in list.
    size_t count() const;

    // Return first element of list if it exists,
    // NULL otherwise.
    T *first() const;

    // Return last element of list if it exists,
    // NULL otherwise.
    T *last() const;

    // Return the successor of an element in the list if it exists,
    // NULL otherwise.
    T *next(
            T *element) const;  // The element of which to return
                                // the successor.

    // Return the predecessor of an element in the list if it exists,
    // NULL otherwise.
    T *previous(
            T *element) const;  // The element of which to return
                                // the predecessor.

    // Append an element to the list.
    void append(
            T *element);        // The element to append.

    // Prepend an element to the list.
    void prepend(
            T *element);        // The element to prepend.

    // Insert a before b.
    void insert_before(
            T *a,                // The element to insert.
            T *b);               // The element before which to insert.

    // Insert a after b.
    void insert_after(
            T *a,                // The element to insert.
            T *b);               // The element after which to insert.

    // Remove first element of list.
    // Return the element if it exists, and NULL otherwise.
    T *remove_first();

    // Remove last element of list.
    // Return the element if it exists, and NULL otherwise.
    T *remove_last();

    // Remove an element from the list.
    // Removing an element from a list that is not part of the list
    // results in a fatal error!
    void remove(
            T *element);        // The element to remove.

    // Check consistency.
    void check_consistency();

private:

    // Pointer to first element.
    T *m_first;

    // Pointer to last element.
    T *m_last;

    // Number of elements in list.
    size_t m_count;

    // Copy constructor.
    Dlist_intr(
                const Dlist_intr&);	// The link to copy.

    // Copy assignment.
    Dlist_intr &operator=(
                const Dlist_intr&);	// The link to assign.

};

// Non-intrusive list, where nodes contain pointers to data.
template<
    class T>    // Type of list elements.
class Dlist : public MEM::Allocatable {

public:

    class Iterator;

private:

    //=========================================================================
    // Node class for Dlist.
    class Node : public MEM::Allocatable {

        friend class Dlist<T>;
        friend class Iterator;

    private:

        // Link fields.
        Dlist_link<Node> m_link;

        // Data.
        T m_value;

        // Constructor.
        Node(
                T element)      // The initial node value.
        {
            m_value = element;
        }

        // Copy constructor.
        Node(
                const Node&);	// The node to copy.

        // Copy assignment.
        Node &operator=(
                const Node&);	// The node to assign.

    };
    //=========================================================================

    // The intrusive list implementing the non-intrusive list.
    Dlist_intr<Node,&Node::m_link> m_list;

public:

    // Constructor.
    Dlist();

    // Copy constructor.
    Dlist(
            const Dlist<T> &list);      // The list to copy.

    // Copy assignment.
    Dlist &operator=(
            const Dlist<T> &list);      // The list to assign.

    // Destructor.
    ~Dlist();

    //=========================================================================
    // Iterator.
    class Iterator : public MEM::Allocatable {

        friend class Dlist<T>;

    public:

        // Constructor.
        // list is the list to iterate over.
        Iterator(
                 const Dlist<T> *list); // The list to iterate over.

        // Destructor.
        ~Iterator();

        // Set iterator to the first element of the list if it exists.
        void to_first();

        // Set iterator to the last element of the list if it exists.
        void to_last();

        // Set iterator to the next element of the list if it exists.
        void to_next();

        // Set iterator to the previous element of the list if it exists.
        void to_previous();

        // Return true if iterator is exhausted, false otherwise.
        bool at_end() const;

        // Return a reference to the current element.
        T& operator*() const;

        // Apply member selection to the current element.
        T* operator->() const;

    private:

        // Pointer to the list to iterate over.
        const Dlist<T> *m_list;

        // Pointer to the current node in the list.
        typename Dlist<T>::Node *m_current;

    };
    //=========================================================================

    // Return the count of elements in list.
    size_t count() const;

    // Return pointer to the first element of the list if it exists,
    // NULL otherwise.
    T *first() const;

    // Return the first element of the list in f.
    // Return true if it the element exists, and false otherwise.
    // f is only set when the return value is true. Note that f must not be 0!
    bool first(
		T *f) const;	// Buffer will contain a copy of first element

    // Return pointer to the last element of the list if it exists,
    // NULL otherwise.
    T *last() const;

    // Return the last element of the list in l.
    // Return true if it the element exists, and false otherwise.
    // l is only set when the return value is true. Note that l must not be 0!
    bool last(
                T *l) const;    // Buffer will contain a copy of last element

    // Append element to list.
    void append(
                const T &element);     // The element to append.

    // Prepend element to list.
    void prepend(
                const T &element);     // The element to prepend.

    // Insert a before current position of it.
    void insert_before(
            T *a,                // The element to insert.
            const Iterator &it); // The iterator giving the position
                                 // at which to insert.

    // Insert a after current position of it.
    void insert_after(
            T *a,                // The element to insert.
            const Iterator &it); // The iterator giving the position
                                 // at which to insert.

    // Remove first element of list and return it in f.
    // Return true if it the element exists, and false otherwise.
    // f is only set when the return value is true.
    bool remove_first(
                T *f);          // Pointer to the first element.

    // Remove last element of list and return it in l.
    // Return true if it the element exists, and false otherwise.
    // l is only set when the return value is true.
    bool remove_last(
                T *l);          // Pointer to the last element.

    // Remove an element from the list.
    void remove(
                const T &element);     // The element to remove.

    // Remove all occurrences of element from the list.
    void remove_all(
                const T &element);     // The element to remove.

    // Remove element from list at iterator position and advance iterator.
    void remove(
                Iterator &it);  // The iterator giving the position
                                // of the element to remove.

    // Clear all elements from the list.
    void clear();

    // Check consistency.
    void check_consistency();
};

}

}

#include "cont_dlist_inline.h"

#endif
