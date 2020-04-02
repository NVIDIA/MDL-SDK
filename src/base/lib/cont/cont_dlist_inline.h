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

namespace MI {

namespace CONT {

// Constructor.
template<class T,Dlist_link<T> T::*link>
inline Dlist_intr<T,link>::Dlist_intr()
{
    m_first = 0;
    m_last = 0;
    m_count = 0;
}

// Destructor.
template<class T,Dlist_link<T> T::*link>
Dlist_intr<T,link>::~Dlist_intr()
{}

// Constructor.
// list is the list to iterate over.
template<class T,Dlist_link<T> T::*link>
inline Dlist_intr<T,link>::Iterator::Iterator(
    const Dlist_intr<T,link> *list)	// The list to iterate over.
  : m_list(list)
  , m_current(NULL)
{}

// Destructor.
template<class T,Dlist_link<T> T::*link>
Dlist_intr<T,link>::Iterator::~Iterator()
{}

// Set iterator to the first element of the list if it exists.
template<class T,Dlist_link<T> T::*link>
inline void Dlist_intr<T,link>::Iterator::to_first()
{
    m_current = m_list->first();
}

// Set iterator to the last element of the list if it exists.
template<class T,Dlist_link<T> T::*link>
inline void Dlist_intr<T,link>::Iterator::to_last()
{
    m_current = m_list->last();
}

// Set iterator to the next element of the list if it exists.
template<class T,Dlist_link<T> T::*link>
inline void Dlist_intr<T,link>::Iterator::to_next()
{
    if(!m_current)
        return;
    m_current = m_list->next(m_current);
}

// Set iterator to the previous element of the list if it exists.
template<class T,Dlist_link<T> T::*link>
inline void Dlist_intr<T,link>::Iterator::to_previous()
{
    if(!m_current)
        return;
    m_current = m_list->previous(m_current);
}

// Set the iterator to element of the list.
template<class T,Dlist_link<T> T::*link>
inline void Dlist_intr<T,link>::Iterator::to_element(
    T *element)				// The element to set the iterator to.
{
    ASSERT(M_CONT,(element->*link).m_cont == this);
    m_current = element;
}

// Return true if iterator is exhausted, false otherwise.
template<class T,Dlist_link<T> T::*link>
inline bool Dlist_intr<T,link>::Iterator::at_end() const
{
    return !m_current;
}

// Return a reference to the current element.
template<class T,Dlist_link<T> T::*link>
inline T &Dlist_intr<T,link>::Iterator::operator*() const
{
    return *m_current;
}

// Apply member selection to the current element.
template<class T,Dlist_link<T> T::*link>
inline T *Dlist_intr<T,link>::Iterator::operator->() const
{
    return m_current;
}

// Return count of elements in list.
template<class T,Dlist_link<T> T::*link>
inline size_t Dlist_intr<T,link>::count() const
{
    return m_count;
}

// Return first element of list if it exists,
// NULL otherwise.
template<class T,Dlist_link<T> T::*link>
inline T *Dlist_intr<T,link>::first() const
{
    return m_first;
}

// Return last element of list if it exists,
// NULL otherwise.
template<class T,Dlist_link<T> T::*link>
inline T *Dlist_intr<T,link>::last() const
{
    return m_last;
}

// Return the successor of an element in the list if it exists,
// NULL otherwise.
template<class T,Dlist_link<T> T::*link>
inline T *Dlist_intr<T,link>::next(
    T *element) const			// The element of which to return
					// the successor.
{
    ASSERT(M_CONT,(element->*link).m_cont == this);
    return (element->*link).m_next;
}

// Return the predecessor of an element in the list if it exists,
// NULL otherwise.
template<class T,Dlist_link<T> T::*link>
inline T *Dlist_intr<T,link>::previous(
    T *element) const			// The element of which to return
					// the predecessor.
{
    ASSERT(M_CONT,(element->*link).m_cont == this);
    return (element->*link).m_previous;
}

// Append an element to the list.
template<class T,Dlist_link<T> T::*link>
inline void Dlist_intr<T,link>::append(
    T *element)				// The element to append.
{
    (element->*link).m_next = 0;
#ifdef ENABLE_ASSERT
    ASSERT(M_CONT,!(element->*link).m_cont);
    (element->*link).m_cont = this;
#endif
    if(m_last)
        (m_last->*link).m_next = element;
    else
        m_first = element;
    (element->*link).m_previous = m_last;
    m_last = element;
    ++m_count;
}

// Prepend an element to the list.
template<class T,Dlist_link<T> T::*link>
inline void Dlist_intr<T,link>::prepend(
    T *element)				// The element to prepend.
{
    (element->*link).m_previous = 0;
#ifdef ENABLE_ASSERT
    ASSERT(M_CONT,!(element->*link).m_cont);
    (element->*link).m_cont = this;
#endif
    if(m_first)
        (m_first->*link).m_previous = element;
    else
        m_last = element;
    (element->*link).m_next = m_first;
    m_first = element;
    ++m_count;
}

// Insert a before b.
template<class T,Dlist_link<T> T::*link>
inline void Dlist_intr<T,link>::insert_before(
    T *a,				// The element to insert.
    T *b)				// The element to insert before.
{
#ifdef ENABLE_ASSERT
    ASSERT(M_CONT,(b->*link).m_cont == this);
    ASSERT(M_CONT,!(a->*link).m_cont);
    (a->*link).m_cont = this;
#endif
    (a->*link).m_next = b;
    (a->*link).m_previous = (b->*link).m_previous;
    if((a->*link).m_previous)
	(((a->*link).m_previous)->*link).m_next = a;
    (b->*link).m_previous = a;
    if(m_first == b)
        m_first = a;
    ++m_count;
}

// Insert a after b.
template<class T,Dlist_link<T> T::*link>
inline void Dlist_intr<T,link>::insert_after(
    T *a,				// The element to insert.
    T *b)				// The element to insert after.
{
#ifdef ENABLE_ASSERT
    ASSERT(M_CONT,(b->*link).m_cont == this);
    ASSERT(M_CONT,!(a->*link).m_cont);
    (a->*link).m_cont = this;
#endif
    (a->*link).m_previous = b;
    (a->*link).m_next = (b->*link).m_next;
    if((a->*link).m_next)
        (((a->*link).m_next)->*link).m_previous = a;
    (b->*link).m_next = a;
    if(m_last == b)
        m_last = a;
    ++m_count;
}

// Remove first element of list.
// Return the element if it exists, and NULL otherwise.
template<class T,Dlist_link<T> T::*link>
inline T *Dlist_intr<T,link>::remove_first()
{
    if(!m_first)
        return 0;
    T *tmp = m_first;
    m_first = (tmp->*link).m_next;
    if(m_first)
        (m_first->*link).m_previous = 0;
    else
        m_last = 0;
#ifdef ENABLE_ASSERT
    (tmp->*link).m_next = 0;
    (tmp->*link).m_previous = 0;
    (tmp->*link).m_cont = 0;
#endif
    --m_count;
    return tmp;
}

// Remove last element of list.
// Return the element if it exists, and NULL otherwise.
template<class T,Dlist_link<T> T::*link>
inline T *Dlist_intr<T,link>::remove_last()
{
    if(!m_last)
        return 0;
    T *tmp = m_last;
    m_last = (tmp->*link).m_previous;
    if(m_last)
        (m_last->*link).m_next = 0;
    else
        m_first = 0;
#ifdef ENABLE_ASSERT
    (tmp->*link).m_next = 0;
    (tmp->*link).m_previous = 0;
    (tmp->*link).m_cont = 0;
#endif
    --m_count;
    return tmp;
}

// Remove an element from the list.
// Removing an element from a list that is not part of the list
// results in a fatal error!
template<class T,Dlist_link<T> T::*link>
inline void Dlist_intr<T,link>::remove(
    T *element)				// The element to remove.
{
    ASSERT(M_CONT,(element->*link).m_cont == this);
    if(element == m_first) {
        m_first = (element->*link).m_next;
        if(m_first)
            (m_first->*link).m_previous = 0;
    } else {
        ((element->*link).m_previous->*link).m_next =
            (element->*link).m_next;
    }
    if(element == m_last) {
        m_last = (element->*link).m_previous;
        if(m_last)
            (m_last->*link).m_next = 0;
    } else {
        ((element->*link).m_next->*link).m_previous =
            (element->*link).m_previous;
    }
#ifdef ENABLE_ASSERT
    (element->*link).m_previous = 0;
    (element->*link).m_next = 0;
    (element->*link).m_cont = 0;
#endif
    --m_count;
}

// Check consistency.
template<class T,Dlist_link<T> T::*link>
inline void Dlist_intr<T,link>::check_consistency()
{
    T *p;

    for(p = m_first; p && (p->*link).m_next; p = (p->*link).m_next) {
        ASSERT(M_CONT,(p->*link).m_cont == this);
        ASSERT(M_CONT,p == ((p->*link).m_next->*link).m_previous);
    }
    ASSERT(M_CONT,p == m_last);
#ifdef ENABLE_ASSERT
    if(p) {
        ASSERT(M_CONT,(p->*link).m_cont == this);
    }
#endif
}

// Constructor.
template<class T>
inline Dlist<T>::Dlist()
{}

// Copy constructor.
template<class T>
inline Dlist<T>::Dlist(
    const Dlist<T> &list)		// The list to copy.
{
    Iterator i(&list);

    for(i.to_first(); !i.at_end(); i.to_next())
        append(*i);
}

// Copy assignment.
template<class T>
inline Dlist<T> &Dlist<T>::operator=(
    const Dlist<T> &list)		// The list to assign.
{
    if(this != &list)
	return *this;
    Iterator i(&list);

    for(i.to_first(); !i.at_end(); i.to_next())
        append(*i);
    return *this;
}

// Destructor.
template<class T>
Dlist<T>::~Dlist()
{
    clear();
}

// Constructor.
// list is the list to iterate over.
template<class T>
inline Dlist<T>::Iterator::Iterator(
    const Dlist<T> *list)		// The list to iterate over.
  : m_list(list)
  , m_current(NULL)
{}

// Destructor.
template<class T>
Dlist<T>::Iterator::~Iterator()
{}

// Set iterator to the first element of the list if it exists.
template<class T>
inline void Dlist<T>::Iterator::to_first()
{
    m_current = m_list->m_list.first();
}

// Set iterator to the last element of the list if it exists.
template<class T>
inline void Dlist<T>::Iterator::to_last()
{
    m_current = m_list->m_list.last();
}

// Set iterator to the next element of the list if it exists.
template<class T>
inline void Dlist<T>::Iterator::to_next()
{
    if(!m_current)
        return;
    m_current = m_list->m_list.next(m_current);
}

// Set iterator to the previous element of the list if it exists.
template<class T>
inline void Dlist<T>::Iterator::to_previous()
{
    if(!m_current)
        return;
    m_current = m_list->m_list.previous(m_current);
}

// Return true if iterator is exhausted, false otherwise.
template<class T>
inline bool Dlist<T>::Iterator::at_end() const
{
    return !m_current;
}

// Return a reference to the current element.
template<class T>
inline T& Dlist<T>::Iterator::operator*() const
{
    return m_current->m_value;
}

// Apply member selection to the current element.
template<class T>
inline T* Dlist<T>::Iterator::operator->() const
{
    return &(operator*());
}

// Return the count of elements in list.
template<class T>
inline size_t Dlist<T>::count() const
{
    return m_list.count();
}

// Return pointer to the first element of the list if it exists,
// NULL otherwise.
template<class T>
inline T *Dlist<T>::first() const
{
    Node *n;

    if((n = m_list.first())) {
        return &n->m_value;
    } else {
        return 0;
    }
}

// Return a copy of the first element of the list in f.
// Return true if it the element exists, and false otherwise.
// f is only set when the return value is true.
template<class T>
inline bool Dlist<T>::first(
    T *f) const				// Pointer to first element.
{
    Node *n;

    if((n = m_list.first())) {
        *f = n->m_value;
        return true;
    } else {
        return false;
    }
}

// Return pointer to the last element of the list if it exists,
// NULL otherwise.
template<class T>
inline T *Dlist<T>::last() const
{
    Node *n;

    if((n = m_list.last())) {
        return &n->m_value;
    } else {
        return 0;
    }
}

// Return a copy of the last element of the list in l.
// Return true if it the element exists, and false otherwise.
// l is only set when the return value is true.
template<class T>
inline bool Dlist<T>::last(
    T *l) const				// Pointer to last element.
{
    Node *n;

    if((n = m_list.last())) {
        *l = n->m_value;
        return true;
    } else {
        return false;
    }
}

// Append element to list.
template<class T>
inline void Dlist<T>::append(
    const T &element)			// The element to append.
{
    m_list.append(new Node(element));
}

// Prepend element to list.
template<class T>
inline void Dlist<T>::prepend(
    const T &element)			// The element to prepend.
{
    m_list.prepend(new Node(element));
}

// Insert a before current position of it.
template<class T>
void Dlist<T>::insert_before(
    T *element,				// The element to insert.
    const Iterator &it)			// The iterator giving the position
					// at which to insert.
{
    m_list.insert_before(new Node(element),it.m_current);
}

// Insert a after current position of it.
template<class T>
void Dlist<T>::insert_after(
    T *element,				// The element to insert.
    const Iterator &it)			// The iterator giving the position
					// at which to insert.
{
    m_list.insert_after(new Node(element),it.m_current);
}

// Remove last element of list and return it in l.
// Return true if it the element exists, and false otherwise.
// l is only set when the return value is true.
template<class T>
inline bool Dlist<T>::remove_first(
    T *f)				// Pointer to first element.
{
    Node *n;

    if(0 != (n = m_list.remove_first())) {
        *f = n->m_value;
        delete n;
        return true;
    } else {
        return false;
    }
}

// Remove last element of list and return it in l.
// Return true if it the element exists, and false otherwise.
// l is only set when the return value is true.
template<class T>
inline bool Dlist<T>::remove_last(
    T *l)				// Pointer to last element.
{
    Node *n;

    if((n = m_list.remove_last())) {
        *l = n->m_value;
        delete n;
        return true;
    } else {
        return false;
    }
}

// Remove an element from the list.
template<class T>
inline void Dlist<T>::remove(
    const T &element)			// The element to remove.
{
    for(Node *n = m_list.first(); n; n = m_list.next(n))
        if(n->m_value == element) {
            m_list.remove(n);
            delete n;
            return;
        }
}

// Remove an element from the list.
template<class T>
inline void Dlist<T>::remove_all(
    const T &element)			// The element to remove.
{
    for(Node *n = m_list.first(); n;)
        if(n->m_value == element) {
            Node *tmp = m_list.next(n);
            m_list.remove(n);
            delete n;
            n = tmp;
        } else {
            n = m_list.next(n);
        }
}

// Remove element from list at iterator position and advance iterator.
template<class T>
void Dlist<T>::remove(
    Iterator &it)			// The iterator giving the position
					// of the element to remove.
{
    Node *c = it.m_current;

    it.to_next();
    m_list.remove(c);
    delete c;
}

// Clear all elements from the list.
template<class T>
inline void Dlist<T>::clear()
{
    Node *n;

    while((n = m_list.remove_first()))
        delete n;
}

// Check consistency.
template<class T>
inline void Dlist<T>::check_consistency()
{
    m_list.check_consistency();
}

}

}
