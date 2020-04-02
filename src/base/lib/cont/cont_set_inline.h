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
 /// \brief Implements sets.
 ///
 ///        Sets implement ordered collections of items.
 ///        Class Set is parametrized by a node type, a compare function,
 ///        and, in the case of intrusive sets, a pointer to member giving
 ///        the location of the link fields in the elements.
 ///
 ///        A set can never contain two elements that compare equal.
 ///        Inserting an element into a set that already contains an element
 ///        that compares to equal with the element to be inserted will result
 ///        in a fatal error.
 ///
 ///        Maps are implemented as splay trees, see:
 ///        Mark Allen Weiss,
 ///        Data Structures and Algorithm Analysis
 ///        Benjamin/Cummings, 1995

#include <base/lib/log/i_log_assert.h>
#include <algorithm>  // swap

namespace MI {

namespace CONT {

// Constructor.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline Set_intr<T,compare,link>::Set_intr()
{
    m_root = 0;
    m_count = 0;
}

// Destructor.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
Set_intr<T,compare,link>::~Set_intr()
{}

// Constructor.
// set is the set to iterate over.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline Set_intr<T,compare,link>::Iterator::Iterator(
            const Set_intr<T,compare,link> *set)    // The set to iterate over.
    : m_map(set)
    , m_current(NULL)
{}

// Destructor.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
Set_intr<T,compare,link>::Iterator::~Iterator()
{}

// Set iterator to the first element of the set if it exists.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline void Set_intr<T,compare,link>::Iterator::to_first()
{
    m_current = m_map->first();
}

// Set iterator to the last element of the set if it exists.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline void Set_intr<T,compare,link>::Iterator::to_last()
{
    m_current = m_map->last();
}

// Set iterator to the next element of the set if it exists.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline void Set_intr<T,compare,link>::Iterator::to_next()
{
    if(!m_current)
        return;
    m_current = m_map->next(m_current);
}

// Set iterator to the previous element of the set if it exists.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline void Set_intr<T,compare,link>::Iterator::to_previous()
{
    if(!m_current)
        return;
    m_current = m_map->previous(m_current);
}

// Set iterator to element of the set.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline void Set_intr<T,compare,link>::Iterator::to_element(
                        T *element)     // The element to set the iterator to.
{
    m_current = element;
}

// Return true if iterator is exhausted, false otherwise.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline bool Set_intr<T,compare,link>::Iterator::at_end()
{
    return !m_current;
}

// Return a reference to the current element.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline T &Set_intr<T,compare,link>::Iterator::operator*()
{
    return *m_current;
}

// Apply member selection to the current element.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline T *Set_intr<T,compare,link>::Iterator::operator->()
{
    return m_current;
}

// Return the number of elements in the set.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline int Set_intr<T,compare,link>::count() const
{
    return m_count;
}

// Insert element into set.
// If the element is in the set already, do not insert and return false.
// Otherwise, return true.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline bool Set_intr<T,compare,link>::insert(
                                T *element) // The element to insert.
{
    if (m_root == 0) {
        m_root = element;
	(element->*link).m_parent = 0;
        (element->*link).m_left = 0;
        (element->*link).m_right = 0;
        ++m_count;
        return true;
    }

    T* parent = 0;
    T* node = m_root;
    bool left_child;
    do {
	int c = compare(element, node);
	parent = node;
	if(c < 0)
	{
	    node = (node->*link).m_left;
	    left_child = true;
	}
	else if(0 < c)
	{
	    node = (node->*link).m_right;
	    left_child = false;
	}
	else
	    return false;
    } while (node);

    if (left_child)
	(parent->*link).m_left = element;
    else
	(parent->*link).m_right = element;
    (element->*link).m_parent = parent;
    (element->*link).m_left = 0;
    (element->*link).m_right = 0;
    ++m_count;
    splay(element);
    return true;
}

// Remove element from set.
// Removing an element from a set that is not part of the set
// results in a fatal error!
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline void Set_intr<T,compare,link>::remove(
                                T *element) // The element to remove.
{
    // tmp eventually replaces element
    T* tmp = 0;
    T* left = (element->*link).m_left;
    T* right = (element->*link).m_right;

    if (!right)
        tmp = left;
    else if (!left)
        tmp = right;
    else {
        // find the first element of "right"
        T* p = right;
        for(; (p->*link).m_left; p = (p->*link).m_left)
            /* skip */;
        // make "left" the left child of the first element of "right"
        (p->*link).m_left = left;
        (left->*link).m_parent = p;
        tmp = right;
    }

    T* parent = (element->*link).m_parent;
    if(parent) {
        if((parent->*link).m_left == element) {
            (parent->*link).m_left = tmp;
        } else if((parent->*link).m_right == element) {
            (parent->*link).m_right = tmp;
        } else {
            ASSERT(M_CONT,0);
        }
        if(tmp)
            (tmp->*link).m_parent = parent;
    } else {
        m_root = tmp;
        if(tmp)
            (tmp->*link).m_parent = 0;
    }
#ifdef ENABLE_ASSERT
    (element->*link).m_parent = 0;
    (element->*link).m_left = 0;
    (element->*link).m_right = 0;
#endif
    --m_count;
}

// Fast data exchange of two Set_intr<T,compare,link>.
// This function swaps two sets by exchanging the elements, which is done
// in constant time. Note that the global swap() functionality falls back to
// this function due to its template specialization.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline void Set_intr<T,compare,link>::swap(
            Set_intr<T,compare,link> &other)    // The other
{
    std::swap(m_root,  other.m_root);
    std::swap(m_count, other.m_count);
}

// Find pattern in set.
// Return pointer to element in set if it exists, NULL otherwise.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline T *Set_intr<T,compare,link>::find(
                                   T *pattern) const // The pattern to find.
{
    return const_cast<Set_intr<T,compare,link> *>(this)->do_find(pattern);
}

// Find greatest element less than or equal to pattern in set.
// Return pointer to element in set if it exists, NULL otherwise.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline T *Set_intr<T,compare,link>::find_leq(
                                    T *pattern) const // The pattern to find.
{
    return const_cast<Set_intr<T,compare,link> *>(this)->do_find_leq(pattern);
}

// Find smallest element greater than or equal to pattern in set.
// Return pointer to element in set if it exists, NULL otherwise.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline T *Set_intr<T,compare,link>::find_geq(
                                    T *pattern) const // The pattern to find.
{
    return const_cast<Set_intr<T,compare,link> *>(this)->do_find_geq(pattern);
}

// Return first element of set if it exists,
// NULL otherwise.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline T *Set_intr<T,compare,link>::first() const
{
    return const_cast<Set_intr<T,compare,link> *>(this)->get_first(m_root);
}

// Return last element of set if it exists,
// NULL otherwise.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline T *Set_intr<T,compare,link>::last() const
{
    return const_cast<Set_intr<T,compare,link> *>(this)->get_last(m_root);
}

// Return the successor of an element in the set if it exists,
// NULL otherwise.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline T *Set_intr<T,compare,link>::next(
        T *element) const // The element of which to return the successor.
{
    if(!element)
        return 0;
    return const_cast<Set_intr<T,compare,link> *>(this)->get_next(element);
}

// Return the predecessor of an element in the set if it exists,
// NULL otherwise.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline T *Set_intr<T,compare,link>::previous(
        T *element) const // The element of which to return the predecessor.
{
    if(!element)
        return 0;
    return const_cast<Set_intr<T,compare,link> *>(this)->get_previous(element);
}

// Return first element in set.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline T *Set_intr<T,compare,link>::get_first(
        T *set) // The submap of which to return the first element.
{
    if(!set)
        return 0;
    T *p;
    for(p = set; (p->*link).m_left; p = (p->*link).m_left)
        /* skip */;
    splay(p);
    return p;
}

// Return last element in set.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline T *Set_intr<T,compare,link>::get_last(
        T *set) // The submap of which to return the last element.
{
    if(!set)
        return 0;
    T *p;
    for(p = set; (p->*link).m_right; p = (p->*link).m_right)
        /* skip */;
    splay(p);
    return p;
}

// Return the successor of an element in the set if it exists,
// NULL otherwise.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline T *Set_intr<T,compare,link>::get_next(
        T *element) // The element of which to return the successor.
{
    if((element->*link).m_right)
        return get_first((element->*link).m_right);
    while((element->*link).m_parent) {
        if(((element->*link).m_parent->*link).m_left
                == element)
	{
	    element = (element->*link).m_parent;
            splay(element);
            return element;
	}
        element = (element->*link).m_parent;
    }
    return 0;
}

// Return the predecessor of an element in the set if it exists,
// NULL otherwise.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline T *Set_intr<T,compare,link>::get_previous(
        T *element) // The element of which to return the predecessor.
{
    if((element->*link).m_left)
        return get_last((element->*link).m_left);
    while((element->*link).m_parent) {
        if(((element->*link).m_parent->*link).m_right
                == element)
	{
	    element = (element->*link).m_parent;
            splay(element);
            return element;
	}
        element = (element->*link).m_parent;
    }
    return 0;
}

// Find pattern in set.
// Return pointer to element in set if it exists, NULL otherwise.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline T *Set_intr<T,compare,link>::do_find(
                                   T *pattern) // The pattern to find.
{
    T *node = m_root;
    T *last = 0;

    while(node) {
        int c = compare(node,pattern);

        if(c < 0) {
            last = node;
            node = (node->*link).m_right;
        } else if(0 < c) {
            last = node;
            node = (node->*link).m_left;
        } else {
            splay(node);
            return node;
        }
    }
    if(last)
        splay(last);
    return 0;
}

// Find greatest element less than or equal to pattern in set.
// Return pointer to element in set if it exists, NULL otherwise.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline T *Set_intr<T,compare,link>::do_find_leq(
                                    T *pattern) // The pattern to find.
{
    if(!m_root)
        return 0;

    T *node = m_root;
    T *last = 0;

    for(;;) {
        int c = compare(node,pattern);
        if(c < 0) {
            if((node->*link).m_right) {
		last = node;
                node = (node->*link).m_right;
            } else {
                break;
            }
        } else if(0 < c) {
            if((node->*link).m_left) {
		last = node;
                node = (node->*link).m_left;
            } else {
		last = node;
                node = previous(node);
                break;
            }
        } else {
            break;
        }
    }
    if(node)
        splay(node);
    else if(last)
        splay(last);
    return node;
}

// Find smallest element greater than or equal to pattern in set.
// Return pointer to element in set if it exists, NULL otherwise.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline T *Set_intr<T,compare,link>::do_find_geq(
                                    T *pattern) // The pattern to find.
{
    if(!m_root)
        return 0;

    T *node = m_root;
    T *last = 0;

    for(;;) {
        int c = compare(node,pattern);
        if(c < 0) {
            if((node->*link).m_right) {
		last = node;
                node = (node->*link).m_right;
            } else {
		last = node;
                node = next(node);
                break;
            }
        } else if(0 < c) {
            if((node->*link).m_left) {
		last = node;
                node = (node->*link).m_left;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    if(node)
	splay(node);
    else if(last)
        splay(last);
    return node;
}

// Move element to top of set, partially balancing the tree representing
// the set in the process.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
void Set_intr<T,compare,link>::splay(
                                T *x)   // The element to move to the top.
{
    for(T *p = (x->*link).m_parent; p; p = (x->*link).m_parent) {
        if((p->*link).m_parent) {
            T *g = (p->*link).m_parent;

            if((p->*link).m_left == x) {
                if((g->*link).m_left == p)
                    zig_zig(x);
                else
                    zag_zig(x);
            } else {
                if((g->*link).m_left == p)
                    zig_zag(x);
                else
                    zag_zag(x);
            }
        } else {
            if((p->*link).m_left == x)
                zig(x);
            else
                zag(x);
        }
    }
}

// Single right rotation of x and its parent.
//
//                  P                               X
//
//          X               C       =>      A               P
//
//      A       B                                       B       C
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline void Set_intr<T,compare,link>::zig(
                                            T *x)   // The element to move.
{
    T *p = (x->*link).m_parent;
    T *b = (x->*link).m_right;

    (x->*link).m_right = p;
    (x->*link).m_parent = 0;
    m_root = x;
    if(b)
        (b->*link).m_parent = p;
    (p->*link).m_left = b;
    (p->*link).m_parent = x;
}

// Single left rotation of x and its parent.
//
//                  P                               X
//
//          A               X       =>      P           C
//
//                      B       C       A       B
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline void Set_intr<T,compare,link>::zag(
                                            T *x)   // The element to move.
{
    T *p = (x->*link).m_parent;
    T *b = (x->*link).m_left;

    (x->*link).m_left = p;
    (x->*link).m_parent = 0;
    m_root = x;
    if(b)
        (b->*link).m_parent = p;
    (p->*link).m_right = b;
    (p->*link).m_parent = x;
}

// Double right rotation at x.
//
//                      G                               X
//
//              P               D               A               P
//                                  =>
//          X       C                                       B       G
//
//      A       B                                               C       D
//
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline void Set_intr<T,compare,link>::zig_zig(
                                            T *x)   // The element to move.
{
    T *p = (x->*link).m_parent;
    T *g = (p->*link).m_parent;
    T *b = (x->*link).m_right;
    T *c = (p->*link).m_right;
    T *gg = (g->*link).m_parent;

    (x->*link).m_right = p;
    (p->*link).m_parent = x;

    (p->*link).m_right = g;
    (g->*link).m_parent = p;

    if(b)
        (b->*link).m_parent = p;
    (p->*link).m_left = b;

    if(c)
        (c->*link).m_parent = g;
    (g->*link).m_left = c;

    (x->*link).m_parent = gg;
    if(gg) {
        if((gg->*link).m_left == g)
            (gg->*link).m_left = x;
        else
            (gg->*link).m_right = x;
    } else {
        m_root = x;
    }
}

// Left-right rotation at x.
//
//                      G                               X
//
//              P               D               P               G
//                                  =>
//          A       X                       A       B       C       D
//
//              B       C
//
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline void Set_intr<T,compare,link>::zig_zag(
                                                T *x)   // The element to move.
{
    T *p = (x->*link).m_parent;
    T *g = (p->*link).m_parent;
    T *b = (x->*link).m_left;
    T *c = (x->*link).m_right;
    T *gg = (g->*link).m_parent;

    (x->*link).m_left = p;
    (p->*link).m_parent = x;

    (x->*link).m_right = g;
    (g->*link).m_parent = x;

    if(b)
        (b->*link).m_parent = p;
    (p->*link).m_right = b;

    if(c)
        (c->*link).m_parent = g;
    (g->*link).m_left = c;

    (x->*link).m_parent = gg;
    if(gg) {
        if((gg->*link).m_left == g)
            (gg->*link).m_left = x;
        else
            (gg->*link).m_right = x;
    } else {
        m_root = x;
    }
}

// Right-left rotation at x.
//
//              G                                       X
//
//      A                   P                   G               P
//                                  =>
//                      X       D           A       B       C       D
//
//                  B       C
//
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline void Set_intr<T,compare,link>::zag_zig(
                                                T *x)   // The element to move.
{
    T *p = (x->*link).m_parent;
    T *g = (p->*link).m_parent;
    T *b = (x->*link).m_left;
    T *c = (x->*link).m_right;
    T *gg = (g->*link).m_parent;

    (x->*link).m_left = g;
    (g->*link).m_parent = x;

    (x->*link).m_right = p;
    (p->*link).m_parent = x;

    if(b)
        (b->*link).m_parent = g;
    (g->*link).m_right = b;

    if(c)
        (c->*link).m_parent = p;
    (p->*link).m_left = c;

    (x->*link).m_parent = gg;
    if(gg) {
        if((gg->*link).m_left == g)
            (gg->*link).m_left = x;
        else
            (gg->*link).m_right = x;
    } else {
        m_root = x;
    }
}

// Double left rotation at x.
//
//                  G                                           X
//
//          A               P                           P               D
//                                      =>
//                      B       X                   G       C
//
//                          C       D           A       B
//
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline void Set_intr<T,compare,link>::zag_zag(
                                                T *x)   // The element to move.
{
    T *p = (x->*link).m_parent;
    T *g = (p->*link).m_parent;
    T *b = (p->*link).m_left;
    T *c = (x->*link).m_left;
    T *gg = (g->*link).m_parent;

    (x->*link).m_left = p;
    (p->*link).m_parent = x;

    (p->*link).m_left = g;
    (g->*link).m_parent = p;

    if(c)
        (c->*link).m_parent = p;
    (p->*link).m_right = c;

    if(b)
        (b->*link).m_parent = g;
    (g->*link).m_right = b;

    (x->*link).m_parent = gg;
    if(gg) {
        if((gg->*link).m_left == g)
            (gg->*link).m_left = x;
        else
            (gg->*link).m_right = x;
    } else {
        m_root = x;
    }
}

// Clear the set
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline void Set_intr<T,compare,link>::clear()
{
    m_root  = NULL;
    m_count = 0;

    // TO DO: in the DEBUG case, add a loop which sets all
    // pointers to zero.
}

// Dump the tree to standard output.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline void Set_intr<T,compare,link>::dump(
            void (*print)(T *element)) const    // The function to use
                                                // for printing elements.
{
    dump(print,0,m_root);
}

// Dump tree at level.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
void Set_intr<T,compare,link>::dump(
                    void (*print)(T *element),  // The function to use
                                                // for printing elements.
                    int level,                  // The indentation level.
                    T *set) const               // The submap to dump.
{
    if(!set)
        return;
    dump(print,level+1,(set->*link).m_right);
//    for(int i = 0; i < level; i++)
//        printf("\t");
    print(set);
//    printf("\n");
    dump(print,level+1,(set->*link).m_left);
}

// Check consistency.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline void Set_intr<T,compare,link>::check_consistency() const
{
    if(m_root) {
        ASSERT(M_CONT,(m_root->*link).m_parent == 0);
        check_consistency(m_root);
    }
}

// Check consistency.
template<class T,int (*compare)(T *left,T *right),Set_link<T> T::*link>
inline void Set_intr<T,compare,link>::check_consistency(
                                        T *set) const // The submap to check.
{
    if((set->*link).m_left) {
        ASSERT(M_CONT,set == ((set->*link).m_left->*link).m_parent);
        check_consistency((set->*link).m_left);
    }
    if((set->*link).m_right) {
        ASSERT(M_CONT,set == ((set->*link).m_right->*link).m_parent);
        check_consistency((set->*link).m_right);
    }
}

// See Set_itr::swap().
template<
    class T,                            // Type of set elements.
    int (*compare)(T *left,T *right),   // Comparison function.
    Set_link<T> T::*link>               // Member pointer for link fields.
inline
void swap(
    Set_intr<T,compare,link> &one,	// The one
    Set_intr<T,compare,link> &other)	// The other
{
    one.swap(other);
}

// Constructor.
template<class T,int (*compare)(const T &left,const T &right)>
inline Set<T,compare>::Set()
{}

// Copy constructor.
template<class T,int (*compare)(const T &left,const T &right)>
inline Set<T,compare>::Set(
                            const Set<T,compare> &set)  // The set to copy.
{
    Iterator i(&set);

    for(i.to_first(); !i.at_end(); i.to_next())
        insert(*i);
}

// Copy assignment.
template<class T,int (*compare)(const T &left,const T &right)>
inline Set<T,compare> &Set<T,compare>::operator=(
                            const Set<T,compare> &set)  // The set to assign.
{
    if(this == &set)
	return *this;
    Iterator i(&set);

    clear();
    for(i.to_first(); !i.at_end(); i.to_next())
        insert(*i);
    return *this;
}

// Destructor.
template<class T,int (*compare)(const T &left,const T &right)>
Set<T,compare>::~Set()
{
    clear();
}

// Constructor.
// set is the set to iterate over.
template<class T,int (*compare)(const T &left,const T &right)>
inline Set<T,compare>::Const_iterator::Const_iterator(
                        const Set<T,compare> *set)  // The set to iterate over.
    : m_map(set)
    , m_current(NULL)
{}

// Destructor.
template<class T,int (*compare)(const T &left,const T &right)>
Set<T,compare>::Const_iterator::~Const_iterator()
{}

// Set iterator to the first element of the set if it exists.
template<class T,int (*compare)(const T &left,const T &right)>
inline void Set<T,compare>::Const_iterator::to_first()
{
    m_current = m_map->m_map.first();
}

// Set iterator to the last element of the set if it exists.
template<class T,int (*compare)(const T &left,const T &right)>
inline void Set<T,compare>::Const_iterator::to_last()
{
    m_current = m_map->m_map.last();
}

// Set iterator to the next element of the set if it exists.
template<class T,int (*compare)(const T &left,const T &right)>
inline void Set<T,compare>::Const_iterator::to_next()
{
    if(!m_current)
        return;
    m_current = m_map->m_map.next(m_current);
}

// Set iterator to the previous element of the set if it exists.
template<class T,int (*compare)(const T &left,const T &right)>
inline void Set<T,compare>::Const_iterator::to_previous()
{
    if(!m_current)
        return;
    m_current = m_map->m_map.previous(m_current);
}

// Return true if iterator is exhausted, false otherwise.
template<class T,int (*compare)(const T &left,const T &right)>
inline bool Set<T,compare>::Const_iterator::at_end()
{
    return !m_current;
}

// Return a reference to the current element.
template<class T,int (*compare)(const T &left,const T &right)>
inline const T& Set<T,compare>::Const_iterator::operator*()
{
    return m_current->m_value;
}

// Apply member selection to the current element.
template<class T,int (*compare)(const T &left,const T &right)>
inline const T* Set<T,compare>::Const_iterator::operator->()
{
    return &(operator*());
}

// Constructor.
// set is the set to iterate over.
template<class T,int (*compare)(const T &left,const T &right)>
inline Set<T,compare>::Iterator::Iterator(
                        const Set<T,compare> *set)  // The set to iterate over.
    : Const_iterator(set)
{}

// Destructor.
template<class T,int (*compare)(const T &left,const T &right)>
Set<T,compare>::Iterator::~Iterator()
{}

// Return a reference to the current element.
template<class T,int (*compare)(const T &left,const T &right)>
inline T& Set<T,compare>::Iterator::operator*()
{
    return this->m_current->m_value;
}


// Apply member selection to the current element.
template<class T,int (*compare)(const T &left,const T &right)>
inline T* Set<T,compare>::Iterator::operator->()
{
    return &(operator*());
}

// Remove current element from set and advance to next
template<class T,int (*compare)(const T &left,const T &right)>
inline void Set<T,compare>::Iterator::remove()
{
    if (at_end())
	// No current value => nothing to remove
	return;
    Node *current = m_current;
    to_next();
    const_cast<Set_intr<Node,Node::cmp,&Node::m_link> *>
	(&this->m_map->m_map)->remove(current);
}

// Return the number of elements in the set.
template<class T,int (*compare)(const T &left,const T &right)>
inline int Set<T,compare>::count() const
{
    return m_map.count();
}

// Insert element into set.
// If the element is in the set already, do not insert and return false.
// Otherwise, return true.
template<class T,int (*compare)(const T &left,const T &right)>
inline bool Set<T,compare>::insert(
                                    const T &element) // The element to insert.
{
    return m_map.insert(new Node(element));
}

// Remove element from set.
template<class T,int (*compare)(const T &left,const T &right)>
inline void Set<T,compare>::remove(
                                    const T &element) // The element to remove.
{
    Node s(element);

    Node *r = m_map.find(&s);
    if(r) {
        m_map.remove(r);
        delete r;
    }
}

// Clear all elements from set.
template<class T,int (*compare)(const T &left,const T &right)>
inline void Set<T,compare>::clear()
{
    Node *p;

    while((p = m_map.first())) {
        m_map.remove(p);
        delete p;
    }
}

// Fast data exchange of two Set_intr<T,compare,link>.
// This function swaps two sets by exchanging the elements, which is done
// in constant time. Note that the global swap() functionality falls back to
// this function due to its template specialization.
template<class T,int (*compare)(const T &left,const T &right)>
inline void Set<T,compare>::swap(
            Set<T,compare> &other)    // The other
{
    // Thanks to a xion's gcc 3.4.3 20041212 (Red Hat 3.4.3-9.EL4) compiler bug
    // the ADL namelookup-based call had to be changed into an explicit one.
    //using std::swap;
    m_map.swap(other.m_map);
}

// Find pattern in set and return in element.
// Return true if pattern is found in set, and false otherwise.
// element is only set if return value is true.
template<class T,int (*compare)(const T &left,const T &right)>
inline bool Set<T,compare>::find(
			    const T &pattern, // The pattern to find.
			    T *element) const // Pointer to the element found.
{
    Node s(pattern);

    Node *r = m_map.find(&s);
    if(r) {
        *element = r->m_value;
        return true;
    } else {
        return false;
    }
}

// Find greatest element less than pattern in set and return in element.
// Return true if pattern is found in set, and false otherwise.
// element is only set if return value is true.
template<class T,int (*compare)(const T &left,const T &right)>
inline bool Set<T,compare>::find_leq(
			    const T &pattern, // The pattern to find.
			    T *element) const // Pointer to the element found.
{
    Node s(pattern);

    Node *r = m_map.find_leq(&s);
    if(r) {
        *element = r->m_value;
        return true;
    } else {
        return false;
    }
}

// Find smallest element greater than pattern in set and return in element.
// Return true if pattern is found in set, and false otherwise.
// element is only set if return value is true.
template<class T,int (*compare)(const T &left,const T &right)>
inline bool Set<T,compare>::find_geq(
			    const T &pattern, // The pattern to find.
			    T *element) const // Pointer to the element found.
{
    Node s(pattern);

    Node *r = m_map.find_geq(&s);
    if(r) {
        *element = r->m_value;
        return true;
    } else {
        return false;
    }
}

// Check consistency.
template<class T,int (*compare)(const T &left,const T &right)>
inline void Set<T,compare>::check_consistency() const
{
    m_map.check_consistency();
}

// See Set::swap().
template<class T,int (*compare)(const T &left,const T &right)>
inline void swap(
            Set<T,compare> &one,	// The one
            Set<T,compare> &other)	// The other
{
    one.swap(other);
}

}

}
