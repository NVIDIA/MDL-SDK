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
 ///        Sets implement ordered collections of items, typically used
 ///        Class Set is parametrized by a node type, a compare function,
 ///        and, in the case of intrusive maps, a pointer to member giving
 ///        the location of the link fields in the elements.
 ///
 ///        A set can never contain two elements that compare equal.
 ///        An attempt to insert an element into a set that already contains an
 ///        element that compares to equal with the element to be inserted will
 ///        fail.
 ///
 ///        NOTE: Set_intr, Set, and Map are based on splay trees which have
 ///        linear worst-case complexity.
 ///
 ///        This is an example usage of non-intrusive sets:
 ///
 ///        int set_int_compare(const int &a,const int &b) {
 ///            return a - b;
 ///        }
 ///
 ///        void example_map(void)
 ///        {
 ///            using namespace MI::CONT;
 ///
 ///            typedef Set<int,set_int_compare> Set_int;
 ///
 ///            int n;
 ///
 ///            Set_int set;
 ///
 ///            for(n = 0; n < 10; n++)
 ///                set.insert(n);
 ///
 ///            Set_int::Iterator i(&set);
 ///
 ///            for(i.to_first(); !i.at_end(); i.to_next())
 ///                printf("%d\n",*i);
 ///
 ///            if(set.find(1,&n))
 ///                printf("found %d\n",n);
 ///            else
 ///                printf("did not find 1\n");
 ///        }

#ifndef BASE_LIB_CONT_SET_H
#define BASE_LIB_CONT_SET_H

#include <base/lib/mem/i_mem_allocatable.h>
#include "cont_tree.h"

namespace MI {

namespace CONT {

template<
    class T>    // Type of set elements.
class Set_link : public Tree_link<T> {};

// Intrusive set, where links are embedded in nodes carrying the data.
template<
    class T,                            // Type of set elements.
    int (*compare)(T *left,T *right),   // Comparison function.
    Set_link<T> T::*link>               // Member pointer for link fields.
class Set_intr : public MEM::Allocatable {

public:

    // Constructor.
    Set_intr();

    // Destructor.
    ~Set_intr();

    //=========================================================================
    // Iterator.
    class Iterator {

    public:

        // Constructor.
        // set is the set to iterate over.
        Iterator(
            const Set_intr<T,compare,link> *set);   // The set to iterate over.

        // Destructor.
        ~Iterator();

        // Set iterator to the first element of the set if it exists.
        void to_first();

        // Set iterator to the last element of the set if it exists.
        void to_last();

        // Set iterator to the next element of the set if it exists.
        void to_next();

        // Set iterator to the previous element of the set if it exists.
        void to_previous();

        // Set iterator to element of the set.
        void to_element(
                        T *element);    // The lement to set the iterator to.

        // Return true if iterator is exhausted, false otherwise.
        bool at_end();

        // Return a reference to the current element.
        T &operator*();

        // Apply member selection to the current element.
        T *operator->();

    private:

        // Pointer to the set to iterate over.
        const Set_intr<T,compare,link> *m_map;

        // Pointer to the current node in the set.
        T *m_current;

    };
    //=========================================================================

    // Return the number of elements in the set.
    int count() const;

    // Insert element into set.
    // If the element is in the set already, do not insert and return false.
    // Otherwise, return true.
    bool insert(
                T *element);    // The element to insert.

    // Remove element from set.
    // Removing an element from a set that is not part of the set
    // results in a fatal error!
    void remove(
                T *element);    // The element to remove.

    // Fast data exchange of two Set_intr<T,compare,link>.
    void swap(
	Set_intr<T,compare,link> &other);    // The other

    // Find pattern in set.
    // Return pointer to element in set if it exists, NULL otherwise.
    T *find(
                T *pattern) const;    // The pattern to find.

    // Find greatest element less than or equal to pattern in set.
    // Return pointer to element in set if it exists, NULL otherwise.
    T *find_leq(
                T *pattern) const;    // The pattern to find.

    // Find smallest element greater than or equal to pattern in set.
    // Return pointer to element in set if it exists, NULL otherwise.
    T *find_geq(
                T *pattern) const;    // The pattern to find.

    // Return first element of set if it exists,
    // NULL otherwise.
    T *first() const;

    // Return last element of set if it exists,
    // NULL otherwise.
    T *last() const;

    // Return the successor of an element in the set if it exists,
    // NULL otherwise.
    T *next(
        T *element) const;  // The element of which to return the succesor.

    // Return the predecessor of an element in the set if it exists,
    // NULL otherwise.
    T *previous(
        T *element) const;  // The element of which to return the predecessor.

    // Clear the set
    void clear();

    // Dump the set to standard output.
    void dump(
        void (*print)(T *element)) const;   // The function to use
                                            // for printing elements.

    // Check consistency.
    void check_consistency() const;

private:
    // Root of the set.
    T *m_root;

    // Number of elements in set.
    int m_count;

    // Copy constructor.
    Set_intr(
                const Set_intr &);  // The set to copy.

    // Copy assignment.
    Set_intr &operator=(
                const Set_intr &);  // The set to assign.

    // Return first element in set.
    T *get_first(
        T *set);        // The submap in which to find the first element.

    // Return last element in set.
    T *get_last(
        T *set);        // The submap in which to find the last element.

    // Return the successor of an element in the set if it exists,
    // NULL otherwise.
    T *get_next(
        T *element);  // The element of which to return the succesor.

    // Return the predecessor of an element in the set if it exists,
    // NULL otherwise.
    T *get_previous(
        T *element);  // The element of which to return the predecessor.

    // Find pattern in set.
    // Return pointer to element in set if it exists, NULL otherwise.
    T *do_find(
                T *pattern);    // The pattern to find.

    // Find greatest element less than or equal to pattern in set.
    // Return pointer to element in set if it exists, NULL otherwise.
    T *do_find_leq(
                T *pattern);    // The pattern to find.

    // Find smallest element greater than or equal to pattern in set.
    // Return pointer to element in set if it exists, NULL otherwise.
    T *do_find_geq(
                T *pattern);    // The pattern to find.

    // Move element to top of set, partially balancing the tree representing
    // the set in the process.
    void splay(
                T *x);  // The element to move to the top of the set.

    // Single right rotation of x and its parent.
    //
    //              P                               X
    //
    //      X               C       =>      A               P
    //
    //  A       B                                       B       C
    void zig(
                T *x);  // The node to rotate.

    // Single left rotation of x and its parent.
    //
    //              P                               X
    //
    //      A               X       =>      P           C
    //
    //                  B       C       A       B
    void zag(
                T *x);  // The node to rotate.

    // Double right rotation at x.
    //
    //                  G                               X
    //
    //          P               D               A               P
    //                              =>
    //      X       C                                       B       G
    //
    //  A       B                                               C       D
    //
    void zig_zig(
                T *x);  // The node to rotate.

    // Left-right rotation at x.
    //
    //                  G                               X
    //
    //          P               D               P               G
    //                              =>
    //      A       X                       A       B       C       D
    //
    //          B       C
    //
    void zig_zag(
                T *x);  // The node to rotate.

    // Right-left rotation at x.
    //
    //          G                                       X
    //
    //  A                   P                   G               P
    //                              =>
    //                  X       D           A       B       C       D
    //
    //              B       C
    //
    void zag_zig(
                T *x);  // The node to rotate.

    // Double left rotation at x.
    //
    //              G                                           X
    //
    //      A               P                           P               D
    //                                  =>
    //                  B       X                   G       C
    //
    //                      C       D           A       B
    //
    void zag_zag(
                T *x);  // The node to rotate.

    // Dump set at level.
    void dump(
                void (*print)(T *element),  // The function to use
                                            // for printing elements.
                int level,                  // The level of indentation.
                T *set) const;              // The submap to dump.

    // Check consistency.
    void check_consistency(
                T *set) const;              // The submap to check.

};

// Specialization of the default swap() for Set_intr.
// see Set_intr::swap().
template<
    class T,                            // Type of set elements.
    int (*compare)(T *left,T *right),   // Comparison function.
    Set_link<T> T::*link>               // Member pointer for link fields.
void swap(
	Set_intr<T,compare,link> &one,	    // The one
	Set_intr<T,compare,link> &other);   // The other

// Non-intrusive set, where nodes contain pointers to data.
template<
    class T,                                          // Type of set elements.
    int (*compare)(const T &left,const T &right)>     // Comparision function.
class Set {

public:

    class Const_iterator;
    class Iterator;

private:

    //=========================================================================
    // Node class for Set.
    class Node {

        friend class Set<T,compare>;
        friend class Const_iterator;
	friend class Iterator;

    private:

        // Pointers to other nodes.
        Set_link<Node> m_link;

        // Data.
        T m_value;

        // Constructor.
        Node(
                T value)	// The initial value of the node.
        {
            m_value = value;
        }

        // Copy constructor.
        Node(
                const Node&);	// The node to copy.

        // Copy assignment.
        Node &operator=(
                const Node&);	// The node to assign.

        // Compare two nodes,
        // returning a negative value if left is smaller,
        // a positive value if right is smaller,
        // and 0 if both are equal.
        static int cmp(
                        Node *left,     // The left node.
                        Node *right)    // The right node.
        {
            return compare(left->m_value,right->m_value);
        }

    };
    //=========================================================================

    // The intrusive set implementing the non-intrusive set.
    Set_intr<Node,Node::cmp,&Node::m_link> m_map;

public:

    // Constructor.
    Set();

    // Copy constructor.
    Set(
            const Set &set);    // The set to copy.

    // Copy assignment.
    Set &operator=(
            const Set &set);    // The set to assign.

    // Destructor.
    ~Set();

    //=========================================================================
    // Iterator.
    class Const_iterator {

    public:

        // Constructor.
        // set is the set to iterate over.
        Const_iterator(
                    const Set<T,compare> *set); // The set to iterate over.

        // Destructor.
        ~Const_iterator();

        // Set iterator to the first element of the set if it exists.
        void to_first();

        // Set iterator to the last element of the set if it exists.
        void to_last();

        // Set iterator to the next element of the set if it exists.
        void to_next();

        // Set iterator to the previous element of the set if it exists.
        void to_previous();

        // Return true if iterator is exhausted, false otherwise.
        bool at_end();

        // Return a reference to the current element.
        const T& operator*();

        // Apply member selection to the current element.
        const T* operator->();
#ifndef INTEL_COMPILER_BUG_01_WORKAROUND
    protected:
#endif
        // Pointer to the set to iterate over.
        const Set<T,compare> *m_map;

        // Pointer to the current node in the set.
        typename Set<T,compare>::Node *m_current;

    };

    //=========================================================================
    // Iterator.
    class Iterator : public Const_iterator {
    public:
#ifndef INTEL_COMPILER_BUG_01_WORKAROUND
	using Const_iterator::to_first;
	using Const_iterator::to_last;
	using Const_iterator::to_next;
	using Const_iterator::to_previous;
	using Const_iterator::at_end;
	using Const_iterator::m_current;
#endif

        // Constructor.
        // set is the set to iterate over.
        Iterator(
                    const Set<T,compare> *set); // The set to iterate over.

        // Destructor.
        ~Iterator();

	// Return a reference to the current element.
        T& operator*();

        // Apply member selection to the current element.
        T* operator->();

	// Remove current element from set and advance to next
	void remove();
    };

    //=========================================================================

    // Return the number of elements in the set.
    int count() const;

    // Insert element into set.
    // If the element is in the set already, do not insert and return false.
    // Otherwise, return true.
    bool insert(
                const T &element); // The element to insert.

    // Remove element from set.
    void remove(
                const T &element); // The element to remove.

    // Clear all elements from set.
    void clear();

    // Fast data exchange of two Set<T,compare>.
    void swap(
	Set<T,compare>	&other);    // The other

    // Find pattern in set and return in element.
    // Return true if pattern is found in set, and false otherwise.
    // element is only set if return value is true.
    bool find(
                const T &pattern,	// The pattern to find.
                T *element) const;	// Pointer to the element found.

    // Find greatest element less than or equal to pattern in set
    // and return in element.
    // Return true if pattern is found in set, and false otherwise.
    // element is only set if return value is true.
    bool find_leq(
                const T &pattern,	// The pattern to find.
                T *element) const;	// Pointer to the element found.

    // Find smallest element greater than or equal to pattern in set
    // and return in element.
    // Return true if pattern is found in set, and false otherwise.
    // element is only set if return value is true.
    bool find_geq(
                const T &pattern,	// The pattern to find.
                T *element) const;	// Pointer to the element found.

    // Check consistency.
    void check_consistency() const;

};

// Specialization of the default swap() for Set.
// see Set::swap().
template<
    class T,                                          // Type of set elements.
    int (*compare)(const T &left,const T &right)>     // Comparision function.
void swap(
	Set<T,compare> &one,	    // The one
	Set<T,compare> &other);     // The other
}

}

#include "cont_set_inline.h"

#endif
