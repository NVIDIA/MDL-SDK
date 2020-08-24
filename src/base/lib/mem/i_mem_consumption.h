/******************************************************************************
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
 *****************************************************************************/
/// \file
/// \brief Utilities to estimate the dynamic memory consumption of objects.
///
/// - \c has_dynamic_memory_consumption(const T&) Decide whether T uses (might use) dynamic memory.
/// - \c dynamic_memory_consumption(const T&) Some overloaded functions to estimate dynamic memory.
///
///  To estimate the real memory consumption of an object, we cannot simple use sizeof(T),
///  because that gives us only the memory size of the object itself, but not the size of
///  the dynamic (heap allocated) memory, used by the object.
///  This file provides helpers to estimate dynamic memory usage.
///
///  This is a separate header, because it should only be included on demand.

#ifndef BASE_LIB_MEM_CONSUMPTION_H
#define BASE_LIB_MEM_CONSUMPTION_H

#include <cstddef> // size_t

#include <map>
#include <boost/unordered_map.hpp>
#include <set>
#include <string>
#include <utility> // pair
#include <vector>

#include <base/system/main/types.h>

#include <mi/base/config.h>

// This file defines a framework to compute the dynamic memory used by an object.
//
// To support a given type T two functions
//
// - bool has_dynamic_memory_consumption(const T&) and
// - size_t dynamic_memory_consumption(const T&)
//
// need to be implemented. The first one returns whether the type T uses (or might use) dynamic
// memory. It must only use the information about the type T itself, not a particular argument, and
// if necessary, err on the safe side, i.e., return true.
//
// The second one returns the amount of dynamic memory used by a particular instance of T. If
// has_dynamic_memory_consumption() returns false, dynamic_memory_consumption() must return 0.
//
// The first function fulfills the purpose of type traits. But type traits require that all
// specializations are in the same namespace -- which might work for types in this file, but is very
// inconvenient for types defined elsewhere.


// Macros that help with the implementation of these two functions.
//
// Note that using these macros requires to include this header files. That's why most definitions
// for types elsewhere do not use these macros (to avoid the include dependency).


// Macro for types that do not use dynamic memory.
#define MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(T) \
    inline bool has_dynamic_memory_consumption(const T&) { return false; } \
    inline size_t dynamic_memory_consumption(const T&) { return 0; }

// Macro for templates with one argument that do not use dynamic memory.
#define MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION_TEMPLATE(T,TARG) \
    template <class TARG> \
    inline bool has_dynamic_memory_consumption(const T&) { return false; } \
    template <class TARG> \
    inline size_t dynamic_memory_consumption(const T&) { return 0; }

// Macro for templates with two arguments that do not use dynamic memory.
#define MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION_TEMPLATE2(T1,T2,TARG1,TARG2) \
    template <class TARG1, class TARG2> \
    inline bool has_dynamic_memory_consumption(const T1,T2&) { return false; } \
    template <class TARG1, class TARG2> \
    inline size_t dynamic_memory_consumption(const T1,T2&) { return 0; }

// Macro for templates with three arguments that do not use dynamic memory.
#define MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION_TEMPLATE3(T1,T2,T3,TARG1,TARG2,TARG3) \
    template <class TARG1, class TARG2, class TARG3> \
    inline bool has_dynamic_memory_consumption(const T1,T2,T3&) { return false; } \
    template <class TARG1, class TARG2, class TARG3> \
    inline size_t dynamic_memory_consumption(const T1,T2,T3&) { return 0; }

// Macro for templates with five arguments that do not use dynamic memory.
#define MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION_TEMPLATE5( \
    T1,T2,T3,T4,T5,TARG1,TARG2,TARG3,TARG4,TARG5) \
    template <class TARG1, class TARG2, class TARG3, class TARG4, class TARG5> \
    inline bool has_dynamic_memory_consumption(const T1,T2,T3,T4,T5&) { return false; } \
    template <class TARG1, class TARG2, class TARG3, class TARG4, class TARG5> \
    inline size_t dynamic_memory_consumption(const T1,T2,T3,T4,T5&) { return 0; }


// Macro for types that do use dynamic memory. Define dynamic_memory_consumption() yourself.
#define MI_MEM_HAS_DYNAMIC_MEMORY_CONSUMPTION(T) \
    inline bool has_dynamic_memory_consumption(const T&) { return true; }

// Variant for templates with one argument (e.g., std::vector and std::set)
#define MI_MEM_HAS_DYNAMIC_MEMORY_CONSUMPTION_TEMPLATE(T,TARG) \
    template <class TARG> \
    inline bool has_dynamic_memory_consumption(const T&) { return true; }

// Variant for templates with two arguments (e.g., std::map, or std::set with comparison funct.)
#define MI_MEM_HAS_DYNAMIC_MEMORY_CONSUMPTION_TEMPLATE2(T1,T2,TARG1,TARG2) \
    template <class TARG1, class TARG2> \
    inline bool has_dynamic_memory_consumption(const T1,T2&) { return true; }

// Variant for templates with three arguments (e.g., std::map with comparison functor)
#define MI_MEM_HAS_DYNAMIC_MEMORY_CONSUMPTION_TEMPLATE3(T1,T2,T3,TARG1,TARG2,TARG3) \
    template <class TARG1, class TARG2, class TARG3> \
    inline bool has_dynamic_memory_consumption(const T1,T2,T3&) { return true; }

// Variant for templates with five arguments (e.g., std::hash_map)
#define MI_MEM_HAS_DYNAMIC_MEMORY_CONSUMPTION_TEMPLATE5( \
    T1,T2,T3,T4,T5,TARG1,TARG2,TARG3,TARG4,TARG5) \
    template <class TARG1, class TARG2, class TARG3, class TARG4, class TARG5> \
    inline bool has_dynamic_memory_consumption(const T1,T2,T3,T4,T5&) { return true; }


// Macro for pointer to T types that do not use dynamic memory.
#define MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION_PTR(T) \
    inline bool has_dynamic_memory_consumption(const T *) { return false; } \
    inline size_t dynamic_memory_consumption(const T *) { return 0; }


// Memory consumption for a few built-in types.
//
// Note that these definitions are in namespaces MISTD *and* boost::unordered because ADL does not
// work for built-in types (the set of "associated namespaces" and "associated classes" is empty,
// and not the global namespace). In namespace std they are needed for STL containers (see below),
// in namespace boost::unordered they are needed for the boost unordered containers (also see
// below). In theory, one could also duplicate the definitions in the MI namespace, but no-one seems
// to need them there.

namespace std {

MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(int)
MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(unsigned int)
MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(unsigned long long int)
MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(float)
MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(double)
MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(long double)

}

namespace boost {

namespace unordered {

MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(int)
MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(unsigned int)
MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(unsigned long long int)
MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(float)
MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(double)
MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(long double)

}

}


// Memory consumption for a few standard MI types.

namespace MI {

MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(Uint)
MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(Sint)
MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(Uint8)
MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(Sint8)
MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(Uint16)
MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(Sint16)
MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(Uint64)
MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(Sint64)
#ifndef MI_PLATFORM_WINDOWS
#ifdef MI_ARCH_64BIT
MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(size_t)
#endif
#endif

} // MI


// Memory consumption for a few types from the base API.

namespace mi {

namespace base {

template <class T>
class Handle;

template <class T>
bool has_dynamic_memory_consumption( const Handle<T>&) { return true; }

template <class T>
size_t dynamic_memory_consumption( const Handle<T>& h)
{
    return h ? dynamic_memory_consumption( h.get()) : 0;
}

} // namespace base

} // namespace mi


// Memory consumption for a few types from the math API.

namespace mi {

namespace math {

class Color;
MI_MEM_HAS_NO_DYNAMIC_MEMORY_CONSUMPTION(Color)

template <class T, Size DIM>
class Vector;

template <class T, Size DIM>
inline bool has_dynamic_memory_consumption(const Vector<T,DIM>&) { return false; }

template <class T, Size DIM>
inline size_t dynamic_memory_consumption(const Vector<T,DIM>&) { return 0; }

template <class T, Size ROW, Size COL>
class Matrix;

template <class T, Size ROW, Size COL>
inline bool has_dynamic_memory_consumption(const Matrix<T,ROW,COL>&) { return false; }

template <class T, Size ROW, Size COL>
inline size_t dynamic_memory_consumption(const Matrix<T,ROW,COL>&) { return 0; }

} // math

} // mi


// Memory consumption for a few STL types: string, pair, vector, map, multimap, set.

namespace std {


// std::string

MI_MEM_HAS_DYNAMIC_MEMORY_CONSUMPTION(string)

inline size_t dynamic_memory_consumption (const string& s)
{
#ifdef MI_PLATFORM_LINUX
    return s.size() <= 15 ? 0 : s.capacity() + 1;
#else // MI_PLATFORM_LINUX
    return s.capacity() + 1;
#endif // MI_PLATFORM_LINUX
}


// std::pair

MI_MEM_HAS_DYNAMIC_MEMORY_CONSUMPTION_TEMPLATE2(pair<T1,T2>,T1,T2)

template <class T1, class T2>
inline size_t dynamic_memory_consumption (const pair<T1,T2>& the_pair)
{
    // static and dynamic size of the pair elements
    return sizeof(T1) + dynamic_memory_consumption (the_pair.first)
         + sizeof(T2) + dynamic_memory_consumption (the_pair.second);
}


// std::vector

MI_MEM_HAS_DYNAMIC_MEMORY_CONSUMPTION_TEMPLATE(vector<T>,T)
MI_MEM_HAS_DYNAMIC_MEMORY_CONSUMPTION_TEMPLATE2(vector<T,A>,T,A)

template <typename T,typename A>
inline size_t dynamic_memory_consumption (const vector<T,A>& the_vector)
{
    // static size of the vector elements
    size_t total = the_vector.capacity() * sizeof(T);

    // additional dynamic size of the vector elements
    if (the_vector.size() > 0 && has_dynamic_memory_consumption (the_vector[0])) {
        const size_t n = the_vector.size();
        for (size_t i = 0; i < n; ++i)
            total += dynamic_memory_consumption (the_vector[i]);
    }

    return total;
}

template <>
inline size_t dynamic_memory_consumption (const vector<bool>& the_vector)
{
    // static size of the vector elements
    return the_vector.capacity()/8;
}


// Attempt to detect -stdlib=libc++ command-line option for clang on MacOS X (affects the tree node
// types used.)
#ifdef MI_PLATFORM_MACOSX
#ifdef __GLIBCXX__ // attempt to detect -stdlib=libc++ command-line option
#define MI_PLATFORM_MACOSX_USING_RB_TREE_NODE
#else
#define MI_PLATFORM_MACOSX_USING_NODE
#endif
#endif // MI_PLATFORM_MACOSX

// std::map

MI_MEM_HAS_DYNAMIC_MEMORY_CONSUMPTION_TEMPLATE2(map<T1,T2>,T1,T2)

template < class T1 ,class T2 >
inline size_t dynamic_memory_consumption (const map<T1, T2>& the_map)
{
    typedef map<T1, T2> Map_type;

    // static size of the map elements
#if defined(MI_PLATFORM_LINUX) || defined(MI_PLATFORM_MACOSX_USING_RB_TREE_NODE)
    size_t total = the_map.size() * sizeof(std::_Rb_tree_node<std::pair<const T1,T2> >);
#elif defined(MI_PLATFORM_MACOSX_USING_NODE)
    size_t total = the_map.size() * sizeof(std::__tree_node<std::pair<const T1,T2>, void*>);
#elif defined(MI_PLATFORM_WINDOWS)
    struct Sub : public Map_type { typedef Map_type::_Node _Node; };
    size_t total = the_map.size() * sizeof(Sub::_Node);
#else
#warning Unsupported platform
#endif

    // additional dynamic size of the map elements
    if (the_map.size() > 0) {
        bool dynamic_memory_T1 = has_dynamic_memory_consumption (the_map.begin()->first);
        bool dynamic_memory_T2 = has_dynamic_memory_consumption (the_map.begin()->second);
        if (dynamic_memory_T1 || dynamic_memory_T2) {
            typename Map_type::const_iterator it = the_map.begin();
            typename Map_type::const_iterator it_end = the_map.end();
            for (; it != it_end; ++it) {
                if (dynamic_memory_T1) total += dynamic_memory_consumption (it->first);
                if (dynamic_memory_T2) total += dynamic_memory_consumption (it->second);
            }
        }
    }

    return total;
}

MI_MEM_HAS_DYNAMIC_MEMORY_CONSUMPTION_TEMPLATE3(map<T1,T2,A>,T1,T2,A)

template < class T1 ,class T2, class A >
inline size_t dynamic_memory_consumption (const map<T1, T2, A>& the_map)
{
    typedef map<T1, T2, A> Map_type;

    // static size of the map elements
#if defined(MI_PLATFORM_LINUX) || defined(MI_PLATFORM_MACOSX_USING_RB_TREE_NODE)
    size_t total = the_map.size() * sizeof(std::_Rb_tree_node<std::pair<const T1,T2> >);
#elif defined(MI_PLATFORM_MACOSX_USING_NODE)
    size_t total = the_map.size() * sizeof(std::__tree_node<std::pair<const T1,T2>, void*>);
#elif defined(MI_PLATFORM_WINDOWS)
    size_t total = the_map.size() * sizeof(Map_type::_Node);
#endif

    // additional dynamic size of the map elements
    if (the_map.size() > 0) {
        bool dynamic_memory_T1 = has_dynamic_memory_consumption (the_map.begin()->first);
        bool dynamic_memory_T2 = has_dynamic_memory_consumption (the_map.begin()->second);
        if (dynamic_memory_T1 || dynamic_memory_T2) {
            typename Map_type::const_iterator it = the_map.begin();
            typename Map_type::const_iterator it_end = the_map.end();
            for (; it != it_end; ++it) {
                if (dynamic_memory_T1) total += dynamic_memory_consumption (it->first);
                if (dynamic_memory_T2) total += dynamic_memory_consumption (it->second);
            }
        }
    }

    return total;
}


// std::multimap

MI_MEM_HAS_DYNAMIC_MEMORY_CONSUMPTION_TEMPLATE2(multimap<T1,T2>,T1,T2)

template < class T1 ,class T2 >
inline size_t dynamic_memory_consumption (const multimap<T1, T2>& the_map)
{
    typedef multimap<T1, T2> Map_type;

    // static size of the multimap elements
#if defined(MI_PLATFORM_LINUX) || defined(MI_PLATFORM_MACOSX_USING_RB_TREE_NODE)
    size_t total = the_map.size() * sizeof(std::_Rb_tree_node<std::pair<const T1,T2> >);
#elif defined(MI_PLATFORM_MACOSX_USING_NODE)
    size_t total = the_map.size() * sizeof(std::__tree_node<std::pair<const T1,T2>, void*>);
#elif defined(MI_PLATFORM_WINDOWS)
    // subclass to get access to _Node type
    struct Sub : public Map_type { typedef Map_type::_Node _Node; };
    size_t total = the_map.size() * sizeof(Sub::_Node);
#endif

    // additional dynamic size of the multimap elements
    if (the_map.size() > 0) {
        bool dynamic_memory_T1 = has_dynamic_memory_consumption (the_map.begin()->first);
        bool dynamic_memory_T2 = has_dynamic_memory_consumption (the_map.begin()->second);
        if (dynamic_memory_T1 || dynamic_memory_T2) {
            typename Map_type::const_iterator it = the_map.begin();
            typename Map_type::const_iterator it_end = the_map.end();
            for (; it != it_end; ++it) {
                if (dynamic_memory_T1) total += dynamic_memory_consumption (it->first);
                if (dynamic_memory_T2) total += dynamic_memory_consumption (it->second);
            }
        }
    }

    return total;
}

MI_MEM_HAS_DYNAMIC_MEMORY_CONSUMPTION_TEMPLATE3(multimap<T1,T2,A>,T1,T2,A)

template < class T1 ,class T2, class A >
inline size_t dynamic_memory_consumption (const multimap<T1, T2, A>& the_map)
{
    typedef multimap<T1, T2, A> Map_type;

    // static size of the multimap elements
#if defined(MI_PLATFORM_LINUX) || defined(MI_PLATFORM_MACOSX_USING_RB_TREE_NODE)
    size_t total = the_map.size() * sizeof(std::_Rb_tree_node<std::pair<const T1,T2> >);
#elif defined(MI_PLATFORM_MACOSX_USING_NODE)
    size_t total = the_map.size() * sizeof(std::__tree_node<std::pair<const T1,T2>, void*>);
#elif defined(MI_PLATFORM_WINDOWS)
    size_t total = the_map.size() * sizeof(Map_type::_Node);
#endif

    // additional dynamic size of the multimap elements
    if (the_map.size() > 0) {
        bool dynamic_memory_T1 = has_dynamic_memory_consumption (the_map.begin()->first);
        bool dynamic_memory_T2 = has_dynamic_memory_consumption (the_map.begin()->second);
        if (dynamic_memory_T1 || dynamic_memory_T2) {
            typename Map_type::const_iterator it = the_map.begin();
            typename Map_type::const_iterator it_end = the_map.end();
            for (; it != it_end; ++it) {
                if (dynamic_memory_T1) total += dynamic_memory_consumption (it->first);
                if (dynamic_memory_T2) total += dynamic_memory_consumption (it->second);
            }
        }
    }

    return total;
}


// std::set

MI_MEM_HAS_DYNAMIC_MEMORY_CONSUMPTION_TEMPLATE(set<T>,T)

template <class T>
inline size_t dynamic_memory_consumption (const set<T>& the_set)
{
    typedef set<T> Set_type;

    // static size of the set elements
#if defined(MI_PLATFORM_LINUX) || defined(MI_PLATFORM_MACOSX_USING_RB_TREE_NODE)
    size_t total = the_set.size() * sizeof(std::_Rb_tree_node<T>);
#elif defined(MI_PLATFORM_MACOSX_USING_NODE)
    size_t total = the_set.size() * sizeof(std::__tree_node<T, void*>);
#elif defined(MI_PLATFORM_WINDOWS)
    // subclass to get access to _Node type
    struct Sub : public set<T> { typedef set<T>::_Node _Node; };
    size_t total = the_set.size() * sizeof(Sub::_Node);
#endif

    // additional dynamic size of the set elements
    if (the_set.size() > 0 && has_dynamic_memory_consumption (*(the_set.begin()))) {
        typename Set_type::const_iterator it = the_set.begin();
        typename Set_type::const_iterator it_end = the_set.end();
        for (; it != it_end; ++it)
            total += dynamic_memory_consumption (*it);
    }

    return total;
}

MI_MEM_HAS_DYNAMIC_MEMORY_CONSUMPTION_TEMPLATE2(set<T1,A>,T1,A)

template <class T1, class A>
inline size_t dynamic_memory_consumption (const set<T1, A>& the_set)
{
    typedef set<T1, A> Set_type;

    // static size of the set elements
#if defined(MI_PLATFORM_LINUX) || defined(MI_PLATFORM_MACOSX_USING_RB_TREE_NODE)
    size_t total = the_set.size() * sizeof(std::_Rb_tree_node<T1>);
#elif defined(MI_PLATFORM_MACOSX_USING_NODE)
    size_t total = the_set.size() * sizeof(std::__tree_node<T1, void*>);
#elif defined(MI_PLATFORM_WINDOWS)
    // subclass to get access to _Node type
    struct Sub : public Set_type { typedef Set_type::_Node _Node; };
    size_t total = the_set.size() * sizeof(Sub::_Node);
#endif

    // additional dynamic size of the set elements
    if (the_set.size() > 0 && has_dynamic_memory_consumption (*(the_set.begin()))) {
        typename Set_type::const_iterator it = the_set.begin();
        typename Set_type::const_iterator it_end = the_set.end();
        for (; it != it_end; ++it)
            total += dynamic_memory_consumption (*it);
    }

    return total;
}

} // MISTD


// boost::unordered_map

namespace boost {

namespace unordered {

MI_MEM_HAS_DYNAMIC_MEMORY_CONSUMPTION_TEMPLATE5(unordered_map<T1,T2,T3,T4,T5>,T1,T2,T3,T4,T5)

template < class T1 ,class T2, class T3, class T4, class T5 >
inline size_t dynamic_memory_consumption (const unordered_map<T1, T2, T3, T4, T5>& the_map)
{
    typedef unordered_map<T1, T2, T3, T4, T5> Map_type;

    typedef boost::unordered::detail::map<T5, T1, T2, T3, T4> Internal_map_type;

    // static size of the map elements (buckets are missing)
    size_t total = the_map.size() * sizeof(typename Internal_map_type::node);

    // additional dynamic size of the map elements
    if (the_map.size() > 0) {
        bool dynamic_memory_T1 = has_dynamic_memory_consumption (the_map.begin()->first);
        bool dynamic_memory_T2 = has_dynamic_memory_consumption (the_map.begin()->second);
        if (dynamic_memory_T1 || dynamic_memory_T2) {
            typename Map_type::const_iterator it = the_map.begin();
            typename Map_type::const_iterator it_end = the_map.end();
            for (; it != it_end; ++it) {
                if (dynamic_memory_T1) total += dynamic_memory_consumption (it->first);
                if (dynamic_memory_T2) total += dynamic_memory_consumption (it->second);
            }
        }
    }

    return total;
}

} // namespace unordered

} // namespace boost

#undef MI_PLATFORM_MACOSX_USING_RB_TREE_NODE
#undef MI_PLATFORM_MACOSX_USING_NODE


#endif // BASE_LIB_MEM_CONSUMPTION_H
