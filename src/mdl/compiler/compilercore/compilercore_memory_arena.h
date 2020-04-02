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

#ifndef MDL_COMPILERCORE_MEMORY_ARENA_H
#define MDL_COMPILERCORE_MEMORY_ARENA_H 1

#include "compilercore_cc_conf.h"

#include <mi/base/handle.h>

#include <string>
#include <vector>

#if MDL_STD_HAS_UNORDERED
#include <unordered_map>
#include <unordered_set>
#else
 // pre C++11 need boost headers
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#endif

#include <list>
#include <functional>

#include "compilercore_allocator.h"
#include "compilercore_assert.h"

namespace mi {
namespace mdl {

/// Implementation of the memory arena.
class Memory_arena
{
    typedef unsigned char Byte;

    struct Header {
        Header *next;
        size_t chunk_size;

        Byte load[1];
    };

    enum sizes {
        CHUNK_SIZE = 4096   ///< The default size of the arena memory chunks.
    };

public:
    /// Constructs a new memory arena.
    ///
    /// \param alloc       the allocator
    /// \param chunk_size  the size of the memory chunks allocated from alloc
    explicit Memory_arena(IAllocator *alloc, size_t chunk_size = CHUNK_SIZE);

    /// Destructs the memory arena and frees ALL memory.
    ~Memory_arena();

private:
    // non copyable
    Memory_arena(Memory_arena const &) MDL_DELETED_FUNCTION;
    Memory_arena &operator=(Memory_arena const &) MDL_DELETED_FUNCTION;

public:
    /// Allocates size bytes from the memory area.
    ///
    /// \param size   size to allocate
    /// \param align  alignment, 0 for structure alignment
    void *allocate(size_t size, size_t align = 0);

    /// Return the IAllocator interface used, not retained.
    ///
    /// \note Does NOT increase the reference count of the returned
    ///       module, do NOT decrease it just because of this call.
    IAllocator *get_allocator() const { return m_alloc.get(); }

    /// Drop the given object AND all later allocated objects from the arena.
    ///
    /// \param obj  the address of the object to drop, NULL drops the whole memory arena
    void drop(void *obj);

    /// Check if an object lies in this memory arena.
    ///
    /// \param obj  the address of an object
    ///
    /// \return true  if this object is owned by the memory arena,
    ///         false otherwise
    bool contains(void const *obj) const;

    /// Return the size of the allocated memory arena chunks.
    size_t get_chunks_size() const;

    /// Swap this memory arena content with another.
    void swap(Memory_arena &other);

private:

    /// The allocator.
    mi::base::Handle<IAllocator> m_alloc;

    /// The size of the chunks
    size_t m_chunk_size;

    /// The chunk list.
    Header *m_chunks;

    /// Pointer to the next free memory.
    Byte *m_next;

    /// size of the current chunk
    size_t m_curr_size;
};


/// A standards-compliant allocator using a Memory area.
template<typename T>
class Memory_arena_allocator
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
    struct rebind { typedef Memory_arena_allocator<U> other; };

    /// Creates a pooled allocator to the given pool.
    /// Non-explicit to simplify use.
    Memory_arena_allocator(Memory_arena *arena) : m_arena( arena ) 
    {
    }

    /// Creates a pooled allocator to the argument's pool.
    template<typename U>
    Memory_arena_allocator(Memory_arena_allocator<U> const& arg) : m_arena( arg.m_arena )
    {
    }

    /// The largest value that can meaningfully passed to allocate.
    size_type max_size() const { return 0xffffffff; }

    /// Memory is allocated for \c count objects of type \c T but objects are not constructed.
    pointer allocate( size_type count, std::allocator<void>::const_pointer /*hint*/ = 0 ) const
    {
        return reinterpret_cast<T*>(m_arena->allocate(count * sizeof(T)));
    }

    /// Deallocates memory allocated by allocate.
    void deallocate(pointer block, size_type count) const throw()
    {
    }

    /// Constructs an element of \c T at the given pointer.
    void construct(pointer element, T const& arg)
    {
        new( element ) T( arg );
    }

#ifdef MI_ENABLE_MISTD
    /// Constructs an element of \c U at the given pointer with given arguments
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
    Memory_arena *m_arena;
};

/// A specialization of the pooled allocator for the void type.
template<>
class Memory_arena_allocator<void>
{
public:
    typedef size_t      size_type;
    typedef ptrdiff_t   difference_type;

    typedef void        value_type;
    typedef void*       pointer;
    typedef void const* const_pointer;

    template<typename U> 
    struct rebind { typedef Memory_arena_allocator<U> other; };

    /// Creates a pooled allocator to the given pool.
    Memory_arena_allocator(Memory_arena *arena) : m_arena(arena) {}

    /// The memory arena for this allocator.
    Memory_arena *m_arena;
};

/// Returns true if objects allocated from one memory arena can be deallocated from the other.
template<typename T, typename U>
bool operator==( Memory_arena_allocator<T> const& left, Memory_arena_allocator<U> const& right )
{
    return left.m_arena == right.m_arena;
}

/// Returns true if objects allocated from one memory arena cannot be deallocated from the other.
template<typename T, typename U>
bool operator!=( Memory_arena_allocator<T> const& left, Memory_arena_allocator<U> const& right )
{
    return left.m_arena != right.m_arena;
}

/// Implementation of the Memory area builder.
///
/// To allocate an object on a memory arena, do:
///
/// Arena_builder builder(arena);
///
/// T *p = builder.create<T>(args);
///
/// As an equivalent to T *p = new T(args);
class Arena_builder
{
public:
    explicit Arena_builder(Memory_arena &arena)
    : m_arena(arena)
    {
    }

private:
    // non copyable
    Arena_builder(Arena_builder const &) MDL_DELETED_FUNCTION;
    Arena_builder &operator=(Arena_builder const &) MDL_DELETED_FUNCTION;

public:
    template<typename T>
    T *create() {
        if (void *p = m_arena.allocate(sizeof(T))) {
            return new(reinterpret_cast<T*>(p)) T;
        }
        return NULL;
    }

    template<typename T, typename A1>
    T *create(A1 a1) {
        if (void *p = m_arena.allocate(sizeof(T))) {
            return new(reinterpret_cast<T*>(p)) T(a1);
        }
        return NULL;
    }

    template<typename T, typename A1, typename A2>
    T *create(A1 a1, A2 a2) {
        if (void *p = m_arena.allocate(sizeof(T))) {
            return new(reinterpret_cast<T*>(p)) T(a1, a2);
        }
        return NULL;
    }

    template<typename T, typename A1, typename A2, typename A3>
    T *create(A1 a1, A2 a2, A3 a3) {
        if (void *p = m_arena.allocate(sizeof(T))) {
            return new(reinterpret_cast<T*>(p)) T(a1, a2, a3);
        }
        return NULL;
    }

    template<typename T, typename A1, typename A2, typename A3, typename A4>
    T *create(A1 a1, A2 a2, A3 a3, A4 a4) {
        if (void *p = m_arena.allocate(sizeof(T))) {
            return new(reinterpret_cast<T*>(p)) T(a1, a2, a3, a4);
        }
        return NULL;
    }

    template<typename T, typename A1, typename A2, typename A3, typename A4, typename A5>
    T *create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5) {
        if (void *p = m_arena.allocate(sizeof(T))) {
            return new(reinterpret_cast<T*>(p)) T(a1, a2, a3, a4, a5);
        }
        return NULL;
    }

    template<
        typename T, typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6>
        T *create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6) {
            if (void *p = m_arena.allocate(sizeof(T))) {
                return new(reinterpret_cast<T*>(p)) T(a1, a2, a3, a4, a5, a6);
            }
            return NULL;
    }

    template<
        typename T, typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6, typename A7>
        T *create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7) {
            if (void *p = m_arena.allocate(sizeof(T))) {
                return new(reinterpret_cast<T*>(p)) T(a1, a2, a3, a4, a5, a6, a7);
            }
            return NULL;
    }

    template<
        typename T, typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6, typename A7, typename A8>
        T *create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8) {
            if (void *p = m_arena.allocate(sizeof(T))) {
                return new(reinterpret_cast<T*>(p)) T(a1, a2, a3, a4, a5, a6, a7, a8);
            }
            return NULL;
    }

    template<
        typename T, typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6, typename A7, typename A8, typename A9>
        T *create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9) {
            if (void *p = m_arena.allocate(sizeof(T))) {
                return new(reinterpret_cast<T*>(p)) T(a1, a2, a3, a4, a5, a6, a7, a8, a9);
            }
            return NULL;
    }

    template<
        typename T, typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6, typename A7,
        typename A8, typename A9, typename A10>
        T *create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10) {
            if (void *p = m_arena.allocate(sizeof(T))) {
                return new(reinterpret_cast<T*>(p)) T(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
            }
            return NULL;
    }

    template<
        typename T, typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6, typename A7,
        typename A8, typename A9, typename A10, typename A11>
        T *create(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10, A11 a11) {
            if (void *p = m_arena.allocate(sizeof(T))) {
                return new(reinterpret_cast<T*>(p)) T(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
            }
            return NULL;
    }

    /// allocate n elements of type T
    template<typename T>
    T *alloc(Size n_elems = 1) { return (T *)m_arena.allocate(sizeof(T) * n_elems); }

    /// get the memory arena
    inline Memory_arena *get_arena() const { return &m_arena; }

    /// The allocator.
    Memory_arena &m_arena;
};

/// A VLA using a memory arena.
/// \tparam T  element type, must be a POD.
template<typename T>
class Arena_VLA {
public:
    /// Construct a new VLA of size n, allocated an the allocator alloc.
    Arena_VLA(Memory_arena &arena, size_t n)
    : m_data(reinterpret_cast<T*>(arena.allocate(sizeof(T) * n)))
    , m_size(n)
    {
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

private:
    T * const    m_data;
    size_t const m_size;
};

typedef std::basic_string<
    char, 
    std::char_traits<char>, 
    Memory_arena_allocator<char> > Arena_string;

// Helper for dynamic memory consumption: Arena strings have no EXTRA memory allocated.
inline bool has_dynamic_memory_consumption(Arena_string const &) { return false; }
inline size_t dynamic_memory_consumption(Arena_string const &) { return 0; }


template<typename T>
struct Arena_vector
{
    typedef std::vector<T, Memory_arena_allocator<T> > Type;
};

// Helper for dynamic memory consumption: Arena vectors have no EXTRA memory allocated.
template<typename T>
inline bool has_dynamic_memory_consumption(std::vector<T, Memory_arena_allocator<T> > const &) {
    return false;
}
template<typename T>
inline size_t dynamic_memory_consumption(std::vector<T, Memory_arena_allocator<T> > const &) {
    return 0;
}

#if MDL_STD_HAS_UNORDERED
template <
    typename Key,
    typename Tp,
    typename HashFcn = std::hash<Key>,
    typename EqualKey = std::equal_to<Key>
>
struct Arena_hash_map {
    typedef std::unordered_map<
        Key, Tp, HashFcn, EqualKey,
        Memory_arena_allocator<
        typename std::unordered_map<Key, Tp, HashFcn, EqualKey>::value_type>
    > Type;
};

template <
    typename Key,
    typename Tp,
    typename HashFcn = Hash_ptr<Key>,
    typename EqualKey = Equal_ptr<Key>
>
struct Arena_ptr_hash_map {
    typedef std::unordered_map<
        Key *, Tp, HashFcn, EqualKey,
        Memory_arena_allocator<
        typename std::unordered_map<Key *, Tp, HashFcn, EqualKey>::value_type>
    > Type;
};

// Helper for dynamic memory consumption: Arena hash maps have no EXTRA memory allocated.
template<typename T1, typename T2, typename T3, typename T4>
inline bool has_dynamic_memory_consumption(
    std::unordered_map<T1, T2, T3, T4, Memory_arena_allocator<std::pair<T1, T2> > > const &)
{
    return false;
}
template<typename T1, typename T2, typename T3, typename T4>
inline size_t dynamic_memory_consumption(
    std::unordered_map<T1, T2, T3, T4, Memory_arena_allocator<std::pair<T1, T2> > >  const &)
{
    return 0;
}

template <
    typename Key,
    typename HashFcn = std::hash<Key>,
    typename EqualKey = std::equal_to<Key>
>
struct Arena_hash_set {
    typedef std::unordered_set<
        Key, HashFcn, EqualKey, Memory_arena_allocator<Key> > Type;
};

template <
    typename Key,
    typename HashFcn = Hash_ptr<Key>,
    typename EqualKey = Equal_ptr<Key>
>
struct Arena_ptr_hash_set {
    typedef std::unordered_set<
        Key *, HashFcn, EqualKey, Memory_arena_allocator<Key *> > Type;
};

// Helper for dynamic memory consumption: Arena hash sets have no EXTRA memory allocated.
template<typename T1, typename T2, typename T3>
inline bool has_dynamic_memory_consumption(
    std::unordered_set<T1, T2, T3, Memory_arena_allocator<T1> > const &)
{
    return false;
}
template<typename T1, typename T2, typename T3>
inline size_t dynamic_memory_consumption(
    std::unordered_set<T1, T2, T3, Memory_arena_allocator<T1> > const &)
{
    return 0;
}

#else

template <
    typename Key,
    typename Tp,
    typename HashFcn = boost::hash<Key>,
    typename EqualKey = std::equal_to<Key>
>
struct Arena_hash_map {
    typedef boost::unordered_map<
        Key, Tp, HashFcn, EqualKey,
        Memory_arena_allocator<
            typename boost::unordered_map<Key, Tp, HashFcn, EqualKey>::value_type>
    > Type;
};

template <
    typename Key,
    typename Tp,
    typename HashFcn = Hash_ptr<Key>,
    typename EqualKey = Equal_ptr<Key>
>
struct Arena_ptr_hash_map {
    typedef boost::unordered_map<
        Key *, Tp, HashFcn, EqualKey,
        Memory_arena_allocator<
            typename boost::unordered_map<Key *, Tp, HashFcn, EqualKey>::value_type>
    > Type;
};

// Helper for dynamic memory consumption: Arena hash maps have no EXTRA memory allocated.
template<typename T1, typename T2, typename T3, typename T4>
inline bool has_dynamic_memory_consumption(
    boost::unordered_map<T1,T2,T3,T4,Memory_arena_allocator<std::pair<T1,T2> > > const &)
{
    return false;
}
template<typename T1, typename T2, typename T3, typename T4>
inline size_t dynamic_memory_consumption(
    boost::unordered_map<T1,T2,T3,T4,Memory_arena_allocator<std::pair<T1,T2> > >  const &)
{
    return 0;
}

template <
    typename Key,
    typename HashFcn = boost::hash<Key>,
    typename EqualKey = std::equal_to<Key>
>
struct Arena_hash_set {
    typedef boost::unordered_set<
        Key, HashFcn, EqualKey, Memory_arena_allocator<Key> > Type;
};

template <
    typename Key,
    typename HashFcn = Hash_ptr<Key>,
    typename EqualKey = Equal_ptr<Key>
>
struct Arena_ptr_hash_set {
    typedef boost::unordered_set<
        Key *, HashFcn, EqualKey, Memory_arena_allocator<Key *> > Type;
};

// Helper for dynamic memory consumption: Arena hash sets have no EXTRA memory allocated.
template<typename T1, typename T2, typename T3>
inline bool has_dynamic_memory_consumption(
    boost::unordered_set<T1,T2,T3,Memory_arena_allocator<T1> > const &)
{
    return false;
}
template<typename T1, typename T2, typename T3>
inline size_t dynamic_memory_consumption(
    boost::unordered_set<T1,T2,T3,Memory_arena_allocator<T1> > const &)
{
    return 0;
}
#endif

/// A list using a memory arena.
template <typename Tp>
struct Arena_list {
    typedef std::list<Tp, Memory_arena_allocator<Tp> > Type;
};

// Helper for dynamic memory consumption: Arena lists have no EXTRA memory allocated.
template<typename T>
inline bool has_dynamic_memory_consumption(std::list<T,Memory_arena_allocator<T> > const &) {
    return false;
}
template<typename T>
inline size_t dynamic_memory_consumption(std::list<T,Memory_arena_allocator<T> > const &) {
    return 0;
}

/// Put a C-string into the memory arena.
///
/// \param arena  the memory arena
/// \param s      the C-string
char *Arena_strdup(Memory_arena &arena, char const *s);

/// Put a blob into the memory arena.
///
/// \param arena  the memory arena
/// \param mem    start address of the blob
/// \param size   size of the blob
void *Arena_memdup(Memory_arena &arena, void const *mem, size_t size);

}  // mdl
}  // mi

#endif // MDL_COMPILERCORE_MEMORY_ARENA_H
