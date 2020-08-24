/***************************************************************************************************
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
 **************************************************************************************************/

/// \file
/// \brief The implementation of the ATTR::Type inline members

#include <base/lib/log/i_log_assert.h>
#include <base/lib/mem/i_mem_consumption.h>

namespace MI {
namespace ATTR {

// return true if the two types agree exactly, including all child types. This
// is useful to check that the values are compatible, for example for shader
// parameter inheritance and assignment. Names are not compared.
MI_INLINE bool Type::operator==(
    const Type	&other) const
{
    if (m_typecode != other.m_typecode	||
        m_const    != other.m_const	||
        m_spare    != other.m_spare)

        return false;                           // toplevel type mismatch

    if (m_arraysize != other.m_arraysize)
        return false;                           // toplevel arraysize mismatch

    if (m_typecode == TYPE_ENUM) {
        if (bool(m_enum) != bool(other.m_enum))
            return false;                       // only one has an enum collections
        if (!m_enum)
            return true;
        return compare_enum_collections(*m_enum, *other.m_enum);
    }

    if (bool(m_child) != bool(other.m_child))
        return false;                           // only one has a child

    if (!m_child)
        return true;                            // nonstruct: same type

    const Type *tt = get_child();               // struct: compare child chain
    const Type *to = other.get_child();
    for (; tt && to; tt=tt->get_next(), to=to->get_next())
    if (!(*tt == *to))                          // child mismatch (recursion)
        return false;
    return !tt && !to;                          // same number of children

}


MI_INLINE bool Type::operator!=(
    const Type	&other) const
{
    return !(*this == other);
}


// read access functions
MI_INLINE const char *Type::get_name() const
{
    return m_name.empty()? 0 : m_name.c_str();
}


MI_INLINE Type_code Type::get_typecode() const
{
    return (Type_code)m_typecode;
}


MI_INLINE bool Type::get_const() const
{
    return m_const;
}


MI_INLINE Uint Type::get_arraysize() const
{
    return m_arraysize;
}


MI_INLINE const Type *Type::get_next() const
{
    return m_next;
}

MI_INLINE const Type *Type::get_child() const
{
    return m_child;
}

MI_INLINE Type *Type::get_next()
{
    return m_next;
}


MI_INLINE Type* Type::get_child()
{
    ASSERT(M_ATTR,
        get_typecode() == TYPE_STRUCT ||
        get_typecode() == TYPE_ARRAY ||
        get_typecode() == TYPE_ATTACHABLE ||
        get_typecode() == TYPE_CALL);
    return m_child;
}


MI_INLINE std::vector<std::pair<int, std::string> >** Type::set_enum()
{
    ASSERT(M_ATTR, m_typecode == TYPE_ENUM);
    return &m_enum;
}


MI_INLINE std::vector<std::pair<int, std::string> >* Type::get_enum() const
{
    ASSERT(M_ATTR, m_typecode == TYPE_ENUM);
    return m_enum;
}


// write access functions. Setting next/child deletes the entire old chain.
MI_INLINE void Type::set_typecode(
    Type_code		typecode)
{
    ASSERT(M_ATTR, typecode != TYPE_ID);
    m_typecode = typecode;
}


MI_INLINE void Type::set_const(
    bool		isconst)
{
    m_const = isconst;
}


MI_INLINE void Type::set_arraysize(
    Uint arraysize)
{
    // this method is only useful called on TYPE_ARRAY types
    ASSERT(M_ATTR, get_typecode() == TYPE_ARRAY);

    if (get_typecode() == TYPE_ARRAY) {
        m_arraysize = arraysize;
        // currently all static-array unaware code works on the array's elements, hence
        // we have to guarantee that those have the same array size. Except for arrays of
        // arrays...which are not supported! (Hence the following assertion. But we will
        // let it through for now.)
        //ASSERT(M_ATTR, !(m_child && m_child->get_typecode() == TYPE_ARRAY));
        if (m_child && m_child->get_typecode() != TYPE_ARRAY)
            m_child->m_arraysize = arraysize;
    }
}


// return the alignment of one member of this type. The alignment is an address
// multiple where a compiler would store a value of this type. For example, an
// integer is always stored at an address that is a multiple of 4, so return 4.
MI_INLINE size_t Type::align_one() const
{
    size_t s = m_arraysize ? m_typeinfo[m_typecode].size
                           : sizeof(Dynamic_array *);
    ASSERT(M_ATTR, (s|(s-1))+1 == s+s); // may have only one bit set, or 0
    return s ? s : 1;
}


// same thing as align_one, but returns the alignment of the entire (sub)tree.
// This is the alignment of the largest simple element of the tree. For example
// a nested struct that has a double anywhere in it must itself have the
// alignment of a double (8).
MI_INLINE size_t Type::align_all() const
{
    if (m_typecode != TYPE_STRUCT && m_typecode != TYPE_ARRAY && m_typecode != TYPE_ATTACHABLE
        && m_typecode != TYPE_CALL)
    {
       return align_one();
    }

    const Type* start_type = m_child;
    // handle special case of an array of structs
    if (m_typecode == TYPE_ARRAY && m_child &&
        (m_child->get_typecode() == TYPE_STRUCT || m_child->get_typecode() == TYPE_ATTACHABLE
        || m_child->get_typecode() == TYPE_CALL))
    {
        start_type = m_child->m_child;
    }

    size_t align = 1;
    // in the TYPE_CALL case, set initial alignment to the two string pointers
    if (m_typecode == TYPE_CALL || 
        (m_typecode == TYPE_ARRAY && m_child && m_child->get_typecode() == TYPE_CALL))
    {
        bool is_dynamic_array = m_typecode != TYPE_ARRAY && m_arraysize == 0;
        align = is_dynamic_array? sizeof(Dynamic_array*) : m_typeinfo[TYPE_STRING].size;
    }
    for (const Type* t=start_type; t; t=t->m_next) {
        size_t subalign = t->align_all();
        if (subalign > align)
            align = subalign;
    }
    return align;
}


// return the size of one member of this type. For example, a color is 16
// bytes, and a static array of three of those is 48 bytes. Dynamic arrays
// are always Dynamic_array blocks that contain a pointer to the actual data;
// the size of the struct is returned but not the actual arry pointed to. Note
// that Dynamic_array has a null int to make the alignment predictable on
// 64-bit machines. Structs have size 0 since their members are not included.
MI_INLINE size_t Type::sizeof_one() const
{
    return m_arraysize
        ? m_arraysize * m_typeinfo[m_typecode].comp
                      * m_typeinfo[m_typecode].size
        : sizeof(Dynamic_array);
}


// return the ascii name of a type. Useful for all sorts of user messages.
MI_INLINE const char *Type::type_name() const
{
    return m_typeinfo[m_typecode].name;
}


// return info on how a type is put together from components. Ie, a Color is
// 4 components, each of which is a Scalar. This helps code like image/image
// and serializers that can get away with far smaller switches.
MI_INLINE Uint Type::component_count() const
{
    if (m_typecode != TYPE_ARRAY)
        return m_typeinfo[m_typecode].comp;
    else {
        // what to return for a TYPE_ARRAY? The number of array elements? Or the component_count
        // of the array elements' type? But what if these are of TYPE_ARRAY type again?
        // For now I opt for the array's elements' type or 0 when this is another TYPE_ARRAY.
        if (!m_child)
            return 0;
        if (m_child->get_typecode() != TYPE_ARRAY)
            return m_child->component_count();
        else
            return 0;
    }
}


MI_INLINE Type_code Type::component_type() const
{
    if (m_typecode != TYPE_ARRAY)
        return m_typeinfo[m_typecode].base;
    else {
        if (!m_child)
            return TYPE_UNDEF;
        // TYPE_ARRAY of TYPE_ARRAY is not supported but we will let this through
        //ASSERT(M_ATTR, m_child->get_typecode() != TYPE_ARRAY);
        if (m_child->get_typecode() != TYPE_ARRAY)
            return m_child->component_type();
        else
            return TYPE_ARRAY;
    }
}


// Return the approximate size in bytes of the element including all its
// substructures. This is used to make decisions about garbage collection.
// Include the size of the subtree anchored here; it would be messy to ask
// the caller to do that.
MI_INLINE size_t Type::get_size() const
{
    size_t child_size = 0;
    if (m_typecode != TYPE_ENUM)
        child_size = m_child ? m_child->get_size() : 0;
    else {
        if (m_enum)
            child_size += std::dynamic_memory_consumption(*m_enum);
    }
    return sizeof(*this) + child_size
                         + (m_next  ? m_next ->get_size() : 0);
}


// unique class ID so that the receiving host knows which class to create
MI_INLINE SERIAL::Class_id Type::get_class_id() const
{
    return id;
}


MI_INLINE size_t Type::sizeof_all() const
{
    // dynamic arrays without top-level TYPE_ARRAY node
    if (get_typecode() != TYPE_ARRAY && m_arraysize == 0)
        return sizeof(Dynamic_array);
    // dynamic arrays with top-level TYPE_ARRAY node
    if (get_typecode() == TYPE_ARRAY && get_child() && get_child()->get_arraysize() == 0)
        return sizeof(Dynamic_array);
    if (get_typecode() != TYPE_ARRAY || (get_child() && get_child()->get_typecode() != TYPE_ARRAY))
        // note that the following code will NOT work for arrays of arrays
        return sizeof_elem() * m_arraysize;
    else
        return get_child()? m_arraysize * get_child()->sizeof_all() : 0;
}


// size of one element in an array
MI_INLINE size_t Type::sizeof_elem() const
{
    if (get_typecode() == TYPE_ARRAY) {
        if (get_child())
            return get_child()->sizeof_elem();
    }
    Type_code tc = get_typecode();
    if (tc != TYPE_STRUCT && tc != TYPE_ATTACHABLE && tc != TYPE_CALL)
        return sizeof_one(tc);

    size_t total = 0;
    size_t align = 1;
    // add the two string pointers in case of TYPE_CALL - actually is one Tag and one pointer now,
    // but using the original two char * makes it easier
    if (tc == TYPE_CALL) {
        total += 2*Type::sizeof_one(TYPE_STRING);
        align = Type::sizeof_one(TYPE_STRING);
    }
    for (const Type *t=get_child(); t; t=t->get_next()) {
        size_t subalign = t->align_all();
        if (subalign > align)				// align by largest
            align = subalign;
        total  = (total + subalign-1) & (~subalign+1);	// align new child
        total += t->sizeof_all();			// add new child
    }
    total = (total + align-1) & (~align+1);		// pad end of struct
    return total;
}


MI_INLINE const Type* Type::lookup(
    const char* name,
    char* base_address,
    char** ret_address,
    Uint offs) const
{
    const char** const_ret_address = const_cast<const char**>(ret_address);
    return lookup(name, base_address, const_ret_address, offs);
}

}
}
