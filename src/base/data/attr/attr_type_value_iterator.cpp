/***************************************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The implementation of the Type_value_iterator.
///
/// And another iterator iterating over a ATTR::Type and its value.

#include "pch.h"

#include "i_attr_type_value_iterator.h"
#include "i_attr_type_named_value_iterator.h"

#include "attr.h"
#include <base/lib/log/log.h>

namespace MI {
namespace ATTR {

using namespace MI::LOG;

//--------------------------------------------------------------------------------------------------

/// Retrieve the number of elements in its array. This local function abstracts from having to deal
/// with static and dynamic arrays.
/// \param typ the current \c Type
/// \param value_ptr its value_ptr
/// \return the number of elements in an array /* minus 1 (which eases internal handling) */
size_t get_count(
    const Type* typ,
    const char* value_ptr)
{
    if (typ->get_arraysize() == 0 && typ->get_typecode() != TYPE_ARRAY) {
        const ATTR::Dynamic_array* array = (const ATTR::Dynamic_array*)value_ptr;
        return array->m_count;
    }
    else
        return typ->get_arraysize();
}

// Tricky tricky.
// Handling
// - a new element will be made current via the call to set_type()
// - set_type() checks whether this starts a struct or an array and deals with this appropriately
//
// Handling of structs
// - when a struct is hit (via set_type())
//


//==================================================================================================

//--------------------------------------------------------------------------------------------------

// Constructor.
Type_value_iterator::Type_value_iterator(
    const Type* type,					// the type
    const char* values)					// the values
  : m_value_ptr(values),
    m_offset(0),
    m_is_struct_start(false),
    m_is_struct_end(false),
    m_is_array_start(false),
    m_is_array_end(false)
{
    set_type(type);
}

Type_value_iterator::Type_value_iterator()
  : m_value_ptr(0),
    m_offset(0),
    m_is_struct_start(false),
    m_is_struct_end(false),
    m_is_array_start(false),
    m_is_array_end(false)
{}


//--------------------------------------------------------------------------------------------------

// Destructor.
Type_value_iterator::~Type_value_iterator()
{}


//--------------------------------------------------------------------------------------------------

// Retrieve the current type. This is the one on top of the stack.
const Type* Type_value_iterator::get_type() const
{
    if (!m_types_stack.empty())
        return m_types_stack.top().first;
    else
        return 0;
}

//--------------------------------------------------------------------------------------------------
bool Type_value_iterator::is_valid() const
{
    return get_type() != 0;
}

//--------------------------------------------------------------------------------------------------

// Retrieve the current value.
const char* Type_value_iterator::get_value() const
{
    if (is_valid()) {
        // handle both static arrays and single values, and even dynamic arrays
        if (get_type()->get_typecode() != TYPE_STRING) {
            return get_value_ptr() + get_offset();
        }
        // check for dynamic arrays of strings
        else if (is_dyn_array()) {
            char* v = const_cast<char*>(get_value_ptr() + get_offset());
            Dynamic_array* a = (Dynamic_array*)v;
            return a? a->m_value : 0;
        }
        else {
            return *(reinterpret_cast<const char**>(
                const_cast<char*>(get_value_ptr() + get_offset())));
        }
    }
    else
        return 0;
}


//--------------------------------------------------------------------------------------------------

// Retrieve active arraysize. This function abstracts from static or dynamic arrays, for
// non-array type elements this would return 1.
int Type_value_iterator::get_arraycount() const
{
    if (!m_types_stack.empty()) {
        // check for special case of empty dynamic arrays where we have a 0 on top as well
        if (m_types_stack.top().second == 0 && m_types_stack.top().first->get_arraysize() == 0) {
            const Dynamic_array* array = (const Dynamic_array*)get_value();
            if (!array || array->m_count == 0)
                return 0;
            else
                return m_types_stack.top().second + 1;
        }
        else
            return m_types_stack.top().second + 1;
    }
    else
        return 0;
}


//--------------------------------------------------------------------------------------------------

// Are we at the beginning of a struct?
bool Type_value_iterator::is_struct_start() const
{
    return m_is_struct_start;
}


//--------------------------------------------------------------------------------------------------

// Are we at the end of a struct?
bool Type_value_iterator::is_struct_end() const
{
    return m_is_struct_end;
}


//--------------------------------------------------------------------------------------------------

// Are we at the beginning of an array?
bool Type_value_iterator::is_array_start() const
{
    return m_is_array_start;
}


//--------------------------------------------------------------------------------------------------

// Are we at the end of an array?
bool Type_value_iterator::is_array_end() const
{
    return m_is_array_end;
}

//--------------------------------------------------------------------------------------------------
bool Type_value_iterator::is_leaf_value() const
{
    return !(m_is_array_start || m_is_array_end || m_is_struct_start || m_is_struct_end);
}

//--------------------------------------------------------------------------------------------------

// Proceed one element further - which might then be null. This function takes care of alignment
// issues and keeps \c m_offset up to date. Until its very end this function will always have
//   get_type() != 0
// because it is using a stack.
// Dynamic and static arrays are treated here similiar. The element on top of the stack has a
// counter attached which gets incremented as long it is greater than 0.
void Type_value_iterator::progress()
{
    ASSERT(M_ATTR, get_type());			// it is NOT allowed to go past this!
    ASSERT(M_ATTR,
        get_type()->get_typecode() != TYPE_ATTACHABLE &&
        get_type()->get_typecode() != TYPE_CALL);

    // handle closing arrays (this is the increment *after* hitting the end of an array) by
    // setting is_array_end to true and handle already next type
    if (is_array_end())
        m_is_array_end = false;
    // when we found an end of an array, set is_array_end to true and return
    else if (found_end_of_array()) {
        m_is_array_end = true;
        return;
    }

    // add the current type's size to m_offset and align m_offset both prior and after that
    // Note: This holds only for single types. For arrays, the alignment of the elements
    // is the right alignment unit!!
    size_t align = 1;
    size_t sub_align = get_type()->align_all();
    if (get_type()->get_arraysize() != 1 || get_type()->get_typecode() == TYPE_ARRAY)
        sub_align = std::max((size_t)1, Type::sizeof_one(get_type()->get_typecode()));
    align = std::max(align, sub_align);

    size_t offset = (get_offset() + sub_align-1) & (~sub_align+1);
    if (get_type()->get_typecode() != TYPE_STRUCT) {
        if (get_type()->get_arraysize() == 1 && get_type()->get_typecode() != TYPE_ARRAY)
            offset += get_type()->sizeof_one();
        else {
            // PVS does not realize that the set_type() call below affects subsequent get_type()
            // calls which need to be checked for NULL even though it is not necessary here
            offset += get_type()->sizeof_elem(); //-V595
        }
    }
    offset += (align-1) & (~align+1);
    set_offset(offset);

    // if neither struct nor (non-empty) array - pop and advance to next type
    if (get_type()->get_typecode() != TYPE_STRUCT && m_types_stack.top().second == 0) {
        // when it was a dynamic array, increase offset accordingly
        if (is_dyn_array()) {
            m_dyn_values.pop();
            m_dyn_offsets.pop();

            offset = get_offset() + sizeof(ATTR::Dynamic_array);
            offset = (offset + align-1) & (~align+1);
            set_offset(offset);
        }
        const Type* type = get_type();
        m_types_stack.pop();

        // make increment permanent
        set_type(type->get_next());

        // add alignment of new type to the offset
        align = std::max<size_t>(get_type()? get_type()->align_all() : 1, 1);
        offset = (get_offset() + align-1) & (~align+1);
        set_offset(offset);
    }
    // if struct - this should never happen, since it should be handled in do_increment()
    else if (get_type()->get_typecode() == TYPE_STRUCT) {
        ASSERT(M_ATTR, 0);
    }
    // if (non-empty) array advance to next array element
    else
        handle_array();
}


//--------------------------------------------------------------------------------------------------

// Dereferencing the iterator.
const Type& Type_value_iterator::dereference() const
{
    return *get_type();
}


//--------------------------------------------------------------------------------------------------

// Managing the actual incrementation of the iterator. Handle as many special cases here, such that
// progress hasn't to deal with all of them.
void Type_value_iterator::do_increment()
{
    if (!get_type())
        return;

    if (is_struct_start()) {
        // special handling for TYPE_ARRAY
        if (get_type()->get_typecode() != TYPE_ARRAY)
            set_type(get_type()->get_child());
        else {
            // we have now a TYPE_ARRAY containing a TYPE_STRUCT, so deal with the struct's child
            ASSERT(M_ATTR, get_type()->get_child()->get_child());
            set_type(get_type()->get_child()->get_child());
        }
        return;
    }
    else if (is_struct_end()) {
        handle_closing_struct();
        return;
    }
    else if (is_array_start()) {
        m_is_array_start = false;
        ASSERT(M_ATTR,
            get_type()->get_typecode() != TYPE_ATTACHABLE &&
            get_type()->get_typecode() != TYPE_CALL);
        if (get_type()->get_typecode() == TYPE_STRUCT)
            m_is_struct_start = true;

        // handle dynamic arrays here
        if (is_dyn_array()) {
            const Dynamic_array* array = (const Dynamic_array*)get_value();
            // we don't check whether dynamic array is empty or not - since in progress()
            // those values will be pop-ed nevertheless
            m_dyn_values.push(array? array->m_value : 0);
            m_dyn_offsets.push(0);

            // if we have the case of adding an empty Dynamic_array we set the m_is_array_end
            // right after incrementing from m_is_array_stack
            if (!array || !array->m_count) {
                // BUT if we are in m_is_struct_start, ie having an empty array of structs, we
                // have to reset m_is_struct_start first
                m_is_struct_start = false;
                m_is_array_end = true;
            }
        }
        else if (get_type()->get_typecode() == TYPE_ARRAY) {
            ASSERT(M_ATTR, get_type()->get_child() == 0 ||
                get_type()->get_child()->get_typecode() != TYPE_ATTACHABLE ||
                get_type()->get_child()->get_typecode() != TYPE_CALL);
            if (get_type()->get_child() && get_type()->get_child()->get_typecode() == TYPE_STRUCT)
                m_is_struct_start = true;
        }
        return;
    }
    else if (is_array_end()) {
        ASSERT(M_ATTR,
            get_type()->get_typecode() != TYPE_ATTACHABLE &&
            get_type()->get_typecode() != TYPE_CALL);
        if (get_type()->get_typecode() == TYPE_STRUCT) {
            handle_closing_struct();
            return;
        }
    }

    progress();
}


//--------------------------------------------------------------------------------------------------

// Incrementing the iteration.
void Type_value_iterator::increment()
{
    // it is NOT allowed to go past this!
    ASSERT(M_ATTR, get_type());
    pre_increment();
    do_increment();
    post_increment();
}


//--------------------------------------------------------------------------------------------------

// Compare for equality.
bool Type_value_iterator::equal(const Type_value_iterator& it) const
{
    // special case for empty dynamic arrays - otherwise the following test against end returns true
    if (!m_dyn_values.empty() && m_dyn_values.top() == 0 && m_dyn_offsets.top() == 0) {
        if (it.m_dyn_values.empty())
            return false;
        if (it.m_dyn_values.top() != 0)
            return false;
        if (it.m_dyn_offsets.top() != 0)
            return false;
        return true;
    }
    return
        (!get_value() && !it.get_value()) ||		// for testing against end
        (m_value_ptr == it.m_value_ptr && m_offset == it.m_offset);
}


//--------------------------------------------------------------------------------------------------

// Set the given Type \c type. If the given type is 0 try to use the stack to come up with
// the next type. This happens when in a struct the last type was iterated over, its m_next == 0,
// or, at the very end.
void Type_value_iterator::set_type(
    const Type* typ,					// the type
    bool in_struct)					// are we in a struct
{
    if (typ) {
        ASSERT(M_ATTR,
            typ->get_typecode() != TYPE_ATTACHABLE &&
            typ->get_typecode() != TYPE_CALL);
        // the new type will be at the next aligned address - this will be done permanentely
        // in process() (at the very bottom), but for accessing get_count() we require already
        // the correct offset. Needs refactoring.
        size_t align = std::max<size_t>(typ->align_all(), 1);
        size_t offset = (get_offset() + align-1) & (~align+1);
        // at the end of a struct - which is the case when called with in_struct == true - set
        // the offset such that it points to the next element's value
        if (in_struct)
            set_offset(offset);

        // get_count() was returning size-1, but for empty arrays this leads to overflow
        size_t count = get_count(typ, get_value_ptr()+offset);
        if (count)
            count -= 1;
        m_types_stack.push(std::make_pair(typ, count));
        // first array, then struct!! Thatswhy the next if () is not in an else branch - if we have
        // an array of structs, we iterate from array_start to struct_start.
        if (typ->get_arraysize() != 1 || typ->get_typecode() == TYPE_ARRAY) {
            if (!is_array_start())
                m_is_array_start = true;
            else
                m_is_array_start = false;
        }
        if (typ->get_typecode() == TYPE_STRUCT && !is_array_start())
            m_is_struct_start = true;
        else
            m_is_struct_start = false;
    }
    else {
        // handle the struct on the stack
        if (!m_types_stack.empty()) {
            ASSERT(M_ATTR, get_type()->get_typecode() == TYPE_STRUCT ||
                get_type()->get_typecode() == TYPE_ARRAY);
            ASSERT(M_ATTR, !is_struct_start());
            if (in_struct) {
                // if during removal of the last struct we find another struct ... go into
                // end-of-struct mode unless this is of array type. why is the other struct
                // finished? because otherwise typ would had been set to its successor, but it is 0
//		if (handle_array())
//		    m_is_struct_start = true;
//		else
                    m_is_struct_end = true;
            }
            else {
                ASSERT(M_ATTR, !is_struct_end());
                // introduce a null round for this case
                m_is_struct_end = true;
            }
        }
    }
}


//--------------------------------------------------------------------------------------------------

// Debug helper.
void Type_value_iterator::dump() const
{
    mod_log->info(M_ATTR, Mod_log::C_IO, "Current iterator state");

    if (m_types_stack.empty())
        mod_log->debug(M_ATTR, Mod_log::C_IO, "\tno further items on the stack");
    else {
        const Values& values = m_types_stack.top();
        Type_code code = values.first->get_typecode();
        mod_log->debug(M_ATTR, Mod_log::C_IO, "\t%s, %" FMT_SIZE_T " elements left",
            Type::component_name(code), values.second);
    }
}


//--------------------------------------------------------------------------------------------------

// Handle potential array elements. If the element on top of the stack is a non-empty array,
// decrease the number of remaining array elements.
bool Type_value_iterator::handle_array()
{
    if (m_types_stack.empty() || m_types_stack.top().second == 0)
        return false;

    // decrease active count
    --m_types_stack.top().second;

    return true;
}


//--------------------------------------------------------------------------------------------------

// Handle closing struct. Introduce a "null round" for ending structs by removing this struct.
void Type_value_iterator::handle_closing_struct()
{
    ASSERT(M_ATTR, !is_struct_start());

    m_is_struct_end = false;

    // is an array of structs - simply continuing with the next struct
    if (handle_array()) {
        m_is_struct_start = true;
        return;
    }
    // check whether we have reached the end of the array (for the first time)
    else if (!m_is_array_end && m_types_stack.top().second == 0
        && (m_types_stack.top().first->get_arraysize() != 1
            || m_types_stack.top().first->get_typecode() == TYPE_ARRAY)) {
        m_is_array_end = true;
        return;
    }
    m_is_array_end = false;

    ASSERT(M_ATTR, m_types_stack.top().second == 0);
    // check this struct...and then remove it
    const Type* type = get_type()->get_next();
    m_types_stack.pop();

    set_type(type, true);
}


//--------------------------------------------------------------------------------------------------

// Handle closing array. Introduce a "null round" for ending array by closing it + moving to next.
void Type_value_iterator::handle_closing_array()
{
    m_is_array_end = false;

    ASSERT(M_ATTR, m_types_stack.top().second == 0);
    // check this struct...and then remove it
    const Type* type = get_type()->get_next();
    m_types_stack.pop();

    if (type) {
        set_type(type);
    }
}


//--------------------------------------------------------------------------------------------------

// Analyze whether we are at the end of an array.
bool Type_value_iterator::found_end_of_array() const
{
    if (m_types_stack.empty() || m_types_stack.top().second != 0)
        return false;
    if (m_types_stack.top().first->get_arraysize() == 1
     && m_types_stack.top().first->get_typecode() != TYPE_ARRAY)
        return false;

    return true;
}


//--------------------------------------------------------------------------------------------------

// Keep offset up-to-date. This special function is required to get offsets handled properly
// whether they are for dynamic arrays or the "global" offset.
void Type_value_iterator::set_offset(
    size_t value)					// increment
{
    if (m_dyn_offsets.empty())
        m_offset = value;
    else
        m_dyn_offsets.top() = value;
}


//--------------------------------------------------------------------------------------------------

// Retrieve current offset.
size_t Type_value_iterator::get_offset() const
{
    if (m_dyn_offsets.empty())
        return m_offset;
    else
        return m_dyn_offsets.top();
}


//--------------------------------------------------------------------------------------------------

// Retrieve current value_ptr.
const char* Type_value_iterator::get_value_ptr() const
{
    if (m_dyn_values.empty())
        return m_value_ptr;
    else
        return m_dyn_values.top();
}


//--------------------------------------------------------------------------------------------------

bool Type_value_iterator::is_dyn_array() const
{
    return get_type()->get_arraysize() == 0 && get_type()->get_typecode() != TYPE_ARRAY;
}


//==================================================================================================

//--------------------------------------------------------------------------------------------------

Skipping_type_value_iter::Skipping_type_value_iter(
    const Type* type,
    const char* values)
  : Type_value_iterator(type, values)
{
    pre_increment();
}


//--------------------------------------------------------------------------------------------------

// Continue as long as we don't hit the end.
void Skipping_type_value_iter::pre_increment()
{
    if (!get_type())
        return;

    while (is_struct_start() || is_array_start() || is_struct_end() || is_array_end()) {
        do_increment();
        if (!get_type())
            break;
    }
}


//--------------------------------------------------------------------------------------------------

// Continue as long as we don't hit the end.
void Skipping_type_value_iter::post_increment() //-V524 PVS
{
    if (!get_type())
        return;

    while (is_struct_start() || is_array_start() || is_struct_end() || is_array_end()) {
        do_increment();
        if (!get_type())
            break;
    }
}

//==================================================================================================

Type_named_value_iterator::Type_named_value_iterator()
{}


Type_named_value_iterator::Type_named_value_iterator(
    const Type* type,
    const char* values) :
    Type_value_iterator(type, values)
{
    const char* name = type ? get_type()->get_name() : "";
    m_name_stack.push(std::string(name ? name : ""));
}


std::string Type_named_value_iterator::get_qualified_name() const
{
    std::string qualified;
    if (!m_array_sizes.empty()) {
        if (m_array_sizes.top() != -1) {
            // in array
            m_strbuf.str("");
            m_strbuf << "[" << m_array_sizes.top() - get_arraycount() << "]";
            qualified = m_name_stack.top() + m_strbuf.str();
        }
        else {
            // in struct
            qualified = m_name_stack.top() + "." + get_type()->get_name();
        }
    }
    else {
        qualified = m_name_stack.top();
    }
    return qualified;
}


void Type_named_value_iterator::pre_increment()
{
    if (is_array_start()) {
        m_name_stack.push(get_qualified_name());
        m_array_sizes.push(get_arraycount());
    }

    if (is_struct_start()) {
        m_name_stack.push(get_qualified_name());
        m_array_sizes.push(-1); // marker for struct-parent
    }
}

void Type_named_value_iterator::post_increment()
{
    if (is_array_end()) {
        m_name_stack.pop();
        m_array_sizes.pop();
    }

    if (is_struct_end()) {
        m_name_stack.pop();
        m_array_sizes.pop();
    }
}

Type_named_value_iterator::Type_named_value_iterator(
    Type_named_value_iterator const& x)
    : Type_value_iterator(x),
      m_array_sizes(x.m_array_sizes),
      m_name_stack(x.m_name_stack)
{
    // m_strbuf cannot be copied
}

Type_named_value_iterator&
Type_named_value_iterator::operator=(Type_named_value_iterator const& x)
{
    if (&x != this) {
        Type_value_iterator::operator=(x);
        m_array_sizes = x.m_array_sizes;
        m_name_stack = x.m_name_stack;
        // m_strbuf cannot be copied
    }
    return *this;
}


}
}
