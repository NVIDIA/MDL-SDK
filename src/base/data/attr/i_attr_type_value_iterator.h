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
/// \brief The definition of the Type_value_iterator.

#ifndef BASE_DATA_ATTR_TYPE_VALUE_ITERATOR_H
#define BASE_DATA_ATTR_TYPE_VALUE_ITERATOR_H

#include <iterator>
#include <stack>
#include <vector>
#include <utility>
#include <base/system/stlext/i_stlext_iterator_facade.h>

namespace MI {
namespace ATTR  {
    
class Type;

/// Yet another Type-Value iterator. Iterate over the given type(s) and keep the given values in
/// sync. Note the specific iteration - if the increment results in pointing at eg a \c TYPE_STRUCT,
/// it will *not* automatically progress to its children but having its \c get_values() returning 0,
/// while the returned Type will be of type \c TYPE_STRUCT. This behaviour is necessary to find out
/// about \c TYPE_STRUCT, \c TYPE_ARRAY, and arrays.
/// \code
/// Attribute attr = ...;
/// Type_value_iterator it(&attr.get_type(), attr.get_values()), end(0, 0);
/// for (; it != end; ++it) {
///    Type_code typ = it->get_typecode();
///    const char* data = it.get_value();
///    // now cast data appropriately corresponding to the retrieved type code
/// }
/// \endcode
class Type_value_iterator : public STLEXT::iterator_facade<
    Type_value_iterator,
    Type,
    std::forward_iterator_tag,
    const Type&>
{
  public:
    /// Constructor.
    /// \param type the "root" type
    /// \param values the corresponding values pointer
    Type_value_iterator(
        const Type* type,
        const char* values);

    /// Default constructor, creates invalid iterator, usable as end-iterator.
    Type_value_iterator();

    /// Destructor.
    virtual ~Type_value_iterator();

    /// Test if iterator is valid. Alternative to comparison with end-iterator.
    bool is_valid() const;

    /// Retrieve current value.
    /// \return current underlying value, ie the address where the actual value starts
    const char* get_value() const;
    /// Retrieve active arraysize.
    /// \return number of elements still ahead of iteration
    int get_arraycount() const;
    /// Are we at the beginning of a struct?
    /// \return true, when iterating right into a structure
    bool is_struct_start() const;
    /// Are we at the end of a struct?
    /// \return true, when we are iterating right out of a structure
    bool is_struct_end() const;
    /// Are we at the beinning of an array?
    /// \return true, when iterating right into an array
    bool is_array_start() const;
    /// Are we at the end of a array?
    /// \return true, when we are iterating right out of an array
    bool is_array_end() const;
    /// Are we at a leaf value, i.e. neither at struct start/end nor array start/end.
    bool is_leaf_value() const;

    /// Retrieve current offset.
    /// \return current offset
    size_t get_offset() const;


    /// \name Iterator_facade_functionality
    /// Functionality required for implementing the \c iterator_facade.
    //@{
    /// Dereferencing the iterator.
    /// \return the current underlying \c Type
    const Type& dereference() const;
    /// Incrementing the iterator.
    void increment();
    /// Compare for equality.
    /// \param it the other iterator
    /// \return true, when both \c this and \p it are equal, false else
    bool equal(const Type_value_iterator& it) const;
    //@}

  protected:
    /// Retrieve current type.
    /// \return the current type, which is the current top of stack, or 0 else
    const Type* get_type() const;
    /// Increment to the next element. This function relies on the helper \c do_increment.
    void do_increment();
    /// Customizable utility - template method. Called before the actual increment takes place.
    virtual void pre_increment() {}
    /// Customizable utility - template method. Called after the actual increment took place.
    virtual void post_increment() {}

  private:
    const char* m_value_ptr;				///< the values
    size_t m_offset;					///< the offset into the values

    typedef std::pair<const Type*, size_t> Values;

    // Use vector-based stack, because vector is smaller than deque,
    // is simpler to debug, and sufficient for small stack depths.
    typedef std::stack<Values, std::vector<Values> > Values_stack;
    typedef std::stack<const char*, std::vector<const char*> > Pchar_stack;
    typedef std::stack<size_t, std::vector<size_t> >  Size_t_stack;

    Values_stack    m_types_stack;      ///< helper struct for parsing structs
    Pchar_stack     m_dyn_values;	///< start of values of dyn arrays
    Size_t_stack    m_dyn_offsets;	///< current offset into dyn arrays
    bool m_is_struct_start;		///< is struct start?
    bool m_is_struct_end;		///< is struct end?
    bool m_is_array_start;		///< is array start?
    bool m_is_array_end;		///< is array end?

    /// Set the given Type \c type. If the given type is 0 try to use the stack to come up with
    /// the next type. This happens when in a struct the last type was iterated over, ie m_next == 0
    /// or, at the very end.
    /// \param type the new \c Type
    /// \param in_struct are we in a struct?
    void set_type(
        const Type* type,
        bool in_struct=false);
    /// Progress to next element. Proceed one element further - which might then be null. This
    /// function takes care of alignment issues and keeps \c m_offset up to date.
    void progress();

    /// Handle closing struct.
    void handle_closing_struct();
    /// Handle closing array.
    void handle_closing_array();
    /// Handle potentially non-empty array.
    /// \return true, when non-empty array was successfully handled, false otherwise 
    bool handle_array();
    /// Analyze whether we are at the end of an array.
    /// \return true when at end of array
    bool found_end_of_array() const;

    /// Keep offset up-to-date.  This special function is required to get offsets handled properly
    /// whether they are for dynamic arrays or the "global" offset.
    /// \param value new offset increment
    void set_offset(
        size_t value);
    /// Retrieve current value_ptr.
    /// \return current value_ptr
    const char* get_value_ptr() const;
    /// Retrieve whether we deal with a dynamic array right now.
    bool is_dyn_array() const;

    /// Debug helper.
    void dump() const;
};


//==================================================================================================

/// A \c Type_value_iterator which stops only at values. In other words, it does skip at array or
/// struct boundaries.
class Skipping_type_value_iter : public Type_value_iterator
{
  public:
    /// Constructor.
    Skipping_type_value_iter(
        const Type* type,
        const char* values);

  protected:
    /// Customizable utility - template method. Called before the actual increment takes place.
    void pre_increment();
    /// Customizable utility - template method. Called after the actual increment took place.
    void post_increment();
};


//==================================================================================================

/// Increment iterator until next leaf value or end.
/// Usable for all subclasses of Type_value_iterator, like Type_named_value_iterator.
inline void to_next_leaf(Type_value_iterator& iter)
{
    if (iter.is_valid()) {
        ++iter;
        while (iter.is_valid() && !iter.is_leaf_value()) {
            ++iter;
        }
    }
}

}
}

#endif
