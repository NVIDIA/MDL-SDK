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
/** \file
 ** \brief      Fixed-size bitset similar to the STL bitset template.
 **             See \c Bitset for details.
 **
 **/

#ifndef BASE_LIB_CONT_BITSET_H
#define BASE_LIB_CONT_BITSET_H

#include <base/lib/math/i_math.h>

namespace MI {
namespace CONT {

namespace DETAIL {


/** \brief Thin wrapper around \c sizeof, computes the size in bits. */
template <typename T>
struct Bit_sizeof
{
    enum { value = sizeof(T) * CHAR_BIT };
};


/** Computes the number of data elements of type \c T for bitsets of \c bit_count bits. */
template <typename T, size_t bit_count>
struct Data_count
{
    enum { value = (Bit_sizeof<T>::value + bit_count - 1) / Bit_sizeof<T>::value };
};


/** Base functionality and data storage for arbitrarily sized bitsets.

 This class stores an array of \c count integral elements of type \p T.
 */
template <typename T, size_t count>
class Bitset_storage_base
{
public:
    typedef T Data_t;

    Bitset_storage_base();
    Bitset_storage_base(T data);
    bool equals(const Bitset_storage_base& rhs) const;
    void set(size_t bit);
    void unset(size_t bit);
    bool is_set(size_t bit) const;
    bool any(const Bitset_storage_base& data) const;
    bool any() const;
    bool all(const Bitset_storage_base& data) const;
    void do_and(const Bitset_storage_base& rhs);
    void do_or(const Bitset_storage_base& rhs);
    void do_xor(const Bitset_storage_base& rhs);
    void flip();
    T& data(size_t bit);
    T mask(size_t bit);
    const T* begin() const;
    const T* end() const;
    T* begin();
    T* end();

private:
    static size_t index(size_t bit);
    static size_t subbit(size_t bit);

    T m_data[count];
};


/** Specialization of the \c Bitset_storage_base for bit counts which fit a single integral type. */
template <typename T>
class Bitset_storage_base<T,1> {
public:
    typedef T Data_t;

    Bitset_storage_base();
    Bitset_storage_base(T data);
    bool equals(const Bitset_storage_base& rhs) const;
    void set(size_t bit);
    void unset(size_t bit);
    bool is_set(size_t bit) const;
    bool any(T data) const;
    bool any() const;
    bool all(T data) const;
    void do_and(const Bitset_storage_base& rhs);
    void do_or(const Bitset_storage_base& rhs);
    void do_xor(const Bitset_storage_base& rhs);
    void flip();
    T& data(size_t);
    T mask(size_t bit);
    const T* begin() const;
    const T* end() const;
    T* begin();
    T* end();
    operator T() const;
    operator T&();

private:
    T m_data;
};


#define MI_MAKE_STORAGE(T,c) \
: public Bitset_storage_base<T,c> { \
public: \
    Bitset_storage() {} \
    Bitset_storage(T v) : Bitset_storage_base<T,c>(v) {} \
}

#define MI_TEMP_COUNT Data_count<Uint64,byte_count*CHAR_BIT>::value
template <size_t byte_count>
struct Bitset_storage MI_MAKE_STORAGE(Uint64,MI_TEMP_COUNT);
#undef MI_TEMP_COUNT

template <>
struct Bitset_storage<1> MI_MAKE_STORAGE(Uint8,1);

template <>
struct Bitset_storage<2> MI_MAKE_STORAGE(Uint16,1);

template <>
struct Bitset_storage<4> MI_MAKE_STORAGE(Uint32,1);

template <>
struct Bitset_storage<8> MI_MAKE_STORAGE(Uint64,1);

#undef MI_MAKE_STORAGE

}


/** \brief A statically sized collection of bits.

 This class is close to STL's \c bitset in design and functionality. However, the storage
 required for small bitsets is different. This interface also provides some additional
 functionality.

 For sets of less than 64 bits, this class requires exactly the space needed by the smallest
 integral type of \c bit_count or more bits. For example, \code sizeof(Bitset<7>)==1\endcode
 Larger sets are split over a number of integers internally.

 Note that none of the operation on this class are range checked in release mode. However,
 most operations contain assertions about index validity.
 */
template <size_t bit_count>
class Bitset
{
private:
    typedef DETAIL::Bitset_storage<
            bit_count<=64 ?
            (size_t)MATH::Round_up_pow2<DETAIL::Data_count<char,(Uint32)bit_count>::value>::value :
            (size_t)DETAIL::Data_count<char,bit_count>::value
            > Storage;
    class Bit;

public:
    typedef typename Storage::Data_t Data_t;


    /** \brief Sets all bits to 0. */
    Bitset();


    /** \brief Sets the lowest \c bit_count bits to those provided in \p value. */
    explicit Bitset(Data_t value);


    /** \brief Accesses the \p bit_index'th bit. */
    bool operator[](size_t bit_index) const;


    /** \brief Accesses the \p bit_index'th bit. */
    Bit operator[](size_t bit_index);


    /** \brief Accesses the \p bit_index'th bit. */
    bool is_set(size_t bit_index) const;


    /** \brief Sets the \p bit_index'th bit to \c true. */
    Bitset& set(size_t bit_index);


    /** \brief Sets the \p bit_index'th bit to \c false. */
    Bitset& unset(size_t bit_index);


    /** \brief Sets the \p bit_index'th bit to \p value. */
    Bitset& set(size_t bit_index, bool value);


    /** \brief Checks if all of the bits present in \p mask are present in this set.

     This function checks if this set is a superset of \p mask.
     */
    bool all(const Bitset& mask) const;
    bool all(Data_t mask) const;


    /** \brief Checks if any of the bits present in \p mask are present in this set.

     This function checks if this and \p mask are not disjoint.
     */
    bool any(const Bitset& mask) const;
    bool any(Data_t mask) const;


    /** \brief Checks if any bit is set in this set. */
    bool any() const;


    /** \brief Checks that none of the bits present in \p mask are present in this set.

     This function checks if this and \p mask are disjoint.
     */
    bool none(const Bitset& mask) const;
    bool none(Data_t mask) const;


    /** \brief Checks if no bit is set in this set. */
    bool none() const;


    /** \brief Flips all bits in this set.

     This function turns this set into its complement.
     */
    Bitset& flip();


    /** \brief Checks if this and \p rhs contain exactly the same bits. */
    bool operator==(const Bitset& rhs) const;


    /** \brief Checks if this and \p rhs contain different bits. */
    bool operator!=(const Bitset& rhs) const;


    /** \brief Turns this set into the intersection of itself with \p rhs. */
    Bitset& operator&=(const Bitset& rhs);


    /** \brief Turns this set into the union of itself with \p rhs. */
    Bitset& operator|=(const Bitset& rhs);


    /** \brief Turns this set into the symmetric difference between itself with \p rhs */
    Bitset& operator^=(const Bitset& rhs);


    /** \brief Returns the complement of this set. */
    Bitset operator~() const;


    /** \brief See \c Bitset::any. */
    operator bool() const;


    /** \brief See \c Bitset::none. */
    bool operator!() const;


    /** \brief Starting iterator for internal data access. */
    const Data_t* begin_data() const;


    /** \brief End iterator for internal data access. */
    const Data_t* end_data() const;


    /** \brief Starting iterator for internal data access. */
    Data_t* begin_data();


    /** \brief End iterator for internal data access. */
    Data_t* end_data();


private:
    class Bit
    {
    public:
        Bit(Storage& data, size_t index);
        Bit& operator=(const Bit& val);
        Bit& operator=(bool val);
        operator bool() const;
        bool operator~() const;
        Bit& flip();

    private:
        const Data_t    m_mask;
        Data_t&         m_data;
    };

    Storage m_data;

    void clean();
};


/** \brief See \c Bitset::operator&=() */
template <size_t bit_count>
Bitset<bit_count> operator&(Bitset<bit_count> lhs, const Bitset<bit_count>& rhs);


/** \brief See \c Bitset::operator|=() */
template <size_t bit_count>
Bitset<bit_count> operator|(Bitset<bit_count> lhs, const Bitset<bit_count>& rhs);


/** \brief See \c Bitset::operator^=() */
template <size_t bit_count>
Bitset<bit_count> operator^(Bitset<bit_count> lhs, const Bitset<bit_count>& rhs);


}}

#include "cont_bitset_inline.h"

#endif //BASE_LIB_CONT_BITSET_H
