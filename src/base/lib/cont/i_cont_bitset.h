/******************************************************************************
 * Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
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

#include "i_cont_number.h"
#include <base/lib/math/i_math.h>


namespace MI {
namespace CONT {

namespace DETAIL {


/** \brief Thin wrapper around \c sizeof, computes the size in bits. */
template <typename T>
struct Bit_sizeof
{
    enum : std::size_t { value = sizeof(T) * CHAR_BIT };
};


/** Computes the number of data elements of type \c T for bitsets of \c bit_count bits. */
template <typename T, size_t bit_count>
struct Data_count
{
    enum : std::size_t { value = (Bit_sizeof<T>::value + bit_count - 1) / Bit_sizeof<T>::value };
};


/** Base functionality and data storage for arbitrarily sized bitsets.

 This class stores an array of \c count integral elements of type \p T.
 */
template <typename T, std::size_t count>
class Bitset_storage_base
{
public:
    typedef T Data_t;

    constexpr Bitset_storage_base() = default;
    constexpr explicit Bitset_storage_base(T data);
    constexpr bool equals(const Bitset_storage_base& rhs) const;
    constexpr bool is_less_than(const Bitset_storage_base& rhs) const;
    void set(std::size_t bit);
    void unset(std::size_t bit);
    constexpr bool is_set(std::size_t bit) const;
    constexpr bool any(const Bitset_storage_base& data) const;
    constexpr bool any() const;
    constexpr bool all(const Bitset_storage_base& data) const;
    void do_and(const Bitset_storage_base& rhs);
    void do_or(const Bitset_storage_base& rhs);
    void do_xor(const Bitset_storage_base& rhs);
    constexpr void flip();
    constexpr T& data(std::size_t bit);
    T mask(std::size_t bit);
    constexpr const T* begin() const;
    constexpr const T* end() const;
    T* begin();
    T* end();

private:
    static std::size_t index(std::size_t bit);
    static std::size_t subbit(std::size_t bit);

    T m_data[count] = {};
};


/** Specialization of the \c Bitset_storage_base for bit counts which fit a single integral type. */
template <typename T>
class Bitset_storage_base<T,1> {
public:
    typedef T Data_t;

    constexpr Bitset_storage_base() = default;
    constexpr explicit Bitset_storage_base(T data);
    constexpr bool equals(const Bitset_storage_base& rhs) const;
    constexpr bool is_less_than(const Bitset_storage_base& rhs) const;
    void set(std::size_t bit);
    void unset(std::size_t bit);
    constexpr bool is_set(std::size_t bit) const;
    constexpr bool any(T data) const;
    constexpr bool any() const;
    constexpr bool all(T data) const;
    void do_and(const Bitset_storage_base& rhs);
    void do_or(const Bitset_storage_base& rhs);
    void do_xor(const Bitset_storage_base& rhs);
    constexpr void flip();
    constexpr T& data(std::size_t);
    T mask(std::size_t bit);
    constexpr const T* begin() const;
    constexpr const T* end() const;
    T* begin();
    T* end();
    constexpr operator T() const;
    operator T&();

private:
    T m_data = {0};
};


#define MI_MAKE_STORAGE(T,c) \
: public Bitset_storage_base<T,c> { public: using Bitset_storage_base<T,c>::Bitset_storage_base; }

template <std::size_t byte_count>
struct Bitset_storage MI_MAKE_STORAGE(Uint64,(Data_count<Uint64,byte_count*CHAR_BIT>::value));

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


template <typename T>
struct is_index { static const auto value = std::is_enum<T>::value || std::is_integral<T>::value; };


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
template <std::size_t bit_count>
class Bitset
{
private:
    typedef DETAIL::Bitset_storage<
            bit_count<=64 ?
            (std::size_t)MATH::Round_up_pow2<DETAIL::Data_count<char,bit_count>::value>::value :
                                             DETAIL::Data_count<char,bit_count>::value
            > Storage;
    class Bit;

    template <typename T> using if_index = std::enable_if_t<is_index<T>::value>;

public:
    using Base_data_type = typename Storage::Data_t;
    using Data_type = Wrapped_data<struct Bitset_data,Base_data_type>;


    /** \brief Sets all bits to 0. */
    constexpr Bitset() = default;


    /** \brief Sets the lowest \c bit_count bits to those provided in \p value. */
    explicit constexpr Bitset(Data_type value);


    /** \brief Sets the bits at the provided indices */
    template <typename T, typename=if_index<T>>
    Bitset(std::initializer_list<T> indices);


    /** \brief Accesses the \p bit_index'th bit. */
    template <typename T, typename=if_index<T>>
    constexpr bool operator[](T bit_index) const;


    /** \brief Accesses the \p bit_index'th bit. */
    template <typename T, typename=if_index<T>>
    Bit operator[](T bit_index);


    /** \brief Accesses the \p bit_index'th bit. */
    template <typename T, typename=if_index<T>>
    constexpr bool is_set(T bit_index) const;


    /** \brief Sets the \p bit_index'th bit to \c true. */
    template <typename T, typename=if_index<T>>
    Bitset& set(T bit_index);


    /** \brief Sets the \p bit_index'th bit to \c false. */
    template <typename T, typename=if_index<T>>
    Bitset& unset(T bit_index);


    /** \brief Sets the \p bit_index'th bit to \p value. */
    template <typename T, typename=if_index<T>>
    Bitset& set(T bit_index, bool value);


    /** \brief Clears all bits. */
    Bitset& clear();


    /** \brief Checks if all of the bits present in \p mask are present in this set.

     This function checks if this set is a superset of \p mask.
     */
    constexpr bool all(const Bitset& mask) const;
    constexpr bool all(Data_type mask) const;


    /** \brief Checks if any of the bits present in \p mask are present in this set.

     This function checks if this and \p mask are not disjoint.
     */
    constexpr bool any(const Bitset& mask) const;
    constexpr bool any(Data_type mask) const;


    /** \brief Checks if any bit is set in this set. */
    constexpr bool any() const;


    /** \brief Checks that none of the bits present in \p mask are present in this set.

     This function checks if this and \p mask are disjoint.
     */
    constexpr bool none(const Bitset& mask) const;
    constexpr bool none(Data_type mask) const;


    /** \brief Checks if no bit is set in this set. */
    constexpr bool none() const;


    /** \brief Flips all bits in this set.

     This function turns this set into its complement.
     */
    constexpr Bitset& flip();


    /** \brief Checks if this and \p rhs contain exactly the same bits. */
    constexpr bool operator==(const Bitset& rhs) const;


    /** \brief Checks if this and \p rhs contain different bits. */
    constexpr bool operator!=(const Bitset& rhs) const;


    /** \brief Checks if this should be ordered before \p rhs. */
    constexpr bool operator<(const Bitset& rhs) const;


    /** \brief Turns this set into the intersection of itself with \p rhs. */
    Bitset& operator&=(const Bitset& rhs);


    /** \brief Turns this set into the union of itself with \p rhs. */
    Bitset& operator|=(const Bitset& rhs);


    /** \brief Turns this set into the symmetric difference between itself with \p rhs */
    Bitset& operator^=(const Bitset& rhs);


    /** \brief Returns the complement of this set. */
    constexpr Bitset operator~() const;


    /** \brief See \c Bitset::any. */
    explicit constexpr operator bool() const;


    /** \brief See \c Bitset::none. */
    constexpr bool operator!() const;


    /** \brief Starting iterator for internal data access. */
    constexpr const Base_data_type* begin_data() const;


    /** \brief End iterator for internal data access. */
    constexpr const Base_data_type* end_data() const;


    /** \brief Starting iterator for internal data access. */
    Base_data_type* begin_data();


    /** \brief End iterator for internal data access. */
    Base_data_type* end_data();


private:
    class Bit
    {
    public:
        template <typename T, typename=if_index<T>>
        constexpr Bit(Storage& data, T index);
        Bit(Bit&&) = default;
        Bit& operator=(Bit&&) = default;
        Bit& operator=(const Bit& val);
        Bit& operator=(bool val);
        constexpr operator bool() const;
        constexpr bool operator~() const;
        constexpr bool operator!() const;
        constexpr Bit& flip();

    private:
        Bit(const Bit&) = delete;

        const Base_data_type    m_mask;
        Base_data_type&         m_data;
    };

    Storage m_data{};

    constexpr void clean();
};


/** \brief See \c Bitset::operator&=() */
template <std::size_t bit_count>
Bitset<bit_count> operator&(Bitset<bit_count> lhs, const Bitset<bit_count>& rhs);


/** \brief See \c Bitset::operator|=() */
template <std::size_t bit_count>
Bitset<bit_count> operator|(Bitset<bit_count> lhs, const Bitset<bit_count>& rhs);


/** \brief See \c Bitset::operator^=() */
template <std::size_t bit_count>
Bitset<bit_count> operator^(Bitset<bit_count> lhs, const Bitset<bit_count>& rhs);


/** \brief Converts an enumerator to its underlying integral value. */
template <typename T, std::enable_if_t<std::is_enum<T>::value,bool> = true>
constexpr auto to_underlying(const T val) { return std::underlying_type_t<T>(val); }

template <typename T, std::enable_if_t<std::is_integral<T>::value,bool> = true>
constexpr auto to_underlying(const T val) { return val; }


}}

#include "cont_bitset_inline.h"

#endif //BASE_LIB_CONT_BITSET_H
