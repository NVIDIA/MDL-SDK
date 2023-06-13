/******************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef BASE_LIB_CONT_BITSET_INLINE_H
#define BASE_LIB_CONT_BITSET_INLINE_H

#include <algorithm>
#include <ostream>


namespace MI {
namespace CONT {

namespace DETAIL {


template <typename T, std::size_t count, typename E>
constexpr Bitset_storage_base<T,count,E>::Bitset_storage_base(T data)
: m_data{}
{
    m_data[0] = data;
}


template <typename T, std::size_t count, typename E>
constexpr bool Bitset_storage_base<T,count,E>::equals(const Bitset_storage_base& rhs) const
{
    for (std::size_t i=0; i<count; ++i)
        if (m_data[i] != rhs.m_data[i])
            return false;
    return true;
}


template <typename T, std::size_t count, typename E>
constexpr bool Bitset_storage_base<T,count,E>::is_less_than(const Bitset_storage_base& rhs) const
{
    return std::lexicographical_compare(
            std::begin(m_data),std::end(m_data),
            std::begin(rhs.m_data),std::end(rhs.m_data));
}


template <typename T, std::size_t count, typename E>
inline void Bitset_storage_base<T,count,E>::set(std::size_t bit)
{
    m_data[index(bit)] |= (T(1u) << subbit(bit));
}


template <typename T, std::size_t count, typename E>
inline void Bitset_storage_base<T,count,E>::unset(std::size_t bit)
{
    m_data[index(bit)] &= ~(T(1u) << subbit(bit));
}


template <typename T, std::size_t count, typename E>
constexpr bool Bitset_storage_base<T,count,E>::is_set(std::size_t bit) const
{
    return 0 != (m_data[index(bit)] & (T(1u) << subbit(bit)));
}


template <typename T, std::size_t count, typename E>
constexpr bool Bitset_storage_base<T,count,E>::any(const Bitset_storage_base& data) const
{
    for (std::size_t i=0; i<count; ++i)
        if (0 != (data.m_data[i] & m_data[i]))
            return true;
    return false;
}


template <typename T, std::size_t count, typename E>
constexpr bool Bitset_storage_base<T,count,E>::any() const
{
    for (std::size_t i=0; i<count; ++i)
        if (0 != m_data[i])
            return true;
    return false;
}


template <typename T, std::size_t count, typename E>
constexpr bool Bitset_storage_base<T,count,E>::all(const Bitset_storage_base& data) const
{
    for (std::size_t i=0; i<count; ++i)
        if (data.m_data[i] != (data.m_data[i] & m_data[i]))
            return false;
    return true;
}


template <typename T, std::size_t count, typename E>
void Bitset_storage_base<T,count,E>::do_and(const Bitset_storage_base& rhs)
{
    for (std::size_t i=0; i<count; ++i)
        m_data[i] &= rhs.m_data[i];
}


template <typename T, std::size_t count, typename E>
void Bitset_storage_base<T,count,E>::do_or(const Bitset_storage_base& rhs)
{
    for (std::size_t i=0; i<count; ++i)
        m_data[i] |= rhs.m_data[i];
}


template <typename T, std::size_t count, typename E>
void Bitset_storage_base<T,count,E>::do_xor(const Bitset_storage_base& rhs)
{
    for (std::size_t i=0; i<count; ++i)
        m_data[i] ^= rhs.m_data[i];
}


template <typename T, std::size_t count, typename E>
constexpr void Bitset_storage_base<T,count,E>::flip()
{
    for (std::size_t i=0; i<count; ++i)
        m_data[i] = ~m_data[i];
}


template <typename T, std::size_t count, typename E>
constexpr T& Bitset_storage_base<T,count,E>::data(std::size_t bit)
{
    return m_data[index(bit)];
}


template <typename T, std::size_t count, typename E>
constexpr const T* Bitset_storage_base<T,count,E>::begin() const
{
    return m_data;
}


template <typename T, std::size_t count, typename E>
constexpr const T* Bitset_storage_base<T,count,E>::end() const
{
    return m_data+count;
}


template <typename T, std::size_t count, typename E>
inline T* Bitset_storage_base<T,count,E>::begin()
{
    return m_data;
}


template <typename T, std::size_t count, typename E>
inline T* Bitset_storage_base<T,count,E>::end()
{
    return m_data+count;
}


template <typename T, std::size_t count, typename E>
inline T Bitset_storage_base<T,count,E>::mask(std::size_t bit)
{
    return T(1u) << subbit(bit);
}


template <typename T, std::size_t count, typename E>
inline std::size_t Bitset_storage_base<T,count,E>::index(std::size_t bit)
{
    return bit / Bit_sizeof<T>::value;
}


template <typename T, std::size_t count, typename E>
inline std::size_t Bitset_storage_base<T,count,E>::subbit(std::size_t bit)
{
    return bit % Bit_sizeof<T>::value;
}


//-------------------------------------------------------------------------------------------------


template <typename T>
constexpr Bitset_storage_base<T,1>::Bitset_storage_base(T data)
: m_data(data)
{
}


template <typename T>
constexpr bool Bitset_storage_base<T,1>::equals(const Bitset_storage_base& rhs) const
{
    return m_data == rhs.m_data;
}


template <typename T>
constexpr bool Bitset_storage_base<T,1>::is_less_than(const Bitset_storage_base& rhs) const
{
    return m_data < rhs.m_data;
}


template <typename T>
inline void Bitset_storage_base<T,1>::set(std::size_t bit)
{
    m_data |= (T(1u) << bit);
}


template <typename T>
inline void Bitset_storage_base<T,1>::unset(std::size_t bit)
{
    m_data &= ~(T(1u) << bit);
}


template <typename T>
constexpr bool Bitset_storage_base<T,1>::is_set(std::size_t bit) const
{
    return 0 != (m_data & (T(1u) << bit));
}


template <typename T>
constexpr bool Bitset_storage_base<T,1>::any(T data) const
{
    return 0 != (data & m_data);
}


template <typename T>
constexpr bool Bitset_storage_base<T,1>::any() const
{
    return 0 != m_data;
}


template <typename T>
constexpr bool Bitset_storage_base<T,1>::all(T data) const
{
    return data == (data & m_data);
}


template <typename T>
inline void Bitset_storage_base<T,1>::do_and(const Bitset_storage_base& rhs)
{
    m_data &= rhs.m_data;
}


template <typename T>
inline void Bitset_storage_base<T,1>::do_or(const Bitset_storage_base& rhs)
{
    m_data |= rhs.m_data;
}


template <typename T>
inline void Bitset_storage_base<T,1>::do_xor(const Bitset_storage_base& rhs)
{
    m_data ^= rhs.m_data;
}


template <typename T>
constexpr void Bitset_storage_base<T,1>::flip()
{
    m_data = ~m_data;
}


template <typename T>
constexpr T& Bitset_storage_base<T,1>::data(std::size_t)
{
    return m_data;
}


template <typename T>
inline T Bitset_storage_base<T,1>::mask(std::size_t bit)
{
    return T(1u) << bit;
}


template <typename T>
constexpr const T* Bitset_storage_base<T,1>::begin() const
{
    return &m_data;
}


template <typename T>
constexpr const T* Bitset_storage_base<T,1>::end() const
{
    return &m_data+1;
}


template <typename T>
inline T* Bitset_storage_base<T,1>::begin()
{
    return &m_data;
}


template <typename T>
inline T* Bitset_storage_base<T,1>::end()
{
    return &m_data+1;
}


template <typename T>
constexpr Bitset_storage_base<T,1>::operator T() const
{
    return m_data;
}


template <typename T>
inline Bitset_storage_base<T,1>::operator T&()
{
    return m_data;
}


//-------------------------------------------------------------------------------------------------


template <typename T, std::size_t used_bit_count>
struct Bit_cleaner
{
    static constexpr void clean(T& bits) { bits &= ~(T(~T(0u)) << used_bit_count); }
};


template <typename T>
struct Bit_cleaner<T,0> // all bits used
{
    static constexpr void clean(T&) {}
};


}


//-------------------------------------------------------------------------------------------------



template <std::size_t bit_count>
constexpr Bitset<bit_count>::Bitset(Data_type value)
: m_data(value)
{
    clean();
}


template <std::size_t bit_count>
template <typename T, typename>
Bitset<bit_count>::Bitset(const std::initializer_list<T> indices)
{
    for (const auto bit : indices)
        (*this)[to_underlying(bit)] = true;
}


template <std::size_t bit_count>
template <typename T, typename>
constexpr bool Bitset<bit_count>::operator[](T bit_index) const
{
    ASSERT(M_CONT,to_underlying(bit_index)<bit_count);
    return m_data.is_set(to_underlying(bit_index));
}


template <std::size_t bit_count>
template <typename T, typename>
inline typename Bitset<bit_count>::Bit Bitset<bit_count>::operator[](T bit_index)
{
    ASSERT(M_CONT,to_underlying(bit_index)<bit_count);
    return Bit(m_data,to_underlying(bit_index));
}


template <std::size_t bit_count>
template <typename T, typename>
constexpr bool Bitset<bit_count>::is_set(T bit_index) const
{
    ASSERT(M_CONT,to_underlying(bit_index)<bit_count);
    return m_data.is_set(to_underlying(bit_index));
}


template <std::size_t bit_count>
template <typename T, typename>
inline Bitset<bit_count>& Bitset<bit_count>::set(T bit_index)
{
    ASSERT(M_CONT,to_underlying(bit_index)<bit_count);
    m_data.set(to_underlying(bit_index));
    return *this;
}


template <std::size_t bit_count>
template <typename T, typename>
inline Bitset<bit_count>& Bitset<bit_count>::unset(T bit_index)
{
    ASSERT(M_CONT,to_underlying(bit_index)<bit_count);
    m_data.unset(to_underlying(bit_index));
    return *this;
}


template <std::size_t bit_count>
template <typename T, typename>
inline Bitset<bit_count>& Bitset<bit_count>::set(T bit_index, bool value)
{
    return value ? set(bit_index) : unset(bit_index);
}


template <std::size_t bit_count>
inline Bitset<bit_count>& Bitset<bit_count>::clear()
{
    *this = Bitset{};
    return *this;
}


template <std::size_t bit_count>
constexpr bool Bitset<bit_count>::all(const Bitset& mask) const
{
    return m_data.all(mask.m_data);
}


template <std::size_t bit_count>
constexpr bool Bitset<bit_count>::all(const Data_type mask) const
{
    return all(Bitset{mask});
}


template <std::size_t bit_count>
constexpr bool Bitset<bit_count>::any(const Bitset& mask) const
{
    return m_data.any(mask.m_data);
}


template <std::size_t bit_count>
constexpr bool Bitset<bit_count>::any(const Data_type mask) const
{
    return any(Bitset{mask});
}


template <std::size_t bit_count>
constexpr bool Bitset<bit_count>::any() const
{
    return m_data.any();
}


template <std::size_t bit_count>
constexpr bool Bitset<bit_count>::none(const Bitset& mask) const
{
    return !m_data.any(mask.m_data);
}


template <std::size_t bit_count>
constexpr bool Bitset<bit_count>::none(const Data_type mask) const
{
    return none(Bitset{mask});
}


template <std::size_t bit_count>
constexpr bool Bitset<bit_count>::none() const
{
    return !m_data.any();
}


template <std::size_t bit_count>
constexpr Bitset<bit_count>& Bitset<bit_count>::flip()
{
    m_data.flip();
    clean();
    return *this;
}


template <std::size_t bit_count>
constexpr bool Bitset<bit_count>::operator==(const Bitset& rhs) const
{
    return m_data.equals(rhs.m_data);
}


template <std::size_t bit_count>
constexpr bool Bitset<bit_count>::operator!=(const Bitset& rhs) const
{
    return !m_data.equals(rhs.m_data);
}


template <std::size_t bit_count>
constexpr bool Bitset<bit_count>::operator<(const Bitset& rhs) const
{
    return m_data.is_less_than(rhs.m_data);
}


template <std::size_t bit_count>
inline Bitset<bit_count>& Bitset<bit_count>::operator&=(const Bitset& rhs)
{
    m_data.do_and(rhs.m_data);
    return *this;
}


template <std::size_t bit_count>
inline Bitset<bit_count>& Bitset<bit_count>::operator|=(const Bitset& rhs)
{
    m_data.do_or(rhs.m_data);
    return *this;
}


template <std::size_t bit_count>
inline Bitset<bit_count>& Bitset<bit_count>::operator^=(const Bitset& rhs)
{
    m_data.do_xor(rhs.m_data);
    return *this;
}


template <std::size_t bit_count>
constexpr Bitset<bit_count> Bitset<bit_count>::operator~() const
{
    return Bitset(*this).flip();
}


template <std::size_t bit_count>
constexpr Bitset<bit_count>::operator bool() const
{
    return any();
}


template <std::size_t bit_count>
constexpr bool Bitset<bit_count>::operator!() const
{
    return none();
}


template <std::size_t bit_count>
constexpr const typename Bitset<bit_count>::Base_data_type* Bitset<bit_count>::begin_data() const
{
    return m_data.begin();
}


template <std::size_t bit_count>
constexpr const typename Bitset<bit_count>::Base_data_type* Bitset<bit_count>::end_data() const
{
    return m_data.end();
}


template <std::size_t bit_count>
inline typename Bitset<bit_count>::Base_data_type* Bitset<bit_count>::begin_data()
{
    return m_data.begin();
}


template <std::size_t bit_count>
inline typename Bitset<bit_count>::Base_data_type* Bitset<bit_count>::end_data()
{
    return m_data.end();
}


template <std::size_t bit_count>
constexpr void Bitset<bit_count>::clean()
{
    typedef DETAIL::Bit_cleaner<Base_data_type,bit_count%(sizeof(Base_data_type)*CHAR_BIT)> Cleaner;
    Cleaner::clean(m_data.data(bit_count-1));
}


template <std::size_t bit_count>
std::ostream& operator<<(std::ostream& str, const Bitset<bit_count>& set)
{
    for (std::size_t i=0; i<bit_count; ++i)
        str << set[bit_count-1-i];
    return str;
}



//-------------------------------------------------


template <std::size_t bit_count>
template <typename T, typename>
constexpr Bitset<bit_count>::Bit::Bit(Storage& data, T index)
: m_mask(data.mask(to_underlying(index)))
, m_data(data.data(to_underlying(index)))
{
}


template <std::size_t bit_count>
inline typename Bitset<bit_count>::Bit& Bitset<bit_count>::Bit::operator=(const Bit& val)
{
    const bool b = val;
    return *this = b;
}


template <std::size_t bit_count>
inline typename Bitset<bit_count>::Bit& Bitset<bit_count>::Bit::operator=(bool val)
{
    if (val)
        m_data |= m_mask;
    else
        m_data &= ~m_mask;
    return *this;
}


template <std::size_t bit_count>
constexpr Bitset<bit_count>::Bit::operator bool() const
{
    return 0 != (m_data & m_mask);
}


template <std::size_t bit_count>
constexpr bool Bitset<bit_count>::Bit::operator~() const
{
    return 0 == (m_data & m_mask);
}


template <std::size_t bit_count>
constexpr bool Bitset<bit_count>::Bit::operator!() const
{
    return 0 == (m_data & m_mask);
}


template <std::size_t bit_count>
constexpr typename Bitset<bit_count>::Bit& Bitset<bit_count>::Bit::flip()
{
    m_data ^= m_mask;
    return *this;
}


//-------------------------------------------------------------------------------------------------


template <std::size_t bit_count>
inline Bitset<bit_count> operator&(Bitset<bit_count> lhs, const Bitset<bit_count>& rhs)
{
    lhs &= rhs;
    return lhs;
}


template <std::size_t bit_count>
inline Bitset<bit_count> operator|(Bitset<bit_count> lhs, const Bitset<bit_count>& rhs)
{
    lhs |= rhs;
    return lhs;
}


template <std::size_t bit_count>
inline Bitset<bit_count> operator^(Bitset<bit_count> lhs, const Bitset<bit_count>& rhs)
{
    lhs ^= rhs;
    return lhs;
}

}}

#endif //BASE_LIB_CONT_BITSET_INLINE_H
