/***************************************************************************************************
 * Copyright (c) 2007-2022, NVIDIA CORPORATION. All rights reserved.
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
/// \brief 

#include <type_traits>
#include <cassert>

namespace MI {

namespace SERIAL {

template <typename T>
inline void Serializer::write(const DB::Typed_tag<T>& value)
{
    write(value.get_untyped());
}

template <typename T>
inline void Deserializer::read(DB::Typed_tag<T>* value_pointer)
{
    DB::Tag t;
    read(&t);
    *value_pointer = DB::Typed_tag<T>(t);
}

template <typename T>
inline void Serializer::write(const CONT::Array<T>& array)
{
    write(STLEXT::safe_cast<Uint32>(array.size()));
    write_range(*this, array.begin(), array.end());
}

template <typename T>
inline void Serializer::write(const CONT::Array<T*>& array)
{
    const Uint32 size( STLEXT::safe_cast<Uint32>(array.size()) );
    write(size);
    for (Uint32 i(0u); i != size; ++i)
        serialize(array[i]);
}

template <typename T, typename SWO>
inline void Serializer::write(const std::set<T, SWO>& set)
{
    write_size_t(set.size());
    write_range(*this, set.begin(), set.end());
}

template <typename T1, typename T2> 
inline void Serializer::write(const std::pair<T1, T2>& pair)
{
    write(pair.first);
    write(pair.second);
}

template <typename T>
inline void Deserializer::read(CONT::Array<T>* array)
{
    Uint32 size;
    read(&size);
    array->resize(size);
    read_range(*this, array->begin(), array->end());
}

template <typename T>
inline void Deserializer::read(CONT::Array<T*>* array)
{
    Uint32 size;
    read(&size);
    array->resize(size);
    for (Uint32 i(0u); i != size; ++i)
        (*array)[i] = reinterpret_cast<T*>( deserialize() );
}

template <typename T, typename A1, typename A2>
inline void Deserializer::read(std::vector< std::vector<T, A1>, A2>* array)
{
    size_t size;
    this->read_size_t(&size);
    array->resize(size);
    for (size_t i = 0u; i != size; ++i) {
        std::vector<T, A1> &inner = (*array)[i];

        size_t isize;
        this->read_size_t(&isize);
        inner.resize(isize);
        for (size_t j = 0u; j != isize; ++j) {
            T temp;
            this->read(&temp);
            inner[j] = temp;
        }
    }
}

template <typename T, typename SWO>
inline void Deserializer::read(std::set<T, SWO>* set)
{
    size_t size;
    set->clear();
    read_size_t(&size);
    T value;
    for (size_t i(0); i != size; ++i) {
        read(&value);
        // Values were serialized in sequence, so we can use end position as hint.
        // This should amount to an O(n) complexity for deserialization.
        set->insert(set->end(), value);
    }
}

template <typename T1, typename T2>
inline void Deserializer::read(std::pair<T1, T2>* pair)
{
    read(&pair->first);
    read(&pair->second);
}

template <typename T, typename A1, typename A2>
inline void Serializer::write(const std::vector< std::vector<T, A1>, A2>& array)
{
    size_t size = array.size();
    this->write_size_t(size);
    for (size_t i = 0; i < size; ++i) {
        const std::vector<T, A1> &inner = array[i];

        size_t isize = inner.size();
        this->write_size_t(isize);
        for (size_t j = 0; j < isize; ++j) {
            this->write(inner[j]);
        }
    }
}

template <typename S, typename>
inline void write(S* serial, bool value)   { serial->write( &value, 1 ); }

template <typename S, typename>
inline void write(S* serial, Uint8 value)  { serial->write( &value, 1 ); }

template <typename S, typename>
inline void write(S* serial, Uint16 value) { serial->write( &value, 1 ); }

template <typename S, typename>
inline void write(S* serial, Uint32 value) { serial->write( &value, 1 ); }

template <typename S, typename>
inline void write(S* serial, Uint64 value) { serial->write( &value, 1 ); }

template <typename S, typename>
inline void write(S* serial, Sint8 value)  { serial->write( &value, 1 ); }

template <typename S, typename>
inline void write(S* serial, Sint16 value) { serial->write( &value, 1 ); }

template <typename S, typename>
inline void write(S* serial, Sint32 value) { serial->write( &value, 1 ); }

template <typename S, typename>
inline void write(S* serial, Sint64 value) { serial->write( &value, 1 ); }

template <typename S, typename>
inline void write(S* serial, float value)  { serial->write( &value, 1 ); }

template <typename S, typename>
inline void write(S* serial, double value) { serial->write( &value, 1 ); }

template <typename T, typename S, typename, typename>
inline void write(S* serial, T value) { write_enum(serial,value); }

inline void write(
    Serializer* serial,
    const DB::Tag& value)
{
    serial->write( value );
}

template <class T>
inline void write(Serializer* serial, const DB::Typed_tag<T>& value)
{
    write(serial, value.get_untyped());
}

inline void write(Serializer* serial, const DB::Tag_version& value )
{
    write( serial, value.m_tag );
    write( serial, value.m_version );
    write( serial, value.m_transaction_id );
}

template <typename T, Size R, Size C, typename S, typename>
inline void write(S* serial, const mi::math::Matrix_struct<T,R,C>& value)
{
    serial->write(mi::math::matrix_base_ptr(value),R*C);
}

inline void write(Serializer* serial, const char* value)
{
    serial->write( value );
}

template <typename S, typename>
inline void write(S* serial, const std::string& value)
{
    write(serial,(Uint64)value.size()+1);       // silly, only for backwards compatibility
    serial->write(reinterpret_cast<const Uint8*>(value.c_str()),value.size());
}

template <typename S, typename>
inline void write(S* serial, const mi::base::Uuid& value)
{
    write(serial,value.m_id1);
    write(serial,value.m_id2);
    write(serial,value.m_id3);
    write(serial,value.m_id4);
}

template <typename T, Size DIM, typename S, typename>
inline void write(S* serial, const mi::math::Vector_struct<T,DIM>& value)
{
    serial->write(mi::math::vector_base_ptr(value),DIM);
}

template <typename T, Size DIM, typename S, typename>
inline void write(S* serial, const mi::math::Bbox_struct<T,DIM>& value)
{
    write(serial,value.min);
    write(serial,value.max);
}

template <typename T, Size DIM, typename S, typename>
inline void write(S* serial, const mi::math::Bbox<T,DIM>& value)
{
    write(serial,value.min);
    write(serial,value.max);
}

inline void write(Serializer* serial, const CONT::Bitvector& value)
{
    serial->write( value );
}

inline void write(Serializer* serial, const CONT::Dictionary& value)
{
    serial->write( value );
}

inline void write(Serializer* serial, const DB::Transaction_id& value)
{
    serial->write(value());
}

inline void write(Serializer* serial, const Serializable& value)
{
    serial->write( value );
}

inline void write(mi::neuraylib::ISerializer* serial, const mi::neuraylib::ISerializable& value)
{
    value.serialize(serial);
}

template <typename D, typename>
inline void read(D* deser, bool* value_pointer)   { deser->read( value_pointer, 1 ); }
template <typename D, typename>
inline void read(D* deser, Uint8* value_pointer)  { deser->read( value_pointer, 1 ); }
template <typename D, typename>
inline void read(D* deser, Uint16* value_pointer) { deser->read( value_pointer, 1 ); }
template <typename D, typename>
inline void read(D* deser, Uint32* value_pointer) { deser->read( value_pointer, 1 ); }
template <typename D, typename>
inline void read(D* deser, Uint64* value_pointer) { deser->read( value_pointer, 1 ); }
template <typename D, typename>
inline void read(D* deser, Sint8* value_pointer)  { deser->read( value_pointer, 1 ); }
template <typename D, typename>
inline void read(D* deser, Sint16* value_pointer) { deser->read( value_pointer, 1 ); }
template <typename D, typename>
inline void read(D* deser, Sint32* value_pointer) { deser->read( value_pointer, 1 ); }
template <typename D, typename>
inline void read(D* deser, Sint64* value_pointer) { deser->read( value_pointer, 1 ); }
template <typename D, typename>
inline void read(D* deser, float* value_pointer)  { deser->read( value_pointer, 1 ); }
template <typename D, typename>
inline void read(D* deser, double* value_pointer) { deser->read( value_pointer, 1 ); }
template <typename T, typename D, typename, typename>
inline void read(D* deser, T* enum_ptr) { read_enum(deser,enum_ptr); }

inline void read(Deserializer* deser, DB::Tag* value_pointer)
{
    deser->read( value_pointer );
}

template <class T>
inline void read(Deserializer* deser, DB::Typed_tag<T>* value_pointer)
{
    DB::Tag tag;
    read(deser, &tag );
    *value_pointer = tag;
}

inline void read(Deserializer* deser, DB::Tag_version* value_pointer)
{
    read( deser, &value_pointer->m_tag );
    read( deser, &value_pointer->m_version );
    read( deser, &value_pointer->m_transaction_id );
}


template <typename T, Size R, Size C, typename D, typename>
inline void read(D* deser, mi::math::Matrix_struct<T,R,C>* value_pointer)
{
    deser->read( mi::math::matrix_base_ptr(*value_pointer), R*C);
}

inline void read(Deserializer* deser, char** value_pointer)
{
    deser->read( value_pointer );
}

template <typename D, typename>
inline void read(D* deser, std::string* value_pointer)
{
    Uint64 size;
    read(deser,&size);
    value_pointer->resize(size-1);      // silly, only for backwards compatibility
    deser->read(reinterpret_cast<mi::Uint8*>((*value_pointer).data()),size-1);
}

template <typename D, typename>
inline void read(D* deser, mi::base::Uuid* value_pointer)
{
    read(deser,&value_pointer->m_id1);
    read(deser,&value_pointer->m_id2);
    read(deser,&value_pointer->m_id3);
    read(deser,&value_pointer->m_id4);
}

template <typename T, Size DIM, typename D, typename>
inline void read(D* deser, mi::math::Vector_struct<T,DIM>* value)
{
    deser->read(mi::math::vector_base_ptr(*value),DIM);
}

template <typename T, Size DIM, typename D, typename>
inline void read(D* deser, mi::math::Bbox_struct<T,DIM>* value)
{
    read(deser,&value->min);
    read(deser,&value->max);
}

template <typename T, Size DIM, typename D, typename>
inline void read(D* deser, mi::math::Bbox<T,DIM>* value)
{
    read(deser,&value->min);
    read(deser,&value->max);
}

inline void read(Deserializer* deser, CONT::Bitvector* value_type)
{
    deser->read( value_type );
}

inline void read(Deserializer* deser, CONT::Dictionary* value_pointer)
{
    deser->read( value_pointer );
}

inline void read(Deserializer* deser, DB::Transaction_id* value_pointer)
{
    Uint32 value;
    deser->read(&value);
    *value_pointer = value;
}

inline void read(Deserializer* deser, Serializable* value_type)
{
    deser->read( value_type );
}

inline void read(mi::neuraylib::IDeserializer* deser, mi::neuraylib::ISerializable* value_type)
{
    value_type->deserialize(deser);
}

template <class Iterator, typename D, typename>
inline void read_range(D& deserializer, Iterator begin, Iterator end)
{
    while (begin != end)
        read(&deserializer, &(*(begin++)));
}

template <typename T, size_t N, typename D, typename>
inline void read_range(D& deserializer, T (&arr)[N])
{
    read_range(deserializer,arr+0,arr+N);
}

template <class Iterator, typename S, typename>
inline void write_range(S& serializer, Iterator begin, Iterator end)
{
    while (begin != end)
        write(&serializer, *begin++);
}

template <typename T, size_t N, typename S, typename>
inline void write_range(S& serializer, const T (&arr)[N])
{
    write_range(serializer,arr+0,arr+N);
}

template <typename T, typename S, typename>
inline void write(S* serializer, const std::vector<T>& array)
{
    write(serializer,(mi::Uint64)array.size());
    write_range(*serializer, array.begin(), array.end());
}

template <typename T,typename A, typename S, typename>
inline void write(S* serializer, const std::vector<T,A>& array)
{
    write(serializer,(mi::Uint64)array.size());
    write_range(*serializer, array.begin(), array.end());
}

template <typename T, typename S, typename>
inline void write(S* serializer, const std::vector<T*>& array)
{
    const mi::Uint64 size(array.size());
    write(serializer,size);
    for (mi::Uint64 i=0; i != size; ++i)
        write(serializer,array[i]);
}

template <typename T,typename A, typename S, typename>
inline void write(S* serializer, const std::vector<T*,A>& array)
{
    const mi::Uint64 size(array.size());
    write(serializer,size);
    for (mi::Uint64 i=0; i != size; ++i)
        write(serializer,array[i]);
}

template <typename T, typename D, typename>
inline void read(D* deserializer, std::vector<T>* array)
{
    mi::Uint64 size;
    deserializer->read(&size);
    array->resize(size);
    read_range(*deserializer, array->begin(), array->end());
}

// needed because read_range is not compatible with bool vector iterators
template <typename D, typename = enable_if_deserializer_t<D>>
inline void read(D* deserializer, std::vector<bool>* array)
{
    mi::Uint64 size;
    deserializer->read(&size);
    array->resize(size);
    for (std::vector<bool>::iterator it = array->begin(), end = array->end(); it != end; ++it)
    {
        bool tmp;
        deserializer->read(&tmp);
        *it = tmp;
    }
}

template <typename T,typename A, typename D, typename>
inline void read(D* deserializer, std::vector<T,A>* array)
{
    mi::Uint64 size;
    deserializer->read(&size);
    array->resize(size);
    read_range(*deserializer, array->begin(), array->end());
}

template <typename T, typename D, typename>
inline void read(D* deserializer, std::vector<T*>* array)
{
    mi::Uint64 size;
    deserializer->read(&size);
    array->resize(size);
    for (size_t i=0; i != size; ++i)
        (*array)[i] = reinterpret_cast<T*>( deserializer->deserialize() );
}

template <typename T,typename A, typename D, typename>
inline void read(D* deserializer, std::vector<T*,A>* array)
{
    mi::Uint64 size;
    deserializer->read(&size);
    array->resize(size);
    for (size_t i=0; i != size; ++i)
        (*array)[i] = reinterpret_cast<T*>( deserializer->deserialize() );
}

template <typename T, typename S, typename>
inline void write(S* serializer, const std::list<T>& list)
{
    write(serializer,(mi::Uint64)list.size());
    for (typename std::list<T>::const_iterator i=list.begin(),e=list.end(); i!=e; ++i)
        write(serializer,*i);
}

template <typename T, typename D, typename>
inline void read(D* deserializer, std::list<T>* list)
{
    mi::Uint64 size;
    deserializer->read(&size);
    list->resize(size);
    for (typename std::list<T>::iterator i=list->begin(),e=list->end(); i!=e; ++i)
        read(deserializer,&*i);
}

template <typename T, typename U, typename S, typename>
inline void write(S* serializer, const std::pair<T, U>& pair)
{
    write(serializer, pair.first);
    write(serializer, pair.second);
}

template <typename T, typename U, typename D, typename>
inline void read(D* deserializer, std::pair<T, U>* pair)
{
    read(deserializer, &(pair->first));
    read(deserializer, &(pair->second));
}

template <typename T, typename SWO, typename S, typename>
void write(S* serializer, const std::set<T,SWO>& set)
{
    const size_t size(set.size());
    write(serializer,(mi::Uint64)size);
    write_range( *serializer, set.begin(), set.end() );
}

template <typename T, typename SWO, typename D, typename>
void read(D* deserializer, std::set<T,SWO>* set)
{
    set->clear();
    mi::Uint64 size;
    deserializer->read(&size);
    T value;
    for ( size_t i(0); i != size; ++i ) {
        read( deserializer, &value );
        // Values were serialized in sequence, so we can use end position as hint.
        // This should amount to an O(n) complexity for deserialization.
        set->insert( set->end(), value );
    }
}

template <typename T, typename U, typename SWO, typename S, typename>
void write(S* serializer, const std::map<T,U,SWO>& map)
{
    const size_t size(map.size());
    write(serializer,(mi::Uint64)size);
    write_range( *serializer, map.begin(), map.end() );
}

template <typename T, typename U, typename SWO, typename D, typename>
void read(D* deserializer, std::map<T,U,SWO>* map)
{
    map->clear();
    mi::Uint64 size;
    deserializer->read(&size);
    std::pair<T,U> value;
    for ( size_t i(0); i != size; ++i ) {
        read( deserializer, &value );
        // Values were serialized in sequence, so we can use end position as hint.
        // This should amount to an O(n) complexity for deserialization.
        map->insert( map->end(), value );
    }
}

template<class K, class V, class C, class A, typename S, typename>
void write(S* ser, const std::multimap<K,V,C,A>& map)
{
    const size_t size(map.size());
    write(ser,(mi::Uint64)size);
    write_range(*ser,map.begin(),map.end());
}

template<class K, class V, class C, class A, typename D, typename>
void read(D* deser, std::multimap<K,V,C,A>* map)
{
    map->clear();
    mi::Uint64 size;
    deser->read(&size);
    std::pair<K,V> value;
    for (size_t i(0); i != size; ++i) {
        read(deser,&value);
        // Values were serialized in sequence, so we can use end position as hint.
        // This should amount to an O(n) complexity for deserialization.
        map->insert(map->end(),value);
    }
}



template <typename Enum_type, typename S, typename>
void write_enum(S* serializer, Enum_type enum_value )
{
    write(serializer,static_cast<typename std::underlying_type<Enum_type>::type>(enum_value));
}


template <typename Enum_type, typename D, typename>
void read_enum(D* deserializer, Enum_type* enum_value )
{
    typename std::underlying_type<Enum_type>::type v;
    read(deserializer,&v);
    *enum_value = static_cast<Enum_type>(v);
}


namespace DETAIL {

template <std::size_t I, typename S, typename... Tp>
void write_variant(S* ser, const std::variant<Tp...>& val, const std::size_t idx)
{
    if constexpr (I < sizeof...(Tp)) {
        if (idx == 0) {
            const auto& element = std::get<I>(val);
            write(ser,element);
        }
        else {
            write_variant<I+1>(ser,val,idx-1);
        }
        return;
    }
    assert(!"invalid variant index");
}


template <std::size_t I, typename D, typename... Tp>
void read_variant(D* deser, std::variant<Tp...>* vp, const std::size_t idx)
{
    if constexpr (I < sizeof...(Tp)) {
        if (idx == 0) {
            auto& element = vp->template emplace<I>();
            read(deser,&element);
        }
        else {
            read_variant<I+1>(deser,vp,idx-1);
        }
        return;
    }
    assert(!"invalid variant index");
}

}


template <typename... Tp, typename S, typename>
void write(S* ser, const std::variant<Tp...>& val)
{
    write(ser,(mi::Uint64)val.index());
    if (!val.valueless_by_exception()) {
        // not available on MacOS and possibly poorly implemented elsewhere:
        // std::visit([&ser](const auto& v){ write(ser,v); },val);
        DETAIL::write_variant<0>(ser,val,val.index());
    }
}


template <typename... Tp, typename D, typename>
void read(D* deser, std::variant<Tp...>* vp)
{
    mi::Uint64 idx;
    read(deser,&idx);
    if (idx != std::variant_npos) {
        DETAIL::read_variant<0>(deser,vp,idx);
    }
}


template <typename T, std::size_t N, typename S, typename>
void write(S* ser, const std::array<T,N>& val)
{
    write_range(*ser,val.begin(),val.end());
}


template <typename T, std::size_t N, typename D, typename>
void read(D* deser, std::array<T,N>* vp)
{
    read_range(*deser,vp->begin(),vp->end());
}


} // namespace SERIAL

} // namespace MI

