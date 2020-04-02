/***************************************************************************************************
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief      Example implementations for abstract interfaces related to
///             scene element Bsdf_measurement

#ifndef MI_NEURAYLIB_BSDF_ISOTROPIC_DATA_H
#define MI_NEURAYLIB_BSDF_ISOTROPIC_DATA_H

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/neuraylib/ibsdf_isotropic_data.h>

namespace mi {

namespace neuraylib {

/** \addtogroup mi_neuray_misc
@{
*/

/// Example implementation of the abstract interface #mi::neuraylib::IBsdf_buffer.
///
/// The size of the memory block is specified in the constructor and cannot be changed later. This
/// simple implementation simply contains the memory block returned by #get_data().
///
/// \see #mi::neuraylib::IBsdf_buffer
class Bsdf_buffer : public mi::base::Interface_implement<neuraylib::IBsdf_buffer>
{
public:
    /// Constructor.
    ///
    /// \param size   The number of #mi::Float32 elements in the memory block.
    Bsdf_buffer( Size size) : m_buffer( new Float32[size]()) { }

    /// Destructor.
    ~Bsdf_buffer() { delete[] m_buffer; }

    /// Gives access to the memory block (const).
    const Float32* get_data() const { return m_buffer; }

    /// Gives access to the memory block (mutable).
    Float32* get_data() { return m_buffer; }

private:
    Float32* m_buffer;
};

/// Example implementation of the abstract interface #mi::neuraylib::IBsdf_isotropic_data.
///
/// The resolution and type of the BSDF data are specified in the constructor and cannot be changed
/// later. This simple implementation creates (the interface owning) the memory block holding all
/// values in its constructor and keeps it for its lifetime. More advanced implementations might
/// convert the data from other representations on the fly in #get_bsdf_buffer() and might return a
/// temporary instance of #mi::neuraylib::IBsdf_buffer without keeping a reference to that instance.
///
/// \see #mi::neuraylib::IBsdf_isotropic_data
class Bsdf_isotropic_data : public mi::base::Interface_implement<neuraylib::IBsdf_isotropic_data>
{
public:
    /// Constructor.
    Bsdf_isotropic_data( Uint32 resolution_theta, Uint32 resolution_phi, Bsdf_type type)
      : m_resolution_theta( resolution_theta),
        m_resolution_phi( resolution_phi),
        m_type( type)
    {
        Size size = resolution_theta * resolution_theta * resolution_phi;
        if( type == BSDF_RGB)
            size *= 3;
        m_bsdf_buffer = new Bsdf_buffer( size);
    }

    /// Returns the number of values in theta direction.
    Uint32 get_resolution_theta() const { return m_resolution_theta; }

    /// Returns the number of values in phi direction.
    Uint32 get_resolution_phi() const { return m_resolution_phi; }

    /// Returns the type of the values.
    Bsdf_type get_type() const { return m_type; }

    /// Returns the buffer containing the values (const).
    const Bsdf_buffer* get_bsdf_buffer() const
    { m_bsdf_buffer->retain(); return m_bsdf_buffer.get(); }

    /// Returns the buffer containing the values (mutable).
    Bsdf_buffer* get_bsdf_buffer()
    { m_bsdf_buffer->retain(); return m_bsdf_buffer.get(); }

private:
    Uint32 m_resolution_theta;
    Uint32 m_resolution_phi;
    Bsdf_type m_type;
    base::Handle<Bsdf_buffer> m_bsdf_buffer;
};

/*@}*/ // end group mi_neuray_misc

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_BSDF_ISOTROPIC_DATA_H
