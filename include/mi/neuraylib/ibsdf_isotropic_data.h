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
/// \brief      Abstract interfaces related to scene element Bsdf_measurement

#ifndef MI_NEURAYLIB_IBSDF_ISOTROPIC_DATA_H
#define MI_NEURAYLIB_IBSDF_ISOTROPIC_DATA_H

#include <mi/base/interface_declare.h>

namespace mi {

namespace neuraylib {

class IBsdf_buffer;

/** \addtogroup mi_neuray_misc
@{
*/

/// The BSDF type.
enum Bsdf_type {
    BSDF_SCALAR = 0, ///< One scalar per grid value.
    BSDF_RGB    = 1, ///< Three scalars (RGB) per grid value.
    BSDF_TYPES_FORCE_32_BIT = 0xffffffffU // Undocumented, for alignment only
};

mi_static_assert( sizeof( Bsdf_type) == sizeof( Uint32));

/// Abstract interface for isotropic BSDF data.
///
/// The isotropic BSDF data is modeled as a three-dimensional grid of values. The three dimensions
/// of the grid are called \c theta_in, \c theta_out, and \c phi_in. The values can be of two types:
/// scalars or RGB values (see #Bsdf_type). The grid values are uniformly distributed in the range
/// [0,pi/2) for \c theta_in and \c theta_out and in the range [0,pi] for \c phi_in. The resolution,
/// i.e., the number of values, of each dimension is arbitrary with the limitation that the
/// resolution for \c theta_in and \c theta_out has to be identical.
///
/// \see #mi::neuraylib::IBsdf_measurement and #mi::neuraylib::IBsdf_buffer for related interfaces
/// \see #mi::neuraylib::Bsdf_isotropic_data for an example implementation of this interface
class IBsdf_isotropic_data : public
    mi::base::Interface_declare<0x23fd6d83,0x057b,0x4507,0xb4,0x93,0x0e,0xbd,0x44,0x7b,0x07,0xb9>
{
public:
    /// Returns the number of values in theta direction.
    virtual Uint32 get_resolution_theta() const = 0;

    /// Returns the number of values in phi direction.
    virtual Uint32 get_resolution_phi() const = 0;

    /// Returns the type of the values.
    virtual Bsdf_type get_type() const = 0;

    /// Returns the buffer containing the actual values.
    virtual const IBsdf_buffer* get_bsdf_buffer() const = 0;
};

/// Abstract interface for a buffer of BSDF values.
///
/// \see #mi::neuraylib::IBsdf_measurement and #mi::neuraylib::IBsdf_isotropic_data for related
/// interfaces
/// \see #mi::neuraylib::Bsdf_buffer for an example implementation of this interface
class IBsdf_buffer : public
    mi::base::Interface_declare<0xdf3e6121,0x464e,0x424b,0x87,0x6f,0x6e,0x8f,0x6e,0x66,0xe2,0x9a>
{
public:
    /// Returns the memory block containing the actual BSDF values.
    ///
    /// The size of the array is given by
    /// \code
    ///   res_theta^2 * res_phi * factor
    /// \endcode
    /// where \c res_phi is the value returned by
    /// #mi::neuraylib::IBsdf_isotropic_data::get_resolution_phi(), \c res_theta is the value
    /// returned by #mi::neuraylib::IBsdf_isotropic_data::get_resolution_theta(), and \c factor is 1
    /// if #mi::neuraylib::IBsdf_isotropic_data::get_type() returns #mi::neuraylib::BSDF_SCALAR and
    /// 3 otherwise.
    ///
    /// The index of the (first) element for a particular triple
    /// (\c index_theta_in, \c index_theta_out, \c index_phi_in) is given by
    /// \code
    ///   factor * (index_theta_in * (res_phi * res_theta) + index_theta_out * res_phi + index_phi)
    /// \endcode
    ///
    /// The pointer returned by this method has to be valid for the lifetime of this interface.
    virtual const Float32* get_data() const = 0;
};

/*@}*/ // end group mi_neuray_misc

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IBSDF_ISOTROPIC_DATA_H
