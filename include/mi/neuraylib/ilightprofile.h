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
/// \brief      Scene element Lightprofile

#ifndef MI_NEURAYLIB_ILIGHTPROFILE_H
#define MI_NEURAYLIB_ILIGHTPROFILE_H

#include <mi/neuraylib/iscene_element.h>

namespace mi {

namespace neuraylib {

/** \addtogroup mi_neuray_misc
@{
*/

class IReader;

/// Ordering of horizontal angles in a light profile
///
/// The flags can be used to override the horizontal sample order in an IES file
/// [\ref IES02]. There are two IES file types in common use, type B and type C. The IES
/// standard defines that samples are stored in counter-clockwise order. Type C files conform
/// to this standard, but about 30% of the type B files deviate from the standard and store
/// samples in clockwise order, without giving any indication in the IES file that could be
/// used to switch the order. (Sometimes there is an informal comment.) Type A IES files are
/// not supported.
///
/// \see #mi::neuraylib::ILightprofile::reset_file(), #mi::neuraylib::ILightprofile::get_flags()
enum Lightprofile_flags {
    /// Clockwise order, contrary to the IES standard for these (incorrect) type B files.
    LIGHTPROFILE_CLOCKWISE            = 1,
    /// Counter-clockwise, standard-conforming order (default).
    LIGHTPROFILE_COUNTER_CLOCKWISE    = 2,
    /// For 3dsmax
    LIGHTPROFILE_ROTATE_TYPE_B        = 4,
    /// For 3dsmax
    LIGHTPROFILE_ROTATE_TYPE_C_90_270 = 8,
    LIGHTPROFILE_FLAGS_FORCE_32_BIT   = 0xffffffffU
};

mi_static_assert( sizeof( Lightprofile_flags) == sizeof( Uint32));

/// Degree of hermite interpolation.
///
/// Currently only linear (hermite 1) and cubic (hermite 3) degree are supported
/// (see also [\ref DH05]).
///
/// \see #mi::neuraylib::ILightprofile::reset_file(), #mi::neuraylib::ILightprofile::get_degree()
enum Lightprofile_degree {
    LIGHTPROFILE_HERMITE_BASE_1      = 1,   ///< Degree 1 = linear interpolation
    LIGHTPROFILE_HERMITE_BASE_3      = 3,   ///< Degree 3 = cubic interpolation
    LIGHTPROFILE_DEGREE_FORCE_32_BIT = 0xffffffffU
};

mi_static_assert( sizeof( Lightprofile_degree) == sizeof( Uint32));

/// This interface represents light profiles.
///
/// IES light profiles ([\ref IES02]) are supplied by lamp vendors to describe their products. They
/// contain a rectangular grid of measured light intensities.
///
/// A light profile appears in the scene as an argument of an MDL function call (see
/// #mi::neuraylib::IFunction_call) or default argument of an MDL function definition (see
/// #mi::neuraylib::IFunction_definition). The type of such an argument is
/// #mi::neuraylib::IType_light_profile or an alias of it.
class ILightprofile :
    public base::Interface_declare<0xa4ac11fd,0x705d,0x4a0a,0x80,0x0b,0x38,0xe5,0x3d,0x46,0x96,0x47,
                                   neuraylib::IScene_element>
{
public:
    /// Sets the light profile to a file identified by \p filename.
    ///
    /// \param filename           The new file containing the light profile data.
    /// \param resolution_phi     The desired resolution of the equidistant grid in phi-direction.
    ///                           \n
    ///                           The special value 0 leaves the choice of a suitable resolution to
    ///                           the implementation. Currently, the implementation behaves as
    ///                           follows: If the angles in phi-direction are already equidistant,
    ///                           the resolution in the file itself (after unfolding of symmetries)
    ///                           is kept unchanged. If the angles in phi-direction are not
    ///                           equidistant, a suitable resolution that maintains the angles given
    ///                           in the file is chosen. If that fails, a fixed resolution is
    ///                           chosen.
    /// \param resolution_theta   The desired resolution of the equidistant grid in theta-direction.
    ///                           \n
    ///                           The special value 0 leaves the choice of a suitable resolution to
    ///                           the implementation. Currently, the implementation behaves as
    ///                           follows: If the angles in theta-direction are already equidistant,
    ///                           the resolution in the file itself (after unfolding of symmetries)
    ///                           is kept unchanged. If the angles in theta-direction are not
    ///                           equidistant, a suitable resolution that maintains the angles given
    ///                           in the file is chosen. If that fails, a fixed resolution is
    ///                           chosen.
    /// \param degree             The interpolation method to use.
    /// \param flags              Flags to be used when interpreting the file data,
    ///                           see #mi::neuraylib::Lightprofile_flags for details.
    /// \return
    ///                           -  0: Success.
    ///                           - -1: Invalid parameters (\c NULL pointer).
    ///                           - -2: Failure to resolve the given filename, e.g., the file does
    ///                                 not exist.
    ///                           - -3: \p degree or \p flags is invalid (exactly one of
    ///                                 #mi::neuraylib::LIGHTPROFILE_CLOCKWISE or
    ///                                 #mi::neuraylib::LIGHTPROFILE_COUNTER_CLOCKWISE must be set).
    ///                           - -4: File format error.
    ///                           - -5: \p resolution_phi or \p resolution_theta is invalid (must
    ///                                 not be 1).
    virtual Sint32 reset_file(
        const char* filename,
        Uint32 resolution_phi = 0,
        Uint32 resolution_theta = 0,
        Lightprofile_degree degree = LIGHTPROFILE_HERMITE_BASE_1,
        Uint32 flags = LIGHTPROFILE_COUNTER_CLOCKWISE) = 0;

    /// Sets the light profile to the data provided by a reader.
    ///
    /// \param reader             The reader that provides the data for the BSDF measurement in
    ///                           \c .ies format.
    /// \param resolution_phi     The desired resolution of the equidistant grid in phi-direction.
    ///                           \n
    ///                           The special value 0 leaves the choice of a suitable resolution to
    ///                           the implementation. Currently, the implementation behaves as
    ///                           follows: If the angles in phi-direction are already equidistant,
    ///                           the resolution in the file itself (after unfolding of symmetries)
    ///                           is kept unchanged. If the angles in phi-direction are not
    ///                           equidistant, a suitable resolution that maintains the angles given
    ///                           in the file is chosen. If that fails, a fixed resolution is
    ///                           chosen.
    /// \param resolution_theta   The desired resolution of the equidistant grid in theta-direction.
    ///                           \n
    ///                           The special value 0 leaves the choice of a suitable resolution to
    ///                           the implementation. Currently, the implementation behaves as
    ///                           follows: If the angles in theta-direction are already equidistant,
    ///                           the resolution in the file itself (after unfolding of symmetries)
    ///                           is kept unchanged. If the angles in theta-direction are not
    ///                           equidistant, a suitable resolution that maintains the angles given
    ///                           in the file is chosen. If that fails, a fixed resolution is
    ///                           chosen.
    /// \param degree             The interpolation method to use.
    /// \param flags              Flags to be used when interpreting the data,
    ///                           see #mi::neuraylib::Lightprofile_flags for details.
    /// \return
    ///                           -  0: Success.
    ///                           - -1: Invalid parameters (\c NULL pointer).
    ///                           - -3: \p degree or \p flags is invalid (exactly one of
    ///                                 #mi::neuraylib::LIGHTPROFILE_CLOCKWISE or
    ///                                 #mi::neuraylib::LIGHTPROFILE_COUNTER_CLOCKWISE must be set).
    ///                           - -4: File format error.
    ///                           - -5: \p resolution_phi or \p resolution_theta is invalid (must
    ///                                 not be 1).
    virtual Sint32 reset_reader(
        IReader* reader,
        Uint32 resolution_phi = 0,
        Uint32 resolution_theta = 0,
        Lightprofile_degree degree = LIGHTPROFILE_HERMITE_BASE_1,
        Uint32 flags = LIGHTPROFILE_COUNTER_CLOCKWISE) = 0;

    /// Returns the resolved file name of the file containing the light profile.
    ///
    /// The method returns \c NULL if there is no file associated with the light profile, e.g.,
    /// after default construction or failures to resolve the file name passed to #reset_file().
    ///
    /// \see #get_original_filename()
    virtual const char* get_filename() const = 0;

    /// Returns the unresolved file name as passed to #reset_file().
    ///
    /// The method returns \c NULL after default construction.
    ///
    /// \see #get_filename()
    virtual const char* get_original_filename() const = 0;

    /// Returns the resolution of the grid in phi-direction, or 0 after default construction.
    virtual Uint32 get_resolution_phi() const = 0;

    /// Returns the resolution of the grid in theta-direction, or 0 after default construction.
    virtual Uint32 get_resolution_theta() const = 0;

    /// Returns the interpolation degree that was used to interpolate the grid data, or
    /// #mi::neuraylib::LIGHTPROFILE_HERMITE_BASE_1 after default construction.
    virtual Lightprofile_degree get_degree() const = 0;

    /// Returns flags that were used to interpret the light profile data in the file, or
    /// #mi::neuraylib::LIGHTPROFILE_COUNTER_CLOCKWISE after default construction.
    ///
    /// \see #mi::neuraylib::Lightprofile_flags.
    virtual Uint32 get_flags() const = 0;

    /// Returns the \p index -th phi value.
    ///
    /// Note that the grid is an equidistant grid, i.e., the distance between subsequent phi values
    /// is always the same. If \p index is out of bounds or after default construction, 0 is
    /// returned.
    virtual Float32 get_phi( Uint32 index) const = 0;

    /// Returns the \p index -th theta value.
    ///
    /// Note that the grid is an equidistant grid, i.e., the distance between subsequent theta
    /// values is always the same. If \p index is out of bounds or after default construction, 0 is
    /// returned.
    virtual Float32 get_theta( Uint32 index) const = 0;

    /// Returns the normalized data of the entire grid.
    ///
    /// \return   A pointer to the normalized data for all vertices of the grid. The data values are
    ///           stored as array in column-major order (where all elements of a column have the
    ///           same phi value). Returns \c NULL after default construction.
    ///
    /// \see #get_candela_multiplier()
    virtual const Float32* get_data() const = 0;

    /// Returns the normalized data for a grid vertex.
    ///
    /// \param index_phi     Index in phi-direction of the vertex.
    /// \param index_theta   Index in theta-direction of the vertex.
    /// \return              The normalized data for the grid point, or 0 in case of errors or after
    ///                      default construction.
    ///
    /// \see #get_candela_multiplier(), #sample()
    virtual Float32 get_data( Uint32 index_phi, Uint32 index_theta) const = 0;

    /// Returns the normalization factor.
    ///
    /// All data is normalized such that the maximum is 1.0. The values returned by methods like
    /// #get_data() need to be multiplied by this normalization factor to retrieve the true value.
    /// Returns 0 after default construction.
    virtual Float64 get_candela_multiplier() const = 0;

    /// Samples the light profile.
    ///
    /// The method computes a bi-linear interpolation of the light profile at (phi,theta) according
    /// to the resolution.
    ///
    /// \param phi       First dimension of sample point.
    /// \param theta     Second dimension of sample point.
    /// \param candela   If \c false, normalized values are returned, otherwise true values.
    ///                  See #get_candela_multiplier().
    /// \return          The computed sample value, or 0 in case of errors or after default
    ///                  construction.
    ///
    /// \see #get_data()
    virtual Float32 sample( Float32 phi, Float32 theta, bool candela) const = 0;
};

/*@}*/ // end group mi_neuray_misc

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_ILIGHTPROFILE_H
