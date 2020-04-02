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
/// \brief      Scene element Bsdf_measurement

#ifndef MI_NEURAYLIB_IBSDF_MEASUREMENT_H
#define MI_NEURAYLIB_IBSDF_MEASUREMENT_H

#include <mi/neuraylib/iscene_element.h>

namespace mi {

namespace neuraylib {

class IBsdf_isotropic_data;
class IReader;

/** \addtogroup mi_neuray_misc
@{
*/

/// A scene element that stores measured BSDF data.
///
/// The measured BSDF is split into the two components for reflection and transmission. Currently,
/// only isotropic data is supported (see the abstract interface
/// #mi::neuraylib::IBsdf_isotropic_data). The data can be imported from file or can be passed via
/// the API.
///
/// A BSDF measurement appears in the scene as an argument of an MDL function call (see
/// #mi::neuraylib::IFunction_call) or default argument of an MDL function definition (see
/// #mi::neuraylib::IFunction_definition). The type of such an argument is
/// #mi::neuraylib::IType_bsdf_measurement or an alias of it.
///
/// \see #mi::neuraylib::IBsdf_isotropic_data, #mi::neuraylib::IBsdf_buffer
class IBsdf_measurement :
    public base::Interface_declare<0xa05e5a42,0x3f74,0x4ad9,0x8e,0xa9,0x17,0x4f,0x97,0x52,0x39,0x8a,
                                   neuraylib::IScene_element>
{
public:
    /// Sets the BSDF measurement to a file identified by \p filename.
    ///
    /// \return
    ///                   -  0: Success.
    ///                   - -1: Invalid parameters (\c NULL pointer).
    ///                   - -2: Failure to resolve the given filename, e.g., the file does not
    ///                         exist.
    ///                   - -3: Invalid file format or invalid filename extension (only \c .mbsdf is
    ///                         supported).
    virtual Sint32 reset_file( const char* filename) = 0;

    /// Sets the BSDF measurement to the data provided by a reader.
    ///
    /// \param reader      The reader that provides the data for the BSDF measurement in \c .mbsdf
    ///                    format.
    /// \return
    ///                    -  0: Success.
    ///                    - -1: Invalid parameters (\c NULL pointer).
    ///                    - -3: Invalid file format.
    virtual Sint32 reset_reader( IReader* reader) = 0;

    /// Returns the resolved file name of the file containing the BSDF measurement.
    ///
    /// The method returns \c NULL if there is no file associated with the BSDF measurement, e.g.,
    /// after default construction, calls to #set_reflection() or #set_transmission(), or failures
    /// to resolves the file name passed to #reset_file().
    ///
    /// \see #get_original_filename()
    virtual const char* get_filename() const = 0;

    /// Returns the unresolved file name as passed to #reset_file().
    ///
    /// The method returns \c NULL after default construction or calls to #set_reflection() or
    /// #set_transmission().
    ///
    /// \see #get_filename()
    virtual const char* get_original_filename() const = 0;

    /// Sets the BSDF data for the reflection.
    ///
    /// \param bsdf_data   The BSDF data to be used by this BSDF measurement. The value \p NULL
    ///                    can be used to remove the BSDF data for reflection.
    /// \return
    ///                    -  0: Success.
    ///                    - -2: The resolution or type of \p bsdf_data is invalid.
    virtual Sint32 set_reflection( const IBsdf_isotropic_data* bsdf_data) = 0;

    /// Returns the BSDF data for the reflection.
    ///
    /// Note that it is not possible to manipulate the BSDF data.
    ///
    /// \return            The BSDF data for reflection, or \p NULL if there is none.
    virtual const base::IInterface* get_reflection() const = 0;

    /// Returns the BSDF data for reflection.
    ///
    /// Note that it is not possible to manipulate the BSDF data.
    ///
    /// This templated member function is a wrapper of the non-template variant for the user's
    /// convenience. It eliminates the need to call
    /// #mi::base::IInterface::get_interface(const Uuid &)
    /// on the returned pointer, since the return type already is a const pointer to the type \p T
    /// specified as template parameter.
    ///
    /// \tparam T           The interface type of the requested element.
    /// \return             The BSDF data for reflection, or \c NULL on failure (no BSDF data
    ///                     available or if \p T does not match the element's  type).
    template<class T>
    const T* get_reflection() const
    {
        const base::IInterface* ptr_iinterface = get_reflection();
        if( !ptr_iinterface)
            return 0;
        const T* ptr_T = static_cast<const T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    /// Sets the BSDF data for transmission.
    ///
    /// \param bsdf_data   The BSDF data to be used by this BSDF measurement. The value \p NULL
    ///                    can be used to remove the BSDF data for transmission.
    /// \return
    ///                    -  0: Success.
    ///                    - -2: The resolution or type of \p bsdf_data is invalid.
    virtual Sint32 set_transmission( const IBsdf_isotropic_data* bsdf_data) = 0;

    /// Returns the BSDF data for transmission.
    ///
    /// Note that it is not possible to manipulate the BSDF data.
    ///
    /// \return            The BSDF data for transmission, or \p NULL if there is none.
    virtual const base::IInterface* get_transmission() const = 0;

    /// Returns the BSDF data for transmission.
    ///
    /// Note that it is not possible to manipulate the BSDF data.
    ///
    /// This templated member function is a wrapper of the non-template variant for the user's
    /// convenience. It eliminates the need to call
    /// #mi::base::IInterface::get_interface(const Uuid &)
    /// on the returned pointer, since the return type already is a const pointer to the type \p T
    /// specified as template parameter.
    ///
    /// \tparam T          The interface type of the requested element.
    /// \return            The BSDF data for transmission, or \c NULL on failure (no BSDF data
    ///                    available or if \p T does not match the element's  type).
    template<class T>
    const T* get_transmission() const
    {
        const base::IInterface* ptr_iinterface = get_transmission();
        if( !ptr_iinterface)
            return 0;
        const T* ptr_T = static_cast<const T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }
};

/*@}*/ // end group mi_neuray_misc

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IBSDF_MEASUREMENT_H
