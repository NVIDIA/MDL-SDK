/***************************************************************************************************
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
 **************************************************************************************************/
/// \file
/// \brief Spectrum type.

#ifndef MI_NEURAYLIB_ISPECTRUM_H
#define MI_NEURAYLIB_ISPECTRUM_H

#include <mi/math/spectrum.h>
#include <mi/neuraylib/icompound.h>
#include <mi/neuraylib/typedefs.h>

namespace mi {

/** \addtogroup mi_neuray_compounds
@{
*/

/// This interface represents spectrums.
///
/// It can be used to represent spectrums by an interface derived from #mi::base::IInterface.
///
/// \see #mi::Spectrum_struct
class ISpectrum :
    public base::Interface_declare<0x127293dc,0x1fad,0x4df5,0x94,0x38,0xe3,0x48,0xda,0x7b,0x8c,0xf6,
                                   ICompound>
{
public:
    /// Returns the spectrum represented by this interface.
    virtual Spectrum_struct get_value() const = 0;

    /// Returns the spectrum represented by this interface.
    virtual void get_value( Spectrum_struct& value) const = 0;

    /// Sets the spectrum represented by this interface.
    virtual void set_value( const Spectrum_struct& value) = 0;

    using ICompound::get_value;

    using ICompound::set_value;
};

/*@}*/ // end group mi_neuray_compounds

} // namespace mi

#endif // MI_NEURAYLIB_ISPECTRUM_H
