/***************************************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief      API component that gives access to the MDL Encapsulator API.

#ifndef MI_NEURAYLIB_IMDLE_API_H
#define MI_NEURAYLIB_IMDLE_API_H

#include <mi/base/interface_declare.h>
#include <mi/base/enums.h>

namespace mi {

class IStructure;

namespace neuraylib {

class IMdl_execution_context;
class IReader;
class ITransaction;

/** \addtogroup mi_neuray_mdl_types
@{
*/

/// Provides access to functions related to the creation of encapsulated MDL modules (MDLE).
class IMdle_api : public
    base::Interface_declare<0xda82106c,0x658d,0x449d,0x8e,0x1,0xfb,0x55,0x1,0x61,0x9b,0x97>
{
public:

    /// Exports a new MDLE file to disk.
    ///
    /// \param transaction  The transaction to be used.
    /// \param file_name    The new filename (including the .mdle extension).
    /// \param mdle_data    A structure of type \c Mdle_data.
    ///                     This structure has the following members:
    ///                        - #mi::IString \b prototype_name \n
    ///                          The DB name of the prototype for the main material or function
    ///                          definition of the MDLE file. The prototype
    ///                          can be a material definition, a function definition, a
    ///                          material instance or a function call.
    ///                        - #mi::neuraylib::IExpression_list* \b defaults \n
    ///                          The defaults of the main material or function definition of the
    ///                          MDLE file.
    ///                          The type of an argument in the expression list must match the type
    ///                          of the corresponding parameter of the prototype. \n
    ///                          If the list is empty, the resulting MDLE will have no parameters.
    ///                          If \c NULL is passed, the MDLE will inherit the defaults of the
    ///                          prototype.
    ///                        - #mi::neuraylib::IAnnotation_block* \b annotations \n
    ///                          Annotations of the main material or function definition of the
    ///                          MDLE file.
    ///                          If the list is empty, the resulting MDLE will have no annotations.
    ///                          If \c NULL is passed, the MDLE will inherit the annotations of the
    ///                          prototype. Please note that parameter and return type annotations
    ///                          are always inherited from the prototype.
    ///                        - #mi::IString \b thumbnail_path \n
    ///                          Path to a thumbnail image representing the exported material or
    ///                          function.
    ///                          Can be either an absolute MDL url or a file system path.
    ///                        - #mi::base::IInterface* \b user_files \n
    ///                          A static or dynamic array of structures of type \c Mdle_user_file
    ///                          pointing to additional user content (files) that should be added to
    ///                          the MDLE archive. Can be \c NULL.\n
    ///                          The structure has the two members
    ///                             - #mi::IString \b source_path \n
    ///                               MDL url or file system path pointing to the file.
    ///                             - #mi::IString \b target_path \n
    ///                               New path of the file in the archive.\n
    /// \param context      An execution context which can be queried
    ///                     for detailed error messages after the
    ///                     operation has finished. Can be \c NULL.
    ///
    /// \return  
    ///                     -   0: Success
    ///                     -  -1: An error occurred. If provided, please check
    ///                            the context for details.
    virtual Sint32 export_mdle(
        ITransaction* transaction,
        const char* file_name,
        const IStructure* mdle_data,
        IMdl_execution_context* context) const = 0;

    /// Checks the integrity of an MDLE file based on MD5 hashes
    /// that are stored for the contained files.
    ///
    /// \param file_name    The file name of the MDLE to check.
    /// \param context      An execution context which can be queried for detailed error messages
    ///                     after the operation has finished.
    ///                     Can be \c NULL.
    /// \return
    ///                     -   0: Success
    ///                     -  -1: The MDLE file is invalid. If provided, please check
    ///                            the context for details.
    virtual Sint32 validate_mdle(
        const char* file_name,
        IMdl_execution_context* context) const = 0;

    /// Get a user file that has been added to an MDLE during its creation.
    ///
    /// \param mdle_file_name   The file name of the MDLE that contains the user file.
    /// \param user_file_name   The path and name of the file to read inside the MDLE.
    ///                         This equals the \b target_path during the creation.
    /// \param context          An execution context which can be queried for detailed error 
    ///                         messages after the operation has finished.
    ///                         Can be \c NULL.
    /// \return
    ///                         A reader with access to the user file content or NULL in case of
    ///                         errors. Check the context for details in that case.
    virtual IReader* get_user_file(
        const char* mdle_file_name,
        const char* user_file_name,
        IMdl_execution_context* context) const = 0;

    /// Check if two MDLE are identical, meaning that they contain the same content
    /// independent of their file path.
    ///
    /// \param mdle_file_name_a   The file name of the first MDLE to compare.
    /// \param mdle_file_name_b   The file name of the second MDLE to compare.
    /// \param context            An execution context which can be queried for detailed error
    ///                           messages after the operation has finished.
    ///                           Can be \c NULL.
    /// \return
    ///                           -   0: Success
    ///                           -  -1: The files are different or at least one is not existing.
    ///                                  If provided, please check the context for details.
    virtual Sint32 compare_mdle(
        const char* mdle_file_name_a,
        const char* mdle_file_name_b,
        IMdl_execution_context* context) const = 0;

    /// Extracts the hash of the MDLE archive.
    ///
    /// \param mdle_file_name   The file name of the MDLE.
    /// \param[out] hash        The returned hash value.
    /// \param context          An execution context which can be queried for detailed error
    ///                         messages after the operation has finished.
    ///                         Can be \c NULL.
    /// \return
    ///                          -   0: Success
    ///                          -  -1: An error occurred. If provided,
    ///                                 please check the context for details.
    virtual Sint32 get_hash(
        const char* mdle_file_name,
        base::Uuid& hash,
        IMdl_execution_context* context) const = 0;
};

/*@}*/ // end group mi_neuray_mdl_types

} // namespace neuraylib
} // namespace mi

#endif // MI_NEURAYLIB_IMDLE_API_H
