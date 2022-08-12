//*****************************************************************************
// Copyright 2022 NVIDIA Corporation. All rights reserved.
//*****************************************************************************
/// \file
/// \brief      Scene element Volume
//*****************************************************************************

#ifndef MI_NEURAYLIB_IVOLUME_H
#define MI_NEURAYLIB_IVOLUME_H

#include <mi/neuraylib/iscene_element.h>
#include <mi/neuraylib/typedefs.h>

namespace mi {

typedef math::Bbox_struct<Sint32,3> Voxel_block_struct;
typedef math::Bbox<Sint32,3> Voxel_block;

namespace neuraylib {

/** \addtogroup mi_neuray_leaf_nodes
@{
*/
/// Interface representing a top-level volume.
///
/// IVolume represents a volume which encloses a single participating medium but (contrary to
/// meshes) has no hull. Volumes can be instanced and transformed as usual. Volumes displace the
/// vacuum of the scene but are themselves displaced by closed geometry.
class IVolume :
    public base::Interface_declare<0xdc35e746,0x3410,0x46b4,0x91,0x31,0xba,0xde,0x70,0x7d,0xa1,0xe3,
                                   neuraylib::IScene_element>
{
public:
    /// Sets the bounds of this object.
    ///
    /// Volume data outside of these bounds will not be visible.
    ///
    /// When using materials which directly look up a volume data file like VDB,
    /// it is useful to set this object's bounds based on the information provided
    /// by #mi::neuraylib::IVolume_data::get_data_bounds()
    /// and #mi::neuraylib::IVolume_data::get_transform().
    virtual void set_bounds(
            const Bbox3_struct& box,
            const Float32_4_4_struct& tf=Float32_4_4(1)) = 0;

    /// Retrieves the bounds of this object.
    ///
    /// If no bounds were set, this function returns the unit cube.
    ///
    /// \see #set_bounds()
    virtual Bbox3_struct get_bounds() const = 0;

    /// Retrieves the bounds transform of this object.
    ///
    /// If no bounds were set, this function returns the identity transform.
    ///
    /// \see #set_bounds()
    virtual Float32_4_4_struct get_transform() const = 0;

    /// This inline method exists for the user's convenience since #mi::math::Bbox
    /// is not derived from #mi::math::Bbox_struct.
    inline void set_bounds(
            const Bbox3& box,
            const Float32_4_4_struct& tf=Float32_4_4(1))
    {
        const Bbox3_struct box_struct = box;
        set_bounds( box_struct, tf);
    }
};

/**@}*/ // end group mi_neuray_leaf_nodes

/** \addtogroup mi_neuray_misc
@{
*/
/// Interface representing a volume data set.
///
/// IVolume_data represents a set of volume coefficients in three dimensional space.
/// It is conceptually similar to a 3d texture.
///
class IVolume_data :
    public base::Interface_declare<0xe0ce059f,0x51cf,0x4fef,0x8c,0x87,0x91,0x93,0x16,0x59,0xa5,0x44,
                                   neuraylib::IScene_element>
{
public:

    /// Sets the volume data to a file identified by \p filename.
    ///
    /// The optional \p selector string encodes the name of the data set (grid) within the given
    /// file. If no selector is provided a default is selected automatically.
    ///
    /// \return
    ///                       -  0: Success.
    ///                       - -1: Invalid parameters (\c NULL pointer).
    ///                       - -2: Failure to resolve the given filename, e.g., the file does not
    ///                             exist.
    ///                       - -3: Failure to open the file.
    ///                       - -4: No plugin found to handle the file.
    ///                       - -5: The plugin failed to import the file.
    virtual Sint32 reset_file( const char* filename, const char* selector=0) = 0;

    /// Returns the resolved file name of the file containing the volume data.
    ///
    /// This function returns \c NULL if there is no file associated with the data, e.g., after
    /// default construction, or failures to resolve the file name passed to #reset_file().
    ///
    /// \see #get_original_filename()
    virtual const char* get_filename() const = 0;

    /// Returns the unresolved file as passed to #reset_file().
    ///
    /// This function returns \c NULL after default construction.
    ///
    /// \see #get_filename()
    virtual const char* get_original_filename() const = 0;

    /// Returns the name of the file's data selector selected via #reset_file().
    ///
    /// This function returns \c NULL after default construction.
    virtual const char* get_selector() const = 0;

#ifdef MI_NEURAYLIB_DEPRECATED_12_1
    inline const char* get_channel_name() const { return get_selector(); }
#endif // MI_NEURAYLIB_DEPRECATED_12_1

    /// Returns the bounding box stored in the current data set.
    ///
    /// This function returns the bounds of the data grid contained in the file. Bounds
    /// are in voxel space and thus integer coordinates. The transformation from voxel space
    /// to world space may be obtained by calling #get_transform().
    ///
    /// This function returns an empty box when no file was loaded, either because #reset_file
    /// was not yet called, or because loading failed.
    ///
    /// \see #get_transform()
    virtual Voxel_block_struct get_data_bounds() const = 0;

    /// Returns the internal transformation from voxel space to world space.
    ///
    /// This function returns the volume file's internal transformation from voxel space
    /// to world space, or identity if no such transform is available.
    ///
    /// \see #get_data_bounds()
    virtual Float32_4_4_struct get_transform() const = 0;

};

/**@}*/ // end group mi_neuray_misc

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IVOLUME_H
