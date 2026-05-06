# Introduction

Material Definition Language (MDL) enables users to define the reflective, transmissive, emissive
and volumetric properties of objects. For details regarding the elemental distribution functions
and operations, please see the MDL specification.

The `base` module provides a set of texturing functions covering the whole range from bitmap to
procedural texturing. In addition, module `base.mdl` provides helper functions to make some common
tasks easy, as well as provide backwards compatibility support for some legacy parameter semantics.

# Functionality overview

## Texturing functions

The main purpose of the `base` module is to provide material creators with a comprehensive set of
texturing functions to spatial variations to their materials.

The texture functions enable you to add bitmapped textures, various types of procedural patterns,
as well as layered combinations of these textures to your material. The coordinate space for these
texture nodes defaults to the receiving object's first avg coordinate space. Function names for
texturing functions end with `_texture`.

All basic texturing functions return a value of type `struct` containing two fields:

| Field | Purpose |
| ----- | ------- |
| `tint` | Used for texturing parameters of type `color` |
| `mono` | Used for texturing parameters of type `float` |

For bitmap textures, the @c-mono value can be used to access the alpha channel of bitmaps.

If a different coordinate space is used or the coordinate space is transformed then you will need
to use the ancillary functions `base::coordinate_source`, `base::coordinate_projection` and
`base::coordinate_transformation`. Internally, all coordinates for texturing are treated as 3D.

To enable bump mapping, texturing functions with names ending with `_bump_texture` can be attached
to the `bump` input parameter of distribution functions and the material geometry interface. A
dedicated function `base::tangent_space_normal_texture` can be used to load tangent space normal
map textures for normal mapping.

For additional functionality, the output of texturing functions can be modified by using
`base::gradient3_recolor` or combined with other textures and values through
`base::blend_color_layers`, based on a number of blend operations.

## Ancillary functions

A number of ancillary functions complete the `base` module.

Functions `base::coordinate_source` and `base::coordinate_projection` allow advanced handling of 3D
coordinates from various sources as well as procedural generation of coordinate systems through
projection techniques. The enum `base::projection_mode` is used to specify the technique to be
used:

| Field name in `base::projection_mode` | Description |
| ------------------------------------- | ----------- |
| `base::projection_cubic` | The projection is formed by six planar projections. The geometry normal is modified to point away from the origin of the projection and then the major direction of the resulting normal is used to decide for each face which projection is to be used.
| `base::projection_tri_planar` | Similar to projection_cubic, but in addition blends the texture smoothly on the edges. |
| `base::projection_spherical` | Spherical projection around the z axis. The created texture space is scaled with the distance from the origin of the projection so that mapping happens according to the circumference of an object. For example, if the projection is applied to a sphere of radius 2, textures are repeated 4*PI times around the sphere. |
| `base::projection_spherical_normalized` | Spherical projection around the z axis. u on the sphere is between -1 and 1, v is in the range -.5 to 0.5. |
| `base::projection_cylindrical` | Projection targeted at objects close in shape to a capped cylinder aligned with the z axis. The geometry normal is modified to point away from the origin of the projection and the major direction of the resulting normal is z or -z, fitting planar projections are used. For all other directions, a cylindrical projection is used with z mapping to v. The mapping range of u is tied to the distance from the z axis of the projection. |
| `base::projection_cylindrical_normalized` | Like `base::projection_cylindrical`, but u is normalized to always wrap twice onto the circumference of the cylinder (u is in the range from -1 to 1). |
| `base::projection_infinite_cylindrical` | Cylindrical projection around the z axis. z is mapped to v and the mapping range of u is tied to the distance from the z axis of the projection. |
| `base::projection_infinite_cylindrical_normalised` | Cylindrical projection around the z axis. z is mapped to v and u is normalized to always wrap twice onto the circumference of the cylinder (u is in the range from -1 to 1). |
| `base::projection_planar` | Planar projection along the z axis. The plane of projection is xy. |

Function `base::transform_coordinate` allows 3D transformations of those coordinates and
`base::rotation_translation_scale` provides one way to generate the necessary transformation
matrix.

MDL allows the simulation of dispersion effects. The functions `base::abbe_number_ior` and
`base::sellmeier_coefficients_ior` allow the specification of the necessarily varying index of
refraction used by the material.

The new explicit tangential alignment of roughness values for glossy BSDF allows full control over
anisotropic effects and the functions `base::anisotropy_conversion` in conjunction with
`base::gloss_to_rough` and `base::architectural_gloss_to_rough` enable designers use the semantic
for anisotropy they prefer while still retaining the alignment information.
