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


# Language elements

## Enums

[`projection_mode`](#projection_mode)

[`color_layer_mode`](#color_layer_mode)

[`texture_coordinate_system`](#texture_coordinate_system)

[`mono_mode`](#mono_mode)

[`gradient_interpolation_mode`](#gradient_interpolation_mode)

[`gradient_mode`](#gradient_mode)

## Structs

[`texture_coordinate_info`](#texture_coordinate_info)

[`color_layer`](#color_layer)

[`texture_return`](#texture_return)

[`anisotropy_return`](#anisotropy_return)

## Functions

[`coordinate_projection`](#coordinate_projection)

[`coordinate_source`](#coordinate_source)

[`transform_coordinate`](#transform_coordinate)

[`rotation_translation_scale`](#rotation_translation_scale)

[`file_texture`](#file_texture)

[`architectural_gloss_to_rough`](#architectural_gloss_to_rough)

[`gloss_to_rough`](#gloss_to_rough)

[`abbe_number_ior`](#abbe_number_ior)

[`sellmeier_coefficients_ior`](#sellmeier_coefficients_ior)

[`environment_spherical`](#environment_spherical)

[`sun_and_sky`](#sun_and_sky)

[`perez_sun_and_sky`](#perez_sun_and_sky)

[`file_bump_texture`](#file_bump_texture)

[`tangent_space_normal_texture`](#tangent_space_normal_texture)

[`blend_color_layers`](#blend_color_layers)

[`gradient3_texture`](#gradient3_texture)

[`gradient3_bump_texture`](#gradient3_bump_texture)

[`gradient3_recolor`](#gradient3_recolor)

[`checker_texture`](#checker_texture)

[`checker_bump_texture`](#checker_bump_texture)

[`perlin_noise_texture`](#perlin_noise_texture)

[`perlin_noise_bump_texture`](#perlin_noise_bump_texture)

[`worley_noise_texture`](#worley_noise_texture)

[`worley_noise_bump_texture`](#worley_noise_bump_texture)

[`flow_noise_texture`](#flow_noise_texture)

[`flow_noise_bump_texture`](#flow_noise_bump_texture)

[`flake_noise_texture`](#flake_noise_texture)

[`flake_noise_bump_texture`](#flake_noise_bump_texture)

[`tile_texture`](#tile_texture)

[`tile_bump_texture`](#tile_bump_texture)

[`anisotropy_conversion`](#anisotropy_conversion)

[`blend_normals`](#blend_normals)

[`lookup_volume_coefficients`](#lookup_volume_coefficients)

[`blackbody_emission`](#blackbody_emission)


# Enums

## projection_mode

Methods for texture projection

| Name | Description |
| ---- | ----------- |
| `projection_cubic` | Projected space has a cube-shaped appearance |
| `projection_spherical` | Projected space forms a sphere around the projector |
| `projection_cylindrical` | Projected space forms a capped cylinder |
| `projection_infinite_cylindrical` | Projected space forms an infinite cylinder |
| `projection_planar` | Planar projection along the z axis of the projectors space |
| `projection_spherical_normalized` | Like `projection_spherical`, but u is normalized between -1 and 1 and v between -0.5 and 0.5 |
| `projection_cylindrical_normalized` | Like `projection_cylindrical`, but u is normalized between -1 and 1 (if not on the cap) |
| `projection_infinite_cylindrical_normalized` | Like `projection_cylindrical_infinite`, but u is normalized between -1 and 1 |
| `projection_tri_planar` | Like `projection_cubic`, but blends the texture smoothly on the edges |

## color_layer_mode

Texture combination modes between two layers. the two layers are modified in the manner described by the modes, and the result is blended with the bottom layer based on a weighting factor.

| Name | Description |
| ---- | ----------- |
| `color_layer_blend` | Top |
| `color_layer_add` | Top + bottom |
| `color_layer_multiply` | Top * bottom |
| `color_layer_screen` | 1 - ((1 - top) * (1 - bottom)) |
| `color_layer_overlay` | For each channel individually: if bottom <0.5: top * bottom * 2, else: 2 * (top+bottom-top * bottom-0.5) |
| `color_layer_brightness` | Hue of the bottom layer combined with the intensity of the top |
| `color_layer_color` | Intensity of the bottom layer combined with the hue of the top |
| `color_layer_exclusion` | Bottom + top - bottom * top * 2 |
| `color_layer_average` | Average of top and bottom layer |
| `color_layer_lighten` | Maximum of top and bottom layer |
| `color_layer_darken` | Minimum of top and bottom layer |
| `color_layer_sub` | Bottom + top - 1 |
| `color_layer_negation` | 1 - math::abs(1 - (bottom + top)) |
| `color_layer_difference` | Absolute difference of top and bottom layer |
| `color_layer_softlight` | (top < 0.5) ? 2 * (top * bottom + bottom * bottom * (0.5 - top)) : 2 * (math::sqrt(bottom) * (top - 0.5) + bottom - top * bottom) |
| `color_layer_colordodge` | Bottom / (1 - top) |
| `color_layer_reflect` | Bottom * bottom/(1 - top) |
| `color_layer_colorburn` | 1 - (1 - bottom)/top |
| `color_layer_phoenix` | Minimum of both layers minus the maximum of both layers (plus 1.0) |
| `color_layer_hardlight` | For each channel individually: if top >0.5: top * bottom * 2, else: 2 * (top+bottom-top * bottom-0.5) |
| `color_layer_pinlight` |   |
| `color_layer_hardmix` | For each channel individually: (top+bottom <= 1) ? 0 : 1 |
| `color_layer_lineardodge` | Top + bottom (clamped) |
| `color_layer_linearburn` | Bottom + top - 1 (clamped) |
| `color_layer_spotlight` | 2 * top * bottom |
| `color_layer_spotlightblend` | Top * bottom + bottom |
| `color_layer_hue` | Uses hue from top layer, saturation and brightness from bottom |
| `color_layer_saturation` | Uses saturation from top layer, hue and brightness from bottom |

## texture_coordinate_system

Coordinate system selection for textures

| Name | Description |
| ---- | ----------- |
| `texture_coordinate_uvw` | Texture space of surface |
| `texture_coordinate_world` | World coordinate space |
| `texture_coordinate_object` | Object coordinate space |

## mono_mode

Modes for the creation of a gray-scale value from a color

| Name | Description |
| ---- | ----------- |
| `mono_alpha` | Alpha channel of the texture is used |
| `mono_average` | Average intensity of rgb is used |
| `mono_luminance` | Value is calculated using math::math::luminance() |
| `mono_maximum` | Maximum intensity of the texture is used |

## gradient_interpolation_mode

Modes for interpolating between the different colors in a gradient texture

| Name | Description |
| ---- | ----------- |
| `gradient_interpolation_linear` | Linear interpolation |
| `gradient_interpolation_off` | No interpolation ("solid") |
| `gradient_interpolation_ease_in` | Ease-in |
| `gradient_interpolation_ease_out` | Ease-out |
| `gradient_interpolation_ease_in_out` | Ease-in-out |

## gradient_mode

Modes for generating the gradient position based on input uv coordinates

| Name | Description |
| ---- | ----------- |
| `gradient_linear` | Linear |
| `gradient_squared` | Squared |
| `gradient_box` | Box |
| `gradient_diagonal` | Diagonal |
| `gradient_90_degree` | Asymmetric 90 degree |
| `gradient_symmetric_90_degree` | Symmetric 90 degree |
| `gradient_radial` | Radial |
| `gradient_360_degree` | 360 degree |


# Structs

## texture_coordinate_info

The texture coordinate, `tangent_u` and `tangent_v` needed by bump mapping and anisotropy

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `position` | `float3` | `state::texture_coordinate(0)` | Texture coordinate |
| `tangent_u` | `float3` | `state::texture_tangent_u(0)` | Tangent in *u* direction |
| `tangent_v` | `float3` | `state::texture_tangent_v(0)` | Tangent in *v* direction |
| `source_flags` | `int` | `int(texture_coordinate_uvw)` | Internal info to track source space type (first two bits). |

## color_layer

Single texture layer for use in blending

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `layer_color` | `color` | `color(0)` | The color to be combined with a layer "below" this layer |
| `weight` | `float` | `1.0` | Scale factor for blending this color with the color produced by the mode value (the "lower layer") |
| `mode` | `uniform color_layer_mode` | `color_layer_blend` | Method for combining this layer and the lower layer |

## texture_return

Type of the return value from texturing functions

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `tint` | `color` | `color(0)` | Return value suitable to for driving input parameters of type color |
| `mono` | `float` | `0.0` | Gray-scale return value suitable for driving input parameters of type float |

## anisotropy_return

Type of the return value from functions driving roughness and anisotropy parameters of glossy bsdf

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `roughness_u` | `float` | `0.0` | Roughness in *u* direction |
| `roughness_v` | `float` | `0.0` | Roughness in *v* direction |
| `tangent_u` | `float3` | `state::texture_tangent_u(0)` | Tangent in *u* direction |


# Functions

## coordinate_projection

Returns [`texture_coordinate_info`](#texture_coordinate_info)

Constructs a texturing coordinate system based on a variety of projection functions.

| Name | Type | Default | Range  | Varying? | Description |
| ---- | ---- | ------- | -----  | ---------| ----------- |
| `coordinate_system` | [`texture_coordinate_system`](#texture_coordinate_system) | `texture_coordinate_object` |  | no | The projection can be done based on world, object or any `uvw` space. |
| `texture_space` | `int` | `0` | `0..255`  | no | If `texture_coordinate_uvw`, index into the appropriate one |
| `projection_type` | [`projection_mode`](#projection_mode) | `projection_planar` |  | no | Projection method to be used to generate the coordinates |
| `projection_transform` | `float4x4` | `float4x4(1.0)` |  | no | Transformation of the projector in world space |

## coordinate_source

Returns [`texture_coordinate_info`](#texture_coordinate_info)

Access to world coordinates, object coordinates or specifically defined texture spaces

| Name | Type | Default | Range  | Varying? | Description |
| ---- | ---- | ------- | -----  | ---------| ----------- |
| `coordinate_system` | [`texture_coordinate_system`](#texture_coordinate_system) | `texture_coordinate_uvw` |  | no | The function can source coordinates in `uvw`, world or object space |
| `texture_space` | `int` | `0` | `0..255`  | no | If `texture_coordinate_uvw`, index into the appropriate one |

## transform_coordinate

Returns [`texture_coordinate_info`](#texture_coordinate_info)

Transform a texture coordinate by a matrix

| Name | Type | Default  | Varying? | Description |
| ---- | ---- | -------  | ---------| ----------- |
| `transform` | `float4x4` | (none)  | yes | A transformation to be applied to the source coordinates. `rotation_translation_scale()` is a suggested means to compute the transformation matrix |
| `coordinate` | [`texture_coordinate_info`](#texture_coordinate_info) | [`texture_coordinate_info()`](#texture_coordinate_info)  | yes | Coordinate, typically sourced from `coordinate_source` or `coordinate_projection` |

## rotation_translation_scale

Returns `float4x4`

Construct transformation matrix from euler rotation, translation and scale

| Name | Type | Default  | Varying? | Description |
| ---- | ---- | -------  | ---------| ----------- |
| `rotation` | `float3` | `float3(0.0)`  | yes | Rotation applied to every uvw coordinate |
| `translation` | `float3` | `float3(0.0)`  | yes | Offset applied to every uvw coordinate |
| `scaling` | `float3` | `float3(1.0)`  | yes | Scale applied to every uvw coordinate |

## file_texture

Returns [`texture_return`](#texture_return)

General texturing function for 2d bitmap texture stored in a file

| Name | Type | Default  | Varying? | Description |
| ---- | ---- | -------  | ---------| ----------- |
| `texture` | `texture_2d` | (none)  | no | The input texture |
| `color_offset` | `color` | `color(0.0)`  | yes | Fixed offset value added to all texture values |
| `color_scale` | `color` | `color(1.0)`  | yes | Fixed scaling factor applied to all texture values |
| `mono_source` | [`mono_mode`](#mono_mode) | `mono_alpha`  | no | Defines how `mono_result` is computed |
| `uvw` | [`texture_coordinate_info`](#texture_coordinate_info) | [`texture_coordinate_info()`](#texture_coordinate_info)  | yes | Custom value for texture coordinate |
| `crop_u` | `float2` | `float2(0.0, 1.0)`  | no | Restricts the texture access to sub-domain of the texture in the *u* direction |
| `crop_v` | `float2` | `float2(0.0, 1.0)`  | no | Restricts the texture access to sub-domain of the texture in the *v* direction |
| `wrap_u` | `tex::wrap_mode` | `tex::wrap_repeat`  | no | Wrapping mode in the *u* direction |
| `wrap_v` | `tex::wrap_mode` | `tex::wrap_repeat`  | no | Wrapping mode in the *v* direction |
| `clip` | `bool` | `false`  | no | Deprecated, use `wrap_mode`=tex::`wrap_clip`. defines `wrap_clamp` behavior. if true, a lookup outside [0,1] results in black/transparent |
| `animation_start_time` | `float` | `0.0`  | no | When to start playing first frame of the animation  |
| `animation_crop` | `int2` | `int2(0, 0)`  | no | If the texture is an animation, the range of frames to be played can be specified. |
| `animation_wrap` | `tex::wrap_mode` | `tex::wrap_repeat`  | no | Defines what to do outside of regular animation time |
| `animation_fps` | `float` | `30`  | no | Framerate to use for animation playback |

## architectural_gloss_to_rough

Returns `float`

Convert glossiness parameter to roughness parameter, semantics according to the iray v2 arch+design implementation

| Name | Type | Default | Range  | Varying? | Description |
| ---- | ---- | ------- | -----  | ---------| ----------- |
| `glossiness` | `float` | (none) | `0.0..1.0`  | yes | Glossiness according to mia material semantic |

## gloss_to_rough

Returns `float`

Convert glossiness parameter to roughness parameter through simple inversion

| Name | Type | Default | Range  | Varying? | Description |
| ---- | ---- | ------- | -----  | ---------| ----------- |
| `glossiness` | `float` | (none) | `0.0..1.0`  | yes | Inverse of roughness |

## abbe_number_ior

Returns `color`

Calculate spectral index of refraction

| Name | Type | Default  | Varying? | Description |
| ---- | ---- | -------  | ---------| ----------- |
| `ior` | `float` | `1.5`  | no | Index of refraction |
| `abbe_number` | `float` | `0.0`  | no | Dispersion in relation to index of refraction |

## sellmeier_coefficients_ior

Returns `color`

Calculate spectral index of refraction using sellmeier coefficients

| Name | Type | Default  | Varying? | Description |
| ---- | ---- | -------  | ---------| ----------- |
| `sellmeier_B` | `float3` | `float3(1.04, 0.23, 1.01)`  | no | Sellmeier coefficient b |
| `sellmeier_C` | `float3` | `float3(0.006, 0.200, 103.56)`  | no | Sellmeier coefficient c (in um^2) |

## environment_spherical

Returns [`texture_return`](#texture_return)

Environment function implementing a spherical environment

| Name | Type | Default  | Varying? | Description |
| ---- | ---- | -------  | ---------| ----------- |
| `texture` | `texture_2d` | (none)  | no | Spherical environment |

## sun_and_sky

Returns [`texture_return`](#texture_return)

Sun and sky model environment. (for documentation of the parameters, please see the official documentation of the matching mental ray shader.)

| Name | Type | Default  | Varying? | Description |
| ---- | ---- | -------  | ---------| ----------- |
| `on` | `bool` | `true`  | no |  |
| `multiplier` | `float` | `0.025`  | yes |  |
| `rgb_unit_conversion` | `color` | `color(0.000666667)`  | no |  |
| `haze` | `float` | `0.5`  | yes |  |
| `redblueshift` | `float` | `0.0`  | yes |  |
| `saturation` | `float` | `0.5`  | no |  |
| `horizon_height` | `float` | `0.001`  | yes |  |
| `horizon_blur` | `float` | `0.1`  | yes |  |
| `ground_color` | `color` | `color(0.4, 0.4, 0.4)`  | yes |  |
| `night_color` | `color` | `color(0.0, 0.0, 0.0)`  | yes |  |
| `sun_direction` | `float3` | `float3(0.0, 0.229271, 0.418882)`  | no |  |
| `sun_disk_intensity` | `float` | `0.01`  | no |  |
| `sun_disk_scale` | `float` | `0.5`  | no |  |
| `sun_glow_intensity` | `float` | `1.0`  | no |  |
| `y_is_up` | `bool` | `true`  | no |  |
| `flags` | `int` | `0`  | no |  |
| `physically_scaled_sun` | `bool` | `true`  | no |  |

## perez_sun_and_sky

Returns `color`

Perez all weather sun and sky model.

| Name | Type | Default  | Varying? | Description |
| ---- | ---- | -------  | ---------| ----------- |
| `direct_normal_irradiance` | `float` | `500.0`  | no | Direct sun irradiance in w / m^2 |
| `diffuse_horizontal_irradiance` | `float` | `50.0`  | no | Diffuse horizontal sky irradiance in w / m^2 (excluding sun) |
| `sun_direction` | `float3` | `float3(0.0, 0.229271, 0.418882)`  | no | Sun direction |
| `ground_color` | `color` | `color(0.2, 0.2, 0.2)`  | yes | Ground reflectivity |
| `horizon_blur` | `float` | `0.1`  | no | Blur between ground and sky at the horizon |
| `y_is_up` | `bool` | `true`  | no | Sun direction is y-up coordinates |
| `julian_date` | `float` | `180.0`  | no | Day in year |
| `dew_point` | `float` | `11.0`  | no | Dew point in degree celsius |
| `haze` | `float` | `0.5`  | no | Haze value to compute chromaticity of the sky (in analogy to base::`sun_and_sky)` |

## file_bump_texture

Returns `float3`

Computes a normal based on a heightfield-style bump texture

| Name | Type | Default  | Varying? | Description |
| ---- | ---- | -------  | ---------| ----------- |
| `texture` | `texture_2d` | (none)  | no | The input texture |
| `factor` | `float` | `1.0f`  | no | Determines the degree of bumpiness |
| `bump_source` | [`mono_mode`](#mono_mode) | `mono_average`  | no | Defines what value to use for computing the slope of the bump |
| `uvw` | [`texture_coordinate_info`](#texture_coordinate_info) | [`texture_coordinate_info()`](#texture_coordinate_info)  | yes | Parameterization to be used for texture mapping. defaults to texture channel `0`. |
| `crop_u` | `float2` | `float2(0.0, 1.0)`  | no | Restricts the texture access to sub-domain of the texture in the *u* direction |
| `crop_v` | `float2` | `float2(0.0, 1.0)`  | no | Restricts the texture access to sub-domain of the texture in the *v* direction |
| `wrap_u` | `tex::wrap_mode` | `tex::wrap_repeat`  | no | Wrapping mode in the *u* direction |
| `wrap_v` | `tex::wrap_mode` | `tex::wrap_repeat`  | no | Wrapping mode in the *v* direction |
| `normal` | `float3` | `state::normal()`  | yes | Base normal for the bump mapping. |
| `clip` | `bool` | `false`  | no | Deprecated, `usewrap_mode`=tex::`wrap_clip`. defines `wrap_clamp` behavior. if true, lookup outside [0,1] results in no bump |
| `b_spline` | `bool` | `false`  | no | Use b-spline interpolation for smooth result |
| `animation_start_time` | `float` | `0.0`  | no | When to start playing first frame of the animation  |
| `animation_crop` | `int2` | `int2(0, 0)`  | no | If the texture is an animation, the range of frames to be played can be specified. |
| `animation_wrap` | `tex::wrap_mode` | `tex::wrap_repeat`  | no | Defines what to do outside of regular animation time |
| `animation_fps` | `float` | `30`  | no | Framerate to use for animation playback |

## tangent_space_normal_texture

Returns `float3`

Interprets the color values of a bitmap as a vector in tangent space

| Name | Type | Default  | Varying? | Description |
| ---- | ---- | -------  | ---------| ----------- |
| `texture` | `texture_2d` | (none)  | no | The input texture |
| `factor` | `float` | `1.0f`  | no | Determines the degree of bumpiness |
| `flip_tangent_u` | `bool` | `false`  | no | Can be used to fix mismatches between the object's tangent space and the normal map's tangent space |
| `flip_tangent_v` | `bool` | `false`  | no | Can be used to fix mismatches between the object's tangent space and the normal map's tangent space |
| `uvw` | [`texture_coordinate_info`](#texture_coordinate_info) | [`texture_coordinate_info()`](#texture_coordinate_info)  | yes | Parameterization to be used for texture mapping. defaults to texture channel `0`. |
| `crop_u` | `float2` | `float2(0.0, 1.0)`  | no | Restricts the texture access to sub-domain of the texture in the *u* direction |
| `crop_v` | `float2` | `float2(0.0, 1.0)`  | no | Restricts the texture access to sub-domain of the texture in the *v* direction |
| `wrap_u` | `tex::wrap_mode` | `tex::wrap_repeat`  | no | Wrapping mode in the *u* direction |
| `wrap_v` | `tex::wrap_mode` | `tex::wrap_repeat`  | no | Wrapping mode in the *v* direction |
| `clip` | `bool` | `false`  | no | Deprecated, `usewrap_mode`=tex::`wrap_clip`. defines `wrap_clamp` behavior. if true, lookup outside [0,1] results in no bump |
| `scale` | `float` | `1.0`  | no | Scales the value red from the texture file. can be used to adapt to different normal map formats |
| `offset` | `float` | `0.0`  | no | Offset applied to the value red from the texture file. can be used to adapt to different normal map formats |
| `animation_start_time` | `float` | `0.0`  | no | When to start playing first frame of the animation  |
| `animation_crop` | `int2` | `int2(0, 0)`  | no | If the texture is an animation, the range of frames to be played can be specified. |
| `animation_wrap` | `tex::wrap_mode` | `tex::wrap_repeat`  | no | Defines what to do outside of regular animation time |
| `animation_fps` | `float` | `30`  | no | Framerate to use for animation playback |

## blend_color_layers

Returns [`texture_return`](#texture_return)

Texture layering functionality similar to the functionality known from painting programs 

| Name | Type | Default  | Varying? | Description |
| ---- | ---- | -------  | ---------| ----------- |
| `layers` | `color_layer[<P>]` | (none)  | yes | Array of structs describing the layers to blend |
| `base` | `color` | `color(0.0)`  | yes | Color of initial layer |
| `mono_source` | [`mono_mode`](#mono_mode) | `mono_average`  | no | Defines how `mono_result` is computed |

## gradient3_texture

Returns [`texture_return`](#texture_return)

Gradient calculated from three colors at three positions

| Name | Type | Default  | Varying? | Description |
| ---- | ---- | -------  | ---------| ----------- |
| `mode` | [`gradient_mode`](#gradient_mode) | `gradient_linear`  | no | Mode of gradient calculation, describes shape of the gradient |
| `gradient_positions` | `float[3]` | `float[](0.0, 0.5, 1.0 )`  | yes | Position of the gradient colors |
| `gradient_colors` | `color[3]` | `color[](` <br> `   color(0.0),` <br> `   color(0.5),` <br> `   color(1.0)` <br> `)`  | yes | Colors at the positions |
| `interpolation_modes` | `gradient_interpolation_mode[3]` | `gradient_interpolation_mode[](` <br> `   gradient_interpolation_linear,` <br> `   gradient_interpolation_linear,` <br> `   gradient_interpolation_linear ` <br> `)`  | yes | Interpolation mode between gradient colors |
| `uvw` | [`texture_coordinate_info`](#texture_coordinate_info) | [`texture_coordinate_info()`](#texture_coordinate_info)  | yes | Parameterization to be used for texture mapping. defaults to texture channel `0`. |
| `distortion` | `float` | `0.0`  | yes | Distortion value to be added to the position inside the gradient |

## gradient3_bump_texture

Returns `float3`

Gradient for bump mapping calculated from three colors at three positions

| Name | Type | Default  | Varying? | Description |
| ---- | ---- | -------  | ---------| ----------- |
| `mode` | [`gradient_mode`](#gradient_mode) | `gradient_linear`  | no | Mode of gradient calculation, describes shape of the gradient |
| `gradient_positions` | `float[3]` | `float[](0.0, 0.5, 1.0 )`  | yes | Position of the gradient colors |
| `gradient_colors` | `color[3]` | `color[](` <br> `   color(0.0),` <br> `   color(0.5),` <br> `   color(1.0) ` <br> `)`  | yes | Colors at the positions |
| `interpolation_modes` | `gradient_interpolation_mode[3]` | `gradient_interpolation_mode[](` <br> `   gradient_interpolation_linear,` <br> `   gradient_interpolation_linear,` <br> `   gradient_interpolation_linear ` <br> `)`  | yes | Interpolation mode between gradient colors |
| `uvw` | [`texture_coordinate_info`](#texture_coordinate_info) | [`texture_coordinate_info()`](#texture_coordinate_info)  | yes | Parameterization to be used for texture mapping. defaults to texture channel `0`. |
| `distortion` | `float` | `0.0`  | yes | Distortion value to be added to the position inside the gradient |
| `scale` | `float` | `1.0`  | yes | Strength of the bump mapping effect |
| `normal` | `float3` | `state::normal()`  | yes | Base normal for the bump mapping. |

## gradient3_recolor

Returns [`texture_return`](#texture_return)

Function mapping an arbitrary float value into a gradient. can be used to re-color textures

| Name | Type | Default  | Varying? | Description |
| ---- | ---- | -------  | ---------| ----------- |
| `gradient_positions` | `float[3]` | `float[](0.0, 0.5, 1.0 )`  | yes | Position of the gradient colors |
| `gradient_colors` | `color[3]` | `color[](` <br> `   color(0.0),` <br> `   color(0.5),` <br> `   color(1.0) ` <br> `)`  | yes | Colors at the positions |
| `interpolation_modes` | `gradient_interpolation_mode[3]` | `gradient_interpolation_mode[](` <br> `   gradient_interpolation_linear,` <br> `   gradient_interpolation_linear,` <br> `   gradient_interpolation_linear` <br> `)`  | yes | Mode of gradient calculations |
| `mono_source` | [`mono_mode`](#mono_mode) | `mono_alpha`  | no | Deprecated, has no effect in gradient3 recolor |
| `distortion` | `float` | `0.0`  | yes | Distortion value to be added to the position inside the gradient |
| `position` | `float` | `0.0`  | yes | Value driving the gradient calculation |

## checker_texture

Returns [`texture_return`](#texture_return)

3d color checker pattern

| Name | Type | Default  | Varying? | Description |
| ---- | ---- | -------  | ---------| ----------- |
| `uvw` | [`texture_coordinate_info`](#texture_coordinate_info) | [`texture_coordinate_info()`](#texture_coordinate_info)  | yes | Parameterization to be used for texture mapping. defaults to texture channel `0`. |
| `color1` | `color` | `color(0)`  | yes | Color of the even tiles of the checker function |
| `color2` | `color` | `color(1)`  | yes | Color of the odd tiles of the checker function |
| `blur` | `float` | `0`  | no | Softens the border between tiles |
| `checker_position` | `float` | `0.5`  | no | Ratio of division. values other than 0.5 produce non square tiles |

## checker_bump_texture

Returns `float3`

3d bump mapping checker pattern

| Name | Type | Default | Range  | Varying? | Description |
| ---- | ---- | ------- | -----  | ---------| ----------- |
| `uvw` | [`texture_coordinate_info`](#texture_coordinate_info) | [`texture_coordinate_info()`](#texture_coordinate_info) |  | yes | Parameterization to be used for texture mapping. defaults to texture channel `0`. |
| `factor` | `float` | `1` |  | no | Strength of the bump mapping effect |
| `blur` | `float` | `0.0` | `0.0..1.0`  | no | Softens the border between tiles |
| `checker_position` | `float` | `0.5` |  | no | Ratio of division. values other than 0.5 produce non square tiles |
| `normal` | `float3` | `state::normal()` |  | yes | Base normal for the bump mapping. |

## perlin_noise_texture

Returns [`texture_return`](#texture_return)

Color perlin noise

| Name | Type | Default | Range  | Varying? | Description |
| ---- | ---- | ------- | -----  | ---------| ----------- |
| `uvw` | [`texture_coordinate_info`](#texture_coordinate_info) | [`texture_coordinate_info()`](#texture_coordinate_info) |  | yes | Parameterization to be used for texture mapping. defaults to texture channel `0`. |
| `color1` | `color` | `color(0)` |  | yes | The perlin noise function will blend between this color and `color2` |
| `color2` | `color` | `color(1)` |  | yes | The perlin noise function will blend between `color1` and this color |
| `size` | `float` | `1.0` |  | no | Size of the biggest feature of the pattern |
| `apply_marble` | `bool` | `false` |  | no | Triggers a modification to make the pattern have a marble like appearance (cosine) |
| `apply_dent` | `bool` | `false` |  | no | Raises the output of the function to the power of 3 |
| `noise_phase` | `float` | `0.0` |  | no | Controls the 4th dimension of the function (can be time in animations) |
| `noise_levels` | `int` | `1` |  | no | Number of octaves to of perlin to sum up |
| `absolute_noise` | `bool` | `false` |  | no | If set to true, the appearance of the pattern will be more "billowing" and "turbulent" |
| `ridged_noise` | `bool` | `false` |  | no | If set to true, the appearance of the pattern will be more "electrical" |
| `noise_distortion` | `float3` | `float3(0.0 )` |  | no | Weight of additional noise turbulence |
| `noise_threshold_high` | `float` | `1.0` | `0.0..1.0`  | no | Noise values greater then "`noise_threshold_high`" are mapped to `color1` |
| `noise_threshold_low` | `float` | `0.0` | `0.0..1.0`  | no | Noise values greater then "`noise_threshold_low`" are mapped to `color2` |
| `noise_bands` | `float` | `1.0` |  | no | Creates a "tree ring" like banding effect |

## perlin_noise_bump_texture

Returns `float3`

Bump-mapping perlin noise

| Name | Type | Default | Range  | Varying? | Description |
| ---- | ---- | ------- | -----  | ---------| ----------- |
| `uvw` | [`texture_coordinate_info`](#texture_coordinate_info) | [`texture_coordinate_info()`](#texture_coordinate_info) |  | yes | Parameterization to be used for texture mapping. defaults to texture channel `0`. |
| `factor` | `float` | `1.0` |  | no | Strength of the bump mapping effect |
| `size` | `float` | `1.0` |  | no | Size of the biggest feature of the pattern |
| `apply_marble` | `bool` | `false` |  | no | Triggers a modification to make the pattern have a marble like appearance (cosine) |
| `apply_dent` | `bool` | `false` |  | no | Raises the output of the function to the power of 3 |
| `noise_phase` | `float` | `0.0` |  | no | Controls the 4th dimension of the function |
| `noise_levels` | `int` | `1` |  | no | Number of octaves to of perlin to sum up |
| `absolute_noise` | `bool` | `false` |  | no | If set to true, the appearance of the pattern will be more "billowing" and "turbulent" |
| `ridged_noise` | `bool` | `false` |  | no | If set to true, the appearance of the pattern will be more "electrical" |
| `noise_distortion` | `float3` | `float3(0.0 )` |  | no | Weight of additional noise turbulence |
| `noise_threshold_high` | `float` | `1.0` | `0.0..1.0`  | no | Noise values greater then "`noise_threshold_high`" are mapped to the maximum bump height |
| `noise_threshold_low` | `float` | `0.0` | `0.0..1.0`  | no | Noise values greater then "`noise_threshold_low`" are mapped to the minimum bump height |
| `noise_bands` | `float` | `1.0` |  | no | Creates a "tree ring" like banding effect |
| `normal` | `float3` | `state::normal()` |  | yes | Base normal for the bump mapping. |

## worley_noise_texture

Returns [`texture_return`](#texture_return)

Color worley noise

| Name | Type | Default | Range  | Varying? | Description |
| ---- | ---- | ------- | -----  | ---------| ----------- |
| `uvw` | [`texture_coordinate_info`](#texture_coordinate_info) | [`texture_coordinate_info()`](#texture_coordinate_info) |  | yes | Parameterization to be used for texture mapping. defaults to texture channel `0`. |
| `color1` | `color` | `color(0)` |  | yes | The worley noise function will blend between this color and `color2` |
| `color2` | `color` | `color(1)` |  | yes | The worley noise function will blend between `color1` and this color |
| `size` | `float` | `1.0` |  | no | Size of the biggest feature of the pattern |
| `mode` | `int` | `0` | `0..11`  | no | Output function used to modify the noise result (0.0.11) |
| `metric` | `int` | `0` | `0..2`  | no | Metric used in the noise function (0: euclidean, 1: manhattan, 2: chebyshev) |
| `apply_marble` | `bool` | `false` |  | no | Triggers a modification to make the pattern have a marble like appearance (cosine) |
| `apply_dent` | `bool` | `false` |  | no | Raises the output of the function to the power of 3 |
| `noise_distortion` | `float3` | `float3(0.0 )` |  | no | Weight of additional noise turbulence |
| `noise_threshold_high` | `float` | `1.0` | `0.0..1.0`  | no | Noise values greater then "`noise_threshold_high`" are mapped to `color1` |
| `noise_threshold_low` | `float` | `0.0` | `0.0..1.0`  | no | Noise values greater then "`noise_threshold_low`" are mapped to `color2` |
| `noise_bands` | `float` | `1.0` |  | no | Creates a "tree ring" like banding effect |
| `step_threshold` | `float` | `0.2` |  | no | Used only in mode 3 |
| `edge` | `float` | `1.0` | `0.0..1.0`  | no | Smoothness of noise |

## worley_noise_bump_texture

Returns `float3`

Bump-mapping worley noise

| Name | Type | Default | Range  | Varying? | Description |
| ---- | ---- | ------- | -----  | ---------| ----------- |
| `uvw` | [`texture_coordinate_info`](#texture_coordinate_info) | [`texture_coordinate_info()`](#texture_coordinate_info) |  | yes | Parameterization to be used for texture mapping. defaults to texture channel `0`. |
| `factor` | `float` | `1.0` |  | no | Strength of the bump mapping effect |
| `size` | `float` | `1.0` |  | no | Size of the biggest feature of the pattern |
| `mode` | `int` | `0` | `0..11`  | no | Output function used to modify the noise result (0.0.11) |
| `metric` | `int` | `0` | `0..2`  | no | Metric used in the noise function (0: euclidean, 1: manhattan, 2: chebyshev) |
| `apply_marble` | `bool` | `false` |  | no | Triggers a modification to make the pattern have a marble like appearance (cosine) |
| `apply_dent` | `bool` | `false` |  | no | Raises the output of the function to the power of 3 |
| `noise_distortion` | `float3` | `float3(0.0 )` |  | no | Weight of additional noise turbulence |
| `noise_threshold_high` | `float` | `1.0` | `0.0..1.0`  | no | Noise values greater then "`noise_threshold_high`" are mapped to the maximum bump height |
| `noise_threshold_low` | `float` | `0.0` | `0.0..1.0`  | no | Noise values greater then "`noise_threshold_low`" are mapped to the minimum bump height |
| `noise_bands` | `float` | `1.0` |  | no | Creates a "tree ring" like banding effect |
| `step_threshold` | `float` | `0.2` |  | no | Used only in mode 3 |
| `edge` | `float` | `1.0` | `0.0..1.0`  | no | Smoothness of noise |
| `normal` | `float3` | `state::normal()` |  | yes | Base normal for the bump mapping. |

## flow_noise_texture

Returns [`texture_return`](#texture_return)

Color perlin flow noise

| Name | Type | Default  | Varying? | Description |
| ---- | ---- | -------  | ---------| ----------- |
| `uvw` | [`texture_coordinate_info`](#texture_coordinate_info) | [`texture_coordinate_info()`](#texture_coordinate_info)  | yes | Parameterization to be used for texture mapping. defaults to texture channel `0`. |
| `color1` | `color` | `color(0)`  | yes | The flow noise function will blend between this color and `color2` |
| `color2` | `color` | `color(1)`  | yes | The flow noise function will blend between `color1` and this color |
| `size` | `float` | `1.0`  | no | Size of the biggest feature of the pattern |
| `phase` | `float` | `0.0`  | no | Controls the 3rd dimension of the function |
| `levels` | `int` | `1`  | no | Number of octaves to of flow noise to sum up |
| `absolute_noise` | `bool` | `false`  | no | If set to true, the appearance of the pattern will be more "billowing" and "turbulent" |
| `level_gain` | `float` | `0.5`  | no | If multiple levels are used, "`level_gain`" specifies a weighting factor for subsequent levels |
| `level_scale` | `float` | `2.0`  | no | If multiple levels are used, "`level_scale`" specifies a global scaling factor for subsequent levels |
| `level_progressive_u_scale` | `float` | `1.0`  | no | If multiple levels are used, "`level_progressive_u_scale`" specifies an additional scaling factor in the *u* direction |
| `level_progressive_v_motion` | `float` | `0.0`  | no | If multiple levels are used, "`level_progressive_v_motion`" specifies an offset for subsequent levels in the *v* direction |

## flow_noise_bump_texture

Returns `float3`

Bump-mapping flow noise

| Name | Type | Default  | Varying? | Description |
| ---- | ---- | -------  | ---------| ----------- |
| `uvw` | [`texture_coordinate_info`](#texture_coordinate_info) | [`texture_coordinate_info()`](#texture_coordinate_info)  | yes | Parameterization to be used for texture mapping. defaults to texture channel `0`. |
| `factor` | `float` | `1.0`  | no | Strength of the bump mapping effect |
| `size` | `float` | `1.0`  | no | Size of the biggest feature of the pattern |
| `phase` | `float` | `0.0`  | no | Controls the 4th dimension of the function |
| `levels` | `int` | `1`  | no | Number of octaves to of perlin to sum up |
| `absolute_noise` | `bool` | `false`  | no | If set to true, the appearance of the pattern will be more "billowing" and "turbulent" |
| `level_gain` | `float` | `0.5`  | no | If multiple levels are used, "`level_gain`" specifies a weighting factor for subsequent levels |
| `level_scale` | `float` | `2.0`  | no | If multiple levels are used, "`level_scale`" specifies a global scaling factor for subsequent levels |
| `level_progressive_u_scale` | `float` | `1.0`  | no | If multiple levels are used, "`level_progressive_u_scale`" specifies an additional scaling factor in the *u* direction |
| `level_progressive_v_motion` | `float` | `0.0`  | no | If multiple levels are used, "`level_progressive_v_motion`" specifies an offset for subsequent levels in the *v* direction |
| `normal` | `float3` | `state::normal()`  | yes | Base normal for the bump mapping. |

## flake_noise_texture

Returns [`texture_return`](#texture_return)

Flake noise

| Name | Type | Default | Range  | Varying? | Description |
| ---- | ---- | ------- | -----  | ---------| ----------- |
| `uvw` | [`texture_coordinate_info`](#texture_coordinate_info) | [`texture_coordinate_info()`](#texture_coordinate_info) |  | yes | Parameterization to be used for texture mapping. defaults to texture channel `0`. |
| `intensity` | `float` | `1.0` | `0.0..1.0`  | yes | Specifies the maximum reflectivity of any flake |
| `scale` | `float` | `1.0` |  | yes | Size of the features of the pattern |
| `density` | `float` | `1.0` |  | yes | Controls the amount of flakes in the substrate. the higher the number, the bigger is the space between flakes |
| `noise_type` | `int` | `0` | `0..1`  | no | Selects the noise type (0: classic, 1: worley) |
| `maximum_size` | `float` | `1.0` |  | yes | Controls the shape of flakes in the secondary noise mode (worley) |
| `metric` | `int` | `0` | `0..2`  | no | Metric used in the secondary noise mode (worley: 0: euclidean, 1: manhattan, 2: chebyshev) |

## flake_noise_bump_texture

Returns `float3`

Bump-mapping flake noise

| Name | Type | Default | Range  | Varying? | Description |
| ---- | ---- | ------- | -----  | ---------| ----------- |
| `uvw` | [`texture_coordinate_info`](#texture_coordinate_info) | [`texture_coordinate_info()`](#texture_coordinate_info) |  | yes | Parameterization to be used for texture mapping. defaults to texture channel `0`. |
| `scale` | `float` | `1.0` |  | yes | Size of the features of the pattern |
| `strength` | `float` | `1.0` |  | yes | Controls the randomness of the flake orientation |
| `noise_type` | `int` | `0` | `0..1`  | no | Selects the noise type (0: classic, 1: worley) |
| `maximum_size` | `float` | `1.0` |  | yes | Controls the shape of flakes in the secondary noise mode (worley) |
| `metric` | `int` | `0` | `0..2`  | no | Metric used in the secondary noise mode (worley: 0: euclidean, 1: manhattan, 2: chebyshev) |
| `normal` | `float3` | `state::normal()` |  | yes | Base normal for the bump mapping. |

## tile_texture

Returns [`texture_return`](#texture_return)

Color tiling generator

| Name | Type | Default | Range  | Varying? | Description |
| ---- | ---- | ------- | -----  | ---------| ----------- |
| `uvw` | [`texture_coordinate_info`](#texture_coordinate_info) | [`texture_coordinate_info()`](#texture_coordinate_info) |  | yes | Parameterization to be used for texture mapping. defaults to texture channel `0`. |
| `tile_color` | `color` | `color(0)` |  | yes | Color of "bricks" in the function |
| `grout_color` | `color` | `color(1)` |  | yes | Color of "grout" between the "bricks" in the function |
| `number_of_rows` | `float` | `4.0` |  | no | Number of tile rows in the 0-1 texturing domain |
| `number_of_columns` | `float` | `4.0` |  | no | Number of tile columns in the 0-1 texturing domain |
| `grout_width` | `float` | `0.02` | `0.0..1.0`  | no | Absolute width of vertical grout lines |
| `grout_height` | `float` | `0.02` | `0.0..1.0`  | no | Absolute height of horizontal grout lines |
| `grout_roughness` | `float` | `0.0` | `0.0..1.0`  | no | Amount of noise added to grout size |
| `missing_tile_amount` | `float` | `0.0` | `0.0..1.0`  | no | Number of tiles that will end up as grout, rather than as a tile ("holes"). values are [0,1]. |
| `tile_brightness_variation` | `float` | `0.0` | `0.0..1.0`  | no | Randomization factor to the brightness of the tile color |
| `seed` | `float` | `2.284` |  | no | Seeding number for random number generator controlling tile randomization effects |
| `special_row_index` | `int` | `1` |  | no | Every nth row, width of tiles is modified. set nth all to 1 to disable. never set the nth row or column to 0.0 |
| `special_row_width_factor` | `float` | `1.0` |  | no | Change of width for tiles identified through "`special_row_index`" |
| `special_column_index` | `int` | `1` |  | no | Every nth column, height of tiles is modified. set nth all to 1 to disable. never set the nth row or column to 0.0 |
| `special_column_height_factor` | `float` | `1.0` |  | no | Change of height for tiles identified through "`special_column_index`" |
| `odd_row_offset` | `float` | `0.5` | `0.0..1.0`  | no | Controls bonding pattern. 0 will result in a "stack bond", 0.5 a "running bond" |
| `random_row_offset` | `float` | `0.0` | `0.0..1.0`  | no | Randomization factor for "`odd_row_offset`" |

## tile_bump_texture

Returns `float3`

Bump-mapping tiling generator

| Name | Type | Default | Range  | Varying? | Description |
| ---- | ---- | ------- | -----  | ---------| ----------- |
| `uvw` | [`texture_coordinate_info`](#texture_coordinate_info) | [`texture_coordinate_info()`](#texture_coordinate_info) |  | yes | Parameterization to be used for texture mapping. defaults to texture channel `0`. |
| `factor` | `float` | `1` |  | no | Strength of the bump mapping effect |
| `number_of_rows` | `float` | `4.0` |  | no | Number of tile rows in the 0-1 texturing domain |
| `number_of_columns` | `float` | `4.0` |  | no | Number of tile columns in the 0-1 texturing domain |
| `grout_width` | `float` | `0.02` | `0.0..1.0`  | no | Absolute width of vertical grout lines |
| `grout_height` | `float` | `0.02` | `0.0..1.0`  | no | Absolute height of horizontal grout lines |
| `grout_roughness` | `float` | `0.0` | `0.0..1.0`  | no | Amount of noise added to grout size |
| `missing_tile_amount` | `float` | `0.0` |  | no | Number of tiles that will end up as grout, rather than as a tile ("holes"). values are [0,1]. |
| `tile_brightness_variation` | `float` | `0.0` | `0.0..1.0`  | no | Randomization factor to the brightness of the tile color, will affect the bumpiness for that tile |
| `seed` | `float` | `2.284` |  | no | Seeding number for random number generator controlling tile randomization effects |
| `special_row_index` | `int` | `1` |  | no | Every nth row, width of tiles is modified. set nth all to 1 to disable. never set the nth row or column to 0.0 |
| `special_row_width_factor` | `float` | `1.0` |  | no | Change of width for tiles identified through "`special_row_index`" |
| `special_column_index` | `int` | `1` |  | no | Every nth column, height of tiles is modified. set nth all to 1 to disable. never set the nth row or column to 0.0 |
| `special_column_height_factor` | `float` | `1.0` |  | no | Change of height for tiles identified through "`special_column_index`" |
| `odd_row_offset` | `float` | `0.5` |  | no | Controls bonding pattern. 0 will result in a "stack bond", 0.5 a "running bond" |
| `random_row_offset` | `float` | `0.0` | `0.0..1.0`  | no | Randomization factor for "`odd_row_offset`" |
| `normal` | `float3` | `state::normal()` |  | yes | Base normal for the bump mapping. |

## anisotropy_conversion

Returns [`anisotropy_return`](#anisotropy_return)

Convert old anisotropy controls into new ones

| Name | Type | Default | Range  | Varying? | Description |
| ---- | ---- | ------- | -----  | ---------| ----------- |
| `roughness` | `float` | `0.5` | `0.0..1.0`  | yes | The base roughness value |
| `anisotropy` | `float` | `0.0` |  | yes | The anisotropy of the roughness |
| `anisotropy_rotation` | `float` | `0.0` |  | yes | Rotation of direction of anisotropy, where 1 equals 360 degrees |
| `tangent_u` | `float3` | `state::texture_tangent_u(0)` |  | yes | Tangent to align the anisotropy with. "`coordinate_source`" or "`coordinate_projection`" are possible sources |
| `mia_anisotropy_semantic` | `bool` | `false` |  | no | Allows backwards compatibility with mia material |

## blend_normals

Returns `float3`

Blend two normals for sticker-like situations where a detail normal map is applied to a base normal map

| Name | Type | Default  | Varying? | Description |
| ---- | ---- | -------  | ---------| ----------- |
| `base_normal` | `float3` | `state::normal()`  | yes | The base normal |
| `base_normal_weight` | `float` | `1.0f`  | yes | Specifies the strength of the base normal using a linear blend between state::normal and "`base_normal`" |
| `detail_normal` | `float3` | `state::normal()`  | yes | The detail normal |
| `detail_normal_weight` | `float` | `1.0f`  | yes | Specifies the strength of the detail normal using a linear blend between state::normal and "`detail_normal`" |

## lookup_volume_coefficients

Returns `volume_coefficients`

Returns the volume coefficients from an inhomogeneous volume density texture.

| Name | Type | Default  | Varying? | Description |
| ---- | ---- | -------  | ---------| ----------- |
| `density` | `texture_3d` | (none)  | no | The input 3d texture |
| `scattering_multiplier` | `color` | (none)  | no | Multiplier applied to the scattering coefficient |
| `absorption_multiplier` | `color` | (none)  | no | Multiplier applied to the absorption coefficient |
| `scattering_offset` | `color` | `float3(0.0f)`  | no | Offset applied to the scattering coefficient |
| `absorption_offset` | `color` | `float3(0.0f)`  | no | Offset applied to the absorption coefficient |
| `transform` | `float4x4` | `float4x4(1.0f)`  | no | Transform applied when looking up the density value from the texture |
| `density_multiplier` | `float` | `1.0f`  | no | Scaling applied to the density value |
| `density_relative_to_size` | `bool` | `false`  | no | If true, volume density changes with instance scale |

## blackbody_emission

Returns `color`

Blackbody emission intensity for a field of heated particles.

| Name | Type | Default  | Varying? | Description |
| ---- | ---- | -------  | ---------| ----------- |
| `temperature1` | `float` | `6500.0f`  | yes | First correlated color temperature |
| `temperature2` | `float` | `temperature1`  | yes | Second correlated color temperature |
| `blend_value` | `float` | ` 0.0f `  | yes |  |
| `density` | `texture_3d` | `texture_3d()`  | no | Density of emissive particles |
| `temperature_mult` | `float` | `1.0f`  | no | Mulitplier to convert the temperature to kelvin |
| `temperature_offset` | `float` | `0.0f`  | no | Offset to convert the temperature to kelvin |
| `scale` | `float` | `1.0f`  | no | Scale final emission |
| `tint` | `color` | `color(1,1,1)`  | no | Arbitrary color modification |
| `transform` | `float4x4` | `float4x4(1.0f)`  | no | Transform applied when looking up the density value from the texture |


