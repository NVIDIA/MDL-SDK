# NVIDIA MDL SDK

The NVIDIA MDL SDK enables the easy integration of MDL support into
rendering and material editing applications. The SDK contains components
for loading, inspecting, and editing of material definitions as well as
compiling MDL materials and functions to HLSL, GLSL, PTX, LLVM-IR, and
native code for x86 and arm CPUs.


## NVIDIA Material Definition Language (MDL)

![MDL example material renderings](doc/images/mdl_material_examples.jpg)

The [NVIDIA Material Definition Language (MDL)](https://www.nvidia.com/mdl) 
is a domain-specific programming language for defining physically-based 
materials for rendering. It allows you to define  *materials* and *functions*,
which you can organize in *modules* and *packages* to create flexible, 
custom-built material catalogs.

Material definitions are written in a declarative style; they define
what to compute -- not how to compute it. This is the central premise in 
MDL where one material definition delivers the same appearance in many
rendering algorithms. Following is a simple example of a diffuse material 
in MDL:

    material diffuse ( color diffuse_color = color(0.7))
        = material(
            surface: material_surface (
                scattering: df::diffuse_reflection_bsdf (
                    tint: diffuse_color
                )
            )
        );

The function definitions in MDL are written in a procedural programming
style. Their use is limited to computing material parameters in a 
side-effect-free manner.

The clear separation of material definitions from function definitions and 
their respective constraints makes possible the optimization of rendering 
algorithms independently of the material definition.


## Pre-compiled Binaries

NVIDIA offers a binary release of the MDL SDK, see 
[https://developer.nvidia.com/mdl-sdk](https://developer.nvidia.com/mdl-sdk). 
The binary release is different in some functionality as documented in the 
[Change Log](CHANGELOG.md).


## Support

- [NVIDIA MDL SDK Forum](https://devtalk.nvidia.com/default/board/253/mdl-sdk/)


## Building the MDL SDK from Source

The [INSTALL.md](INSTALL.md) file has the detailed instructions on how to build
and install the SDK.


## Repository Structure

The NVIDIA MDL SDK repository consists of the following directories and files:

    include/       - C++ API header files
    examples/      - example programs and MDL files
    src/           - source code for the SDK libraries
    doc/           - API documentation, MDL specification, 
                     core_definitions.mdl and base.mdl documentation
    cmake/         - support files for the CMAKE build system

    INSTALL.md     - how to build and install the SDK
    README.md      - this file: introduction and build instructions
    CHANGELOG.md   - change log and difference to the binary MDL SDK release
    LICENSE.md     - license for the MDL SDK and references to 
                     third-party licenses
    CMakeLists.txt - top level CMAKE file


## Additional Resources

- [NVIDIA MDL Home Page](https://www.nvidia.com/mdl)
- [NVIDIA MDL SDK Forum](https://devtalk.nvidia.com/default/board/253/mdl-sdk/)
