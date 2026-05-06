# Introduction

The Material Definition Language (MDL) module `nvidia::core_definitions` contains a collection of
MDL materials. These materials can be used either independently ("simple materials") or in
combination with other materials through the use of material combiners and modifiers. Texturing
functions provide further control and refinement of material parameter values. Together, materials,
combiners, modifiers and the texturing functions can simulate complex, real-world models of
appearance.

The core definition materials are listed in [Materials and building
blocks](#materials-and-building-blocks). The materials are divided into three groups:

* [Simple materials](#simple-materials) are used either individually to model visual appareance or
  as components when creating more complex materials with material combiners and material
  modifiers.

* [Modifier materials](#modifier-materials) are used to create new materials based on already
  existing materials. They either combine multiple materials into a new material or add additional
  features to an existing one.

* [Emissive materials](#emissive-materials) create light sources from objects by defining how light
  is emitted from an object's surface.

The functions and enumerations ("enums") used by the core materials are described in [Texturing
functions](#texturing-functions) and [Enumerations](#enumerations), respectively.

For materials and functions, two tables describe their parameters. The first lists the "display
names" used by applications for each parameter and a description of that parameter's role in the
material or function. The second table lists the display name along with that parameter's data
type, identifier, and default value. The tables in [Enumerations](#enumerations) list the field
names and their meaning.
