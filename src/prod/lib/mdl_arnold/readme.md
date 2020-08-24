# NVIDIA Arnold MDL Plugin

The NVIDIA Arnold MDL Plugin allows to render MDL materials using the Arnold renderer. The plugin
is based on the NVIDIA MDL SDK 2020.0 and the Arnold SDK 6.0.0.0. It provides a new shader node
called `mdl`.

In Maya the MDL material node appears in the Arnold/Shader section and is called aiMDL.
The material has only one parameter called `Qualified Name` which expects the fully qualified
material name of an MDL material that is present in a local MDL search path.

MDL search paths can be defined using the Arnold options `texture_searchpath` and `procedural_searchpath`
or by setting the environment variable `%MDL_PATHS%`. The default admin-space `%MDL_SYSTEM_PATH%` and user-space `%MDL_USER_PATH%` search paths are 
also available. The exact location of the latter is depending on the operating system.
The priority of the search paths is in exactly this order.

# Installation for Maya

### Plugin Libraries

Point the environment variable `%ARNOLD_PLUGIN_PATH%` to the `lib` directory of the `mdl_arnold` tape. Alternatively, you can copy the content of this directory, namely the files

    - lib/mdl_arnold.dll
    - lib/libmdl_sdk_ai.dll
    - lib/nv_freeimage_ai.dll
    - lib/dds_ai.dll

into the Arnold plugin directory `%MTOA_PATH%/plugins` or to any other directory that is included in the environment variable `%ARNOLD_PLUGIN_PATH%`.

### Template File

This file defines the appearance of the MDL node in Mayas attribute editor. In order to have Maya find it, point the environment variable `%MTOA_TEMPLATES_PATH%` to the `ae` directory of the `mdl_arnold` tape. Alternatively, copy the file

    - ae/aiMDLTemplate.py

to %MTOA PATH%\scripts\mtoa\ui\ae\ or to any other folder that is included in the environment variable `%MTOA_TEMPLATES_PATH%`.
  
### Rendering Omniverse scenes with MDL in Arnold for Maya

To render Omniverse MDL nodes with mtoa, you need to register the MDL Translator plugin with mtoa by adding the `ext` directory of the `mdl_arnold` tape to the environment variable `%MTOA_EXTENSIONS_PATH%`. Alternatively, copy the file

    - ext/ai_mdl_translator.dll

to %MTOA PATH%\extensions.

# Installation for 3ds Max

### Plugin Libraries

Copy the libraries in the `lib` directory, namely the files:

    - lib/mdl_arnold.dll
    - lib/libmdl_sdk_ai.dll
    - lib/nv_freeimage_ai.dll
    - lib/dds_ai.dll

into the Arnold plugin directory. The default for 3ds Max 2020 is `C:\ProgramData\Autodesk\ApplicationPlugins\MAXtoA_2020`

# Using MDL in kick with .ass files.

To add an MDL search paths in an .ass scene, you need to add the path to the global scene options:

```
options
{
  ...
  texture_searchpath "C:\local_path_1\mdl;C:\local_path_2\mdl"
}
```

or pass it on the command line via `-t`.
Add an MDL material Node just like you would add a standard_surface node. The node type is `mdl`.
The name, here `my_mdl_material_instance`, is for referencing the material from the geometry
using its `shader` attribute. The `qualified_name` specifies the MDL material to use, including all
package names, the module name and the name of the material exported by the module.

```
mdl
{
  name my_mdl_material_instance
  qualified_name "::my_package::my_module::my_mdl_material"
  declare tint constant RGB
  tint 0.75 0.25 0.25
}

sphere
{
 radius 1.2
 center 0 -0.8 0.25
 shader my_mdl_material_instance
}
```

Material parameters that are exposed by the MDL material can be specified by declaring a variable of
the matching type and assigning the desired value, like for the `tint` parameter shown above.

To run the kick renderer using a custom plugin directory, start it with the command line argument `-l`
followed by the path plugin libraries, e.g.:

```
kick.exe  -nostdin -l <mdl_arnold_directory>/lib -i <scenes_path>
```
