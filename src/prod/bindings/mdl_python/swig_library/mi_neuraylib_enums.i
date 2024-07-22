/******************************************************************************
 * Copyright 2024 NVIDIA Corporation. All rights reserved.
 *****************************************************************************/


// SWIG converts all enums to integers
// To use the enum type in python we wrap enums manually and this requires us to manually pack and unpack them
// For functions that return an enum, we just put the return value in the python enum constructor using this macro.
// For all enums that are passes as parameters use the `%pythonprepend` feature!
%define WRAP_ENUM_RETURN(NAMESPACE, TYPENAME, FUNCTION, PYTHON_ENUM_NAME)
    %feature("pythonappend") NAMESPACE::TYPENAME::FUNCTION %{ val = PYTHON_ENUM_NAME ## (val) %}
%enddef


// special handling for: mi::neuraylib::INeuray::Status
// ----------------------------------------------------------------------------

// We manually define the enums in the correct proxy class
%extend SmartPtr< mi::neuraylib::INeuray > {
    %pythoncode {

        @post_swig_add_type_hint_mapping("mi::neuraylib::INeuray::Status", "INeuray.Status")
        class Status(Enum):
            r""" The operational status of the library."""

            PRE_STARTING = _pymdlsdk._INeuray_PRE_STARTING
            r""" The library or the cluster has not yet been started."""
            STARTING = _pymdlsdk._INeuray_STARTING
            r""" The library or the cluster is starting."""
            STARTED = _pymdlsdk._INeuray_STARTED
            r""" The library or the cluster is ready for operation."""
            SHUTTINGDOWN = _pymdlsdk._INeuray_SHUTTINGDOWN
            r""" The library or the cluster is shutting down."""
            SHUTDOWN = _pymdlsdk._INeuray_SHUTDOWN
            r""" The library or the cluster has been shut down."""
            FAILURE = _pymdlsdk._INeuray_FAILURE
            r""" There was a failure during operation."""
            FORCE_32_BIT = _pymdlsdk._INeuray_FORCE_32_BIT
    }
}

// unwrap all usages of the enum values in arguments
// - NONE-

// Handle all functions that return this enum type
WRAP_ENUM_RETURN(mi::neuraylib, INeuray, get_status, INeuray.Status)



// special handling for: mi::neuraylib::Mdl_version
// ----------------------------------------------------------------------------
%pythoncode {

    @post_swig_add_type_hint_mapping("mi::neuraylib::Mdl_version", "Mdl_version")
    class Mdl_version(Enum):
        r""" The MDL version."""

        MDL_VERSION_1_0 = _pymdlsdk.MDL_VERSION_1_0
        r""" MDL version 1.0"""
        MDL_VERSION_1_1 = _pymdlsdk.MDL_VERSION_1_1
        r""" MDL version 1.1"""
        MDL_VERSION_1_2 = _pymdlsdk.MDL_VERSION_1_2
        r""" MDL version 1.2"""
        MDL_VERSION_1_3 = _pymdlsdk.MDL_VERSION_1_3
        r""" MDL version 1.3"""
        MDL_VERSION_1_4 = _pymdlsdk.MDL_VERSION_1_4
        r""" MDL version 1.4"""
        MDL_VERSION_1_5 = _pymdlsdk.MDL_VERSION_1_5
        r""" MDL version 1.5"""
        MDL_VERSION_1_6 = _pymdlsdk.MDL_VERSION_1_6
        r""" MDL version 1.6"""
        MDL_VERSION_1_7 = _pymdlsdk.MDL_VERSION_1_7
        r""" MDL version 1.7"""
        MDL_VERSION_1_8 = _pymdlsdk.MDL_VERSION_1_8
        r""" MDL version 1.8"""
        MDL_VERSION_1_9 = _pymdlsdk.MDL_VERSION_1_9
        r""" MDL version 1.9"""
        MDL_VERSION_EXP = _pymdlsdk.MDL_VERSION_EXP
        r""" MDL experimental features."""
        MDL_VERSION_LATEST = _pymdlsdk.MDL_VERSION_LATEST
        r""" Latest MDL version"""
        MDL_VERSION_INVALID = _pymdlsdk.MDL_VERSION_INVALID
        r""" Invalid MDL version"""
        MDL_VERSION_FORCE_32_BIT = _pymdlsdk.MDL_VERSION_FORCE_32_BIT
}
// Handle all functions that return an Mdl_version
WRAP_ENUM_RETURN(mi::neuraylib, IModule, get_mdl_version, Mdl_version)

// Handle Mdl_version as out parameter (tuple return in python)
%apply int* OUTPUT { mi::neuraylib::Mdl_version&, mi::neuraylib::Mdl_version& };
%define WRAP_GET_MDL_VERSIONS_FUNCTIONS(IINTERFACE_TYPE)
    %rename(_get_mdl_version) IINTERFACE_TYPE::get_mdl_version;
    %extend SmartPtr<IINTERFACE_TYPE> {
        %pythoncode{
            def get_mdl_version(self) -> tuple[Mdl_version, Mdl_version]:
                r"""
                Returns the MDL version when this function definition was added and removed.
                :rtype: (Mdl_version, Mdl_version)
                :return: A tuple of (since, removed) versions.
                         since     The MDL version in which this function definition was added. If the
                                   function definition does not belong to the standard library, the
                                   MDL version of the corresponding module is returned.
                         removed   The MDL version in which this function definition was removed, or
                                   mi::neuraylib::MDL_VERSION_INVALID if the function has not been
                                   removed so far or does not belong to the standard library.
                """
                since, remove = self._get_mdl_version()
                return (Mdl_version(since), Mdl_version(remove))
        }
    }
%enddef
WRAP_GET_MDL_VERSIONS_FUNCTIONS(mi::neuraylib::IAnnotation_definition)
WRAP_GET_MDL_VERSIONS_FUNCTIONS(mi::neuraylib::IFunction_definition)

// unwrap all usages of the enum values in arguments
%feature("pythonprepend") mi::neuraylib::IMdl_factory::create_module_builder %{
    min_module_version = min_module_version.value  # unwrap python enum and pass the integer value
    max_module_version = max_module_version.value  # unwrap python enum and pass the integer value
%}
%feature("pythonprepend") mi::neuraylib::IMdl_module_transformer::upgrade_mdl_version%{
    version = version.value  # unwrap python enum and pass the integer value
%}


// special handling for: mi::base::Message_severity
// ----------------------------------------------------------------------------
%pythoncode{

    class Message_severity(Enum):
        r"""Constants for possible message severities."""

        MESSAGE_SEVERITY_FATAL = _pymdlsdk.MESSAGE_SEVERITY_FATAL
        r""" A fatal error has occurred."""
        MESSAGE_SEVERITY_ERROR = _pymdlsdk.MESSAGE_SEVERITY_ERROR
        r""" An error has occurred."""
        MESSAGE_SEVERITY_WARNING = _pymdlsdk.MESSAGE_SEVERITY_WARNING
        r""" A warning has occurred."""
        MESSAGE_SEVERITY_INFO = _pymdlsdk.MESSAGE_SEVERITY_INFO
        r""" This is a normal operational message."""
        MESSAGE_SEVERITY_VERBOSE = _pymdlsdk.MESSAGE_SEVERITY_VERBOSE
        r""" This is a more verbose message."""
        MESSAGE_SEVERITY_DEBUG = _pymdlsdk.MESSAGE_SEVERITY_DEBUG
        r""" This is debug message."""
        MESSAGE_SEVERITY_FORCE_32_BIT = _pymdlsdk.MESSAGE_SEVERITY_FORCE_32_BIT
}

// unwrap all usages of the enum values in arguments
// mi::neuraylib::IMdl_execution_context::add_message is added further down after IMessage::Kind is defined
// - NONE-

// Handle all functions that return this enum type
WRAP_ENUM_RETURN(mi::neuraylib, IMessage, get_severity, Message_severity)



// special handling for: mi::neuraylib::Material_slot
// ----------------------------------------------------------------------------

%pythoncode{

    @post_swig_add_type_hint_mapping("mi::neuraylib::Material_slot", "Material_slot")
    class Material_slot(Enum):
        r"""
        Material slots identify parts of a compiled material.
        See #mi::neuraylib::ICompiled_material and #mi::neuraylib::ICompiled_material::get_slot_hash().
        """

        SLOT_THIN_WALLED = _pymdlsdk.SLOT_THIN_WALLED
        r""" Slot thin_walled"""
        SLOT_SURFACE_SCATTERING = _pymdlsdk.SLOT_SURFACE_SCATTERING
        r""" Slot surface.scattering"""
        SLOT_SURFACE_EMISSION_EDF_EMISSION = _pymdlsdk.SLOT_SURFACE_EMISSION_EDF_EMISSION
        r""" Slot surface.emission.emission"""
        SLOT_SURFACE_EMISSION_INTENSITY = _pymdlsdk.SLOT_SURFACE_EMISSION_INTENSITY
        r""" Slot surface.emission.intensity"""
        SLOT_SURFACE_EMISSION_MODE = _pymdlsdk.SLOT_SURFACE_EMISSION_MODE
        r""" Slot surface.emission.mode"""
        SLOT_BACKFACE_SCATTERING = _pymdlsdk.SLOT_BACKFACE_SCATTERING
        r""" Slot backface.scattering"""
        SLOT_BACKFACE_EMISSION_EDF_EMISSION = _pymdlsdk.SLOT_BACKFACE_EMISSION_EDF_EMISSION
        r""" Slot backface.emission.emission"""
        SLOT_BACKFACE_EMISSION_INTENSITY = _pymdlsdk.SLOT_BACKFACE_EMISSION_INTENSITY
        r""" Slot backface.emission.intensity"""
        SLOT_BACKFACE_EMISSION_MODE = _pymdlsdk.SLOT_BACKFACE_EMISSION_MODE
        r""" Slot backface.emission.mode"""
        SLOT_IOR = _pymdlsdk.SLOT_IOR
        r""" Slot ior"""
        SLOT_VOLUME_SCATTERING = _pymdlsdk.SLOT_VOLUME_SCATTERING
        r""" Slot volume.scattering"""
        SLOT_VOLUME_ABSORPTION_COEFFICIENT = _pymdlsdk.SLOT_VOLUME_ABSORPTION_COEFFICIENT
        r""" Slot volume.absorption_coefficient"""
        SLOT_VOLUME_SCATTERING_COEFFICIENT = _pymdlsdk.SLOT_VOLUME_SCATTERING_COEFFICIENT
        r""" Slot volume.scattering_coefficient"""
        SLOT_VOLUME_EMISSION_INTENSITY = _pymdlsdk.SLOT_VOLUME_EMISSION_INTENSITY
        r""" Slot volume.emission_intensity"""
        SLOT_GEOMETRY_DISPLACEMENT = _pymdlsdk.SLOT_GEOMETRY_DISPLACEMENT
        r""" Slot geometry.displacement"""
        SLOT_GEOMETRY_CUTOUT_OPACITY = _pymdlsdk.SLOT_GEOMETRY_CUTOUT_OPACITY
        r""" Slot geometry.cutout_opacity"""
        SLOT_GEOMETRY_NORMAL = _pymdlsdk.SLOT_GEOMETRY_NORMAL
        r""" Slot geometry.normal"""
        SLOT_HAIR = _pymdlsdk.SLOT_HAIR
        r""" Slot hair"""
        SLOT_FIRST = _pymdlsdk.SLOT_FIRST
        r""" First slot"""
        SLOT_LAST = _pymdlsdk.SLOT_LAST
        r""" Last slot"""
        SLOT_FORCE_32_BIT = _pymdlsdk.SLOT_FORCE_32_BIT
}

// unwrap all usages of the enum values in arguments
%feature("pythonprepend") mi::neuraylib::ICompiled_material::get_slot_hash %{
    slot = slot.value  # unwrap python enum and pass the integer value
%}

// Handle all functions that return this enum type
// - NONE-



// special handling for: mi::neuraylib::Material_opacity
// ----------------------------------------------------------------------------

%pythoncode{

    @post_swig_add_type_hint_mapping("mi::neuraylib::Material_opacity", "Material_opacity")
    class Material_opacity(Enum):
        r"""
        The compiled material's opacity.

        See #mi::neuraylib::ICompiled_material::get_opacity() and #mi::neuraylib::ICompiled_material::get_surface_opacity().
        """

        OPACITY_OPAQUE = _pymdlsdk.OPACITY_OPAQUE
        r""" material is opaque"""
        OPACITY_TRANSPARENT = _pymdlsdk.OPACITY_TRANSPARENT
        r""" material is transparent"""
        OPACITY_UNKNOWN = _pymdlsdk.OPACITY_UNKNOWN
        r""" material might be transparent"""
        OPACITY_FORCE_32_BIT = _pymdlsdk.OPACITY_FORCE_32_BIT
}

// unwrap all usages of the enum values in arguments
// - NONE-

// Handle all functions that return this enum type
WRAP_ENUM_RETURN(mi::neuraylib, ICompiled_material, get_opacity, Material_opacity)
WRAP_ENUM_RETURN(mi::neuraylib, ICompiled_material, get_surface_opacity, Material_opacity)



// special handling for: mi::neuraylib::Uvtile_mode
// ----------------------------------------------------------------------------

%pythoncode {

    @post_swig_add_type_hint_mapping("mi::neuraylib::Uvtile_mode", "Uvtile_mode")
    class Uvtile_mode(Enum):
        r"""
        Supported uvtile modes for resources.

        For light profiles and BSDF measurements only #mi::neuraylib::UVTILE_MODE_NONE is valid.
        See #mi::neuraylib::IImage::reset_file() for details about the different modes.
        """

        UVTILE_MODE_NONE = _pymdlsdk.UVTILE_MODE_NONE
        r""" No uvtile mode."""
        UVTILE_MODE_UDIM = _pymdlsdk.UVTILE_MODE_UDIM
        r""" The UDIM uvtile mode."""
        UVTILE_MODE_UVTILE0 = _pymdlsdk.UVTILE_MODE_UVTILE0
        r""" The UVTILE0 uvtile mode."""
        UVTILE_MODE_UVTILE1 = _pymdlsdk.UVTILE_MODE_UVTILE1
        r""" The UVTILE1 uvtile mode."""
        UVTILE_MODE_FORCE_32_BIT = _pymdlsdk.UVTILE_MODE_FORCE_32_BIT
}

// unwrap all usages of the enum values in arguments
// - NONE-

// Handle all functions that return this enum type
WRAP_ENUM_RETURN(mi::neuraylib, IMdl_resolved_resource, get_uvtile_mode, Uvtile_mode)




// special handling for: mi::neuraylib::IExpression::Kind
// ----------------------------------------------------------------------------

// We manually define the enums in the correct proxy class
%extend SmartPtr<mi::neuraylib::IExpression> {
    %pythoncode{

        @post_swig_add_type_hint_mapping("mi::neuraylib::IExpression::Kind", "IExpression.Kind")
        class Kind(Enum) :
            r"""The possible kinds of expressions."""

            EK_CONSTANT = _pymdlsdk._IExpression_EK_CONSTANT
            r""" A constant expression. See #mi::neuraylib::IExpression_constant."""
            EK_CALL = _pymdlsdk._IExpression_EK_CALL
            r""" An indirect call expression. See #mi::neuraylib::IExpression_call."""
            EK_PARAMETER = _pymdlsdk._IExpression_EK_PARAMETER
            r""" A parameter reference expression. See #mi::neuraylib::IExpression_parameter."""
            EK_DIRECT_CALL = _pymdlsdk._IExpression_EK_DIRECT_CALL
            r""" A direct call expression. See #mi::neuraylib::IExpression_direct_call."""
            EK_TEMPORARY = _pymdlsdk._IExpression_EK_TEMPORARY
            r""" A temporary reference expression. See #mi::neuraylib::IExpression_temporary."""
            EK_FORCE_32_BIT = _pymdlsdk._IExpression_EK_FORCE_32_BIT
    }
}

// unwrap all usages of the enum values in arguments
// - NONE-

// Handle all functions that return this enum type
WRAP_ENUM_RETURN(mi::neuraylib, IExpression, get_kind, IExpression.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IExpression_constant, get_kind, IExpression.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IExpression_call, get_kind, IExpression.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IExpression_parameter, get_kind, IExpression.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IExpression_direct_call, get_kind, IExpression.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IExpression_temporary, get_kind, IExpression.Kind)


// special handling for: mi::neuraylib::IExpression_factory::Comparison_options
// ----------------------------------------------------------------------------

// We manually define the enums in the correct proxy class
%extend SmartPtr<mi::neuraylib::IExpression_factory> {
    %pythoncode{

        @post_swig_add_type_hint_mapping("mi::neuraylib::IExpression_factory::Comparison_options", "IExpression_factory.Comparison_options")
        class Comparison_options(Enum) :
            r"""
            Various options for the comparison of expressions or expression lists.

            see The \p flags parameter of #compare()
            """

            DEFAULT_OPTIONS = _pymdlsdk._IExpression_factory_DEFAULT_OPTIONS
            r""" Default comparison options."""
            DEEP_CALL_COMPARISONS = _pymdlsdk._IExpression_factory_DEEP_CALL_COMPARISONS
            r"""
            This option indicates that call expressions should be compared for equality, not for
            identity.That is, the comparison is not done via
            #mi::neuraylib::IExpression::get_value(), but by traversing into the referenced
            function call, i.e., comparing the function definition reference and the arguments.
            This option is useful if you want to decide whether an argument is* semantically*
            equal to the corresponding default parameter.
            """
            SKIP_TYPE_ALIASES = _pymdlsdk._IExpression_factory_SKIP_TYPE_ALIASES
            r"""
            This option indicates that all type aliases should be skipped before types of
            expression are compared.Defaults and argument might sometimes differ in explicit type
            modifiers, therefore this option is useful if you want to decide whether an argument is
            * semantically* equal to the corresponding default parameter.
            """
            COMPARISON_OPTIONS_FORCE_32_BIT = _pymdlsdk._IExpression_factory_COMPARISON_OPTIONS_FORCE_32_BIT
    }
}

// unwrap all usages of the enum values in arguments
// - NONE-

// Handle all functions that return this enum type
// - NONE-




// special handling for: mi::neuraylib::IMaterial_instance::Compilation_options
// ----------------------------------------------------------------------------

// We manually define the enums in the correct proxy class
%extend SmartPtr<mi::neuraylib::IMaterial_instance> {
    %pythoncode{

        @post_swig_add_type_hint_mapping("mi::neuraylib::IMaterial_instance::Compilation_options", "IMaterial_instance.Compilation_options")
        class Compilation_options(Enum) :
            r"""Various options for the creation of compiled materials."""

            DEFAULT_OPTIONS = _pymdlsdk._IMaterial_instance_DEFAULT_OPTIONS
            r""" Default compilation options (e.g., instance compilation)."""
            CLASS_COMPILATION = _pymdlsdk._IMaterial_instance_CLASS_COMPILATION
            r""" Selects class compilation instead of instance compilation."""
            COMPILATION_OPTIONS_FORCE_32_BIT = _pymdlsdk._IMaterial_instance_COMPILATION_OPTIONS_FORCE_32_BIT
    }
}

// unwrap all usages of the enum values in arguments
// - NONE-

// Handle all functions that return this enum type
// - NONE-



// special handling for: mi::neuraylib::IDatabase::Garbage_collection_priority
// ----------------------------------------------------------------------------

// We manually define the enums in the correct proxy class
%extend SmartPtr<mi::neuraylib::IDatabase> {
    %pythoncode{

        @post_swig_add_type_hint_mapping("mi::neuraylib::Garbage_collection_priority::Garbage_collection_priority", "Garbage_collection_priority.Garbage_collection_priority")
        class Garbage_collection_priority(Enum) :
            r"""Priorities for synchronous garbage collection runs."""

            PRIORITY_LOW = _pymdlsdk._IDatabase_PRIORITY_LOW
            r"""
            Low priority for synchronous garbage collection runs.Use this priority if the
            performance of other concurrent DB operations is more important than a fast synchronous
            garbage collection.
            """
            PRIORITY_MEDIUM = _pymdlsdk._IDatabase_PRIORITY_MEDIUM
            r"""
            Medium priority for synchronous garbage collection runs.This priority attempts to
            maintain a balance between the synchronous garbage collection and other concurrent DB
            operations.
            """
            PRIORITY_HIGH = _pymdlsdk._IDatabase_PRIORITY_HIGH
            r"""
            High priority for synchronous garbage collection runs.Other concurrent DB operations
            will experience a large performance drop.Therefore, this priority should not be used
            in multi - user settings.
            """
            PRIORITY_FORCE_32_BIT = _pymdlsdk._IDatabase_PRIORITY_FORCE_32_BIT
    }
}

// unwrap all usages of the enum values in arguments
%feature("pythonprepend") mi::neuraylib::IDatabase::garbage_collection %{
    argc: int = len(args)
    if argc == 0:
        args = (IDatabase.Garbage_collection_priority.PRIORITY_LOW.value, )
    if argc == 1:
        args = (args[0].value, ) # unwrap python enum and pass the integer value
%}

// Handle all functions that return this enum type
// - NONE-




// special handling for: mi::neuraylib::IMessage::Kind
// ----------------------------------------------------------------------------

%extend SmartPtr<mi::neuraylib::IMessage> {
    %pythoncode{

        @post_swig_add_type_hint_mapping("mi::neuraylib::IMessage::Kind", "IMessage.Kind")
        class Kind(Enum) :
            r"""
            The possible kinds of messages.
            A message can be uniquely identified by the message code and kind, except for uncategorized messages.
            """

            MSG_COMILER_CORE = _pymdlsdk._IMessage_MSG_COMILER_CORE
            r""" MDL Core compiler message."""
            MSG_COMILER_BACKEND = _pymdlsdk._IMessage_MSG_COMILER_BACKEND
            r""" MDL Core compiler backend message."""
            MSG_COMPILER_DAG = _pymdlsdk._IMessage_MSG_COMPILER_DAG
            r""" MDL Core DAG generator message."""
            MSG_COMPILER_ARCHIVE_TOOL = _pymdlsdk._IMessage_MSG_COMPILER_ARCHIVE_TOOL
            r""" MDL Core archive tool message."""
            MSG_IMP_EXP = _pymdlsdk._IMessage_MSG_IMP_EXP
            r""" MDL import/exporter message."""
            MSG_INTEGRATION = _pymdlsdk._IMessage_MSG_INTEGRATION
            r""" MDL integration message."""
            MSG_UNCATEGORIZED = _pymdlsdk._IMessage_MSG_UNCATEGORIZED
            r""" Uncategorized messages do not have a code."""
            MSG_FORCE_32_BIT = _pymdlsdk._IMessage_MSG_FORCE_32_BIT
    }
}

// unwrap all usages of the enum values in arguments
%feature("pythonprepend") mi::neuraylib::IMdl_execution_context::add_message %{
    kind = kind.value  # unwrap python enum and pass the integer value
    severity = severity.value  # unwrap python enum and pass the integer value
%}

// Handle all functions that return this enum type
WRAP_ENUM_RETURN(mi::neuraylib, IMessage, get_kind, IMessage.Kind)



// special handling for: mi::neuraylib::Element_type
// ----------------------------------------------------------------------------
// We manually define the enums
%pythoncode{

    @post_swig_add_type_hint_mapping("mi::neuraylib::Element_type", "Element_type")
    class Element_type(Enum):
        r"""Distinguishes scene elements. See #mi::neuraylib::IScene_element::get_element_type()."""

        ELEMENT_TYPE_INSTANCE = _pymdlsdk.ELEMENT_TYPE_INSTANCE
        r""" #mi::neuraylib::IInstance"""
        ELEMENT_TYPE_GROUP = _pymdlsdk.ELEMENT_TYPE_GROUP
        r""" #mi::neuraylib::IGroup"""
        ELEMENT_TYPE_OPTIONS = _pymdlsdk.ELEMENT_TYPE_OPTIONS
        r""" #mi::neuraylib::IOptions"""
        ELEMENT_TYPE_CAMERA = _pymdlsdk.ELEMENT_TYPE_CAMERA
        r""" #mi::neuraylib::ICamera"""
        ELEMENT_TYPE_LIGHT = _pymdlsdk.ELEMENT_TYPE_LIGHT
        r""" #mi::neuraylib::ILight"""
        ELEMENT_TYPE_LIGHTPROFILE = _pymdlsdk.ELEMENT_TYPE_LIGHTPROFILE
        r""" #mi::neuraylib::ILightprofile"""
        ELEMENT_TYPE_TEXTURE = _pymdlsdk.ELEMENT_TYPE_TEXTURE
        r""" #mi::neuraylib::ITexture"""
        ELEMENT_TYPE_IMAGE = _pymdlsdk.ELEMENT_TYPE_IMAGE
        r""" #mi::neuraylib::IImage"""
        ELEMENT_TYPE_TRIANGLE_MESH = _pymdlsdk.ELEMENT_TYPE_TRIANGLE_MESH
        r""" #mi::neuraylib::ITriangle_mesh"""
        ELEMENT_TYPE_ATTRIBUTE_CONTAINER = _pymdlsdk.ELEMENT_TYPE_ATTRIBUTE_CONTAINER
        r""" #mi::neuraylib::IAttribute_container"""
        ELEMENT_TYPE_POLYGON_MESH = _pymdlsdk.ELEMENT_TYPE_POLYGON_MESH
        r""" #mi::neuraylib::IPolygon_mesh"""
        ELEMENT_TYPE_SUBDIVISION_SURFACE = _pymdlsdk.ELEMENT_TYPE_SUBDIVISION_SURFACE
        r""" #mi::neuraylib::ISubdivision_surface"""
        ELEMENT_TYPE_FREEFORM_SURFACE = _pymdlsdk.ELEMENT_TYPE_FREEFORM_SURFACE
        r""" #mi::neuraylib::IFreeform_surface"""
        ELEMENT_TYPE_FIBERS = _pymdlsdk.ELEMENT_TYPE_FIBERS
        r""" #mi::neuraylib::IFibers"""
        ELEMENT_TYPE_VOLUME = _pymdlsdk.ELEMENT_TYPE_VOLUME
        r""" #mi::neuraylib::IVolume"""
        ELEMENT_TYPE_VOLUME_DATA = _pymdlsdk.ELEMENT_TYPE_VOLUME_DATA
        r""" #mi::neuraylib::IVolume_data"""
        ELEMENT_TYPE_PARTICLES = _pymdlsdk.ELEMENT_TYPE_PARTICLES
        r""" #mi::neuraylib::IParticles"""
        ELEMENT_TYPE_MODULE = _pymdlsdk.ELEMENT_TYPE_MODULE
        r""" #mi::neuraylib::IModule"""
        ELEMENT_TYPE_FUNCTION_DEFINITION = _pymdlsdk.ELEMENT_TYPE_FUNCTION_DEFINITION
        r""" #mi::neuraylib::IFunction_definition"""
        ELEMENT_TYPE_FUNCTION_CALL = _pymdlsdk.ELEMENT_TYPE_FUNCTION_CALL
        r""" #mi::neuraylib::IFunction_call"""
        ELEMENT_TYPE_MATERIAL_INSTANCE = _pymdlsdk.ELEMENT_TYPE_MATERIAL_INSTANCE
        r""" #mi::neuraylib::IMaterial_instance"""
        ELEMENT_TYPE_COMPILED_MATERIAL = _pymdlsdk.ELEMENT_TYPE_COMPILED_MATERIAL
        r""" #mi::neuraylib::ICompiled_material"""
        ELEMENT_TYPE_BSDF_MEASUREMENT = _pymdlsdk.ELEMENT_TYPE_BSDF_MEASUREMENT
        r""" #mi::neuraylib::IBsdf_measurement"""
        ELEMENT_TYPE_IRRADIANCE_PROBES = _pymdlsdk.ELEMENT_TYPE_IRRADIANCE_PROBES
        r""" #mi::neuraylib::IIrradiance_probes"""
        ELEMENT_TYPE_DECAL = _pymdlsdk.ELEMENT_TYPE_DECAL
        r""" #mi::neuraylib::IDecal"""
        ELEMENT_TYPE_ON_DEMAND_MESH = _pymdlsdk.ELEMENT_TYPE_ON_DEMAND_MESH
        r""" #mi::neuraylib::IOn_demand_mesh"""
        ELEMENT_TYPE_PROJECTOR = _pymdlsdk.ELEMENT_TYPE_PROJECTOR
        r""" #mi::neuraylib::IProjector"""
        ELEMENT_TYPE_SECTION_OBJECT = _pymdlsdk.ELEMENT_TYPE_SECTION_OBJECT
        r""" #mi::neuraylib::ISection_object"""
        ELEMENT_TYPE_PROXY = _pymdlsdk.ELEMENT_TYPE_PROXY
        r""" #mi::neuraylib::IProxy"""
        ELEMENT_TYPE_FORCE_32_BIT = _pymdlsdk.ELEMENT_TYPE_FORCE_32_BIT
}

// unwrap all usages of the enum values in arguments
// - NONE-

// Handle all functions that return this enum type
WRAP_ENUM_RETURN(mi::neuraylib, IScene_element, get_element_type, Element_type)
WRAP_ENUM_RETURN(mi::neuraylib, IBsdf_measurement, get_element_type, Element_type)
WRAP_ENUM_RETURN(mi::neuraylib, ICompiled_material, get_element_type, Element_type)
WRAP_ENUM_RETURN(mi::neuraylib, IFunction_call, get_element_type, Element_type)
WRAP_ENUM_RETURN(mi::neuraylib, IFunction_definition, get_element_type, Element_type)
WRAP_ENUM_RETURN(mi::neuraylib, IImage, get_element_type, Element_type)
WRAP_ENUM_RETURN(mi::neuraylib, ILight_profile, get_element_type, Element_type)
WRAP_ENUM_RETURN(mi::neuraylib, IMaterial_instance, get_element_type, Element_type)
WRAP_ENUM_RETURN(mi::neuraylib, IModule, get_element_type, Element_type)
WRAP_ENUM_RETURN(mi::neuraylib, ITexture, get_element_type, Element_type)



// special handling for: mi::neuraylib::IFunction_definition::Semantics
// ----------------------------------------------------------------------------

%extend SmartPtr<mi::neuraylib::IFunction_definition> {
    %pythoncode{

        @post_swig_add_type_hint_mapping("mi::neuraylib::IFunction_definition::Semantics", "IFunction_definition.Semantics")
        class Semantics(Enum):
            r"""
            All known semantics of functions definitions.

            Material definitions always have the semantic #DS_UNKNOWN.

            note  Do not rely on the numeric values of the enumerators since they may change without
                  further notice.
            """

            DS_UNKNOWN = _pymdlsdk._IFunction_definition_DS_UNKNOWN
            r""" Unknown semantics."""
            DS_CONV_CONSTRUCTOR = _pymdlsdk._IFunction_definition_DS_CONV_CONSTRUCTOR
            r""" The conversion constructor."""
            DS_ELEM_CONSTRUCTOR = _pymdlsdk._IFunction_definition_DS_ELEM_CONSTRUCTOR
            r""" The elemental constructor."""
            DS_COLOR_SPECTRUM_CONSTRUCTOR = _pymdlsdk._IFunction_definition_DS_COLOR_SPECTRUM_CONSTRUCTOR
            r""" The color from spectrum constructor."""
            DS_MATRIX_ELEM_CONSTRUCTOR = _pymdlsdk._IFunction_definition_DS_MATRIX_ELEM_CONSTRUCTOR
            r""" The matrix elemental constructor."""
            DS_MATRIX_DIAG_CONSTRUCTOR = _pymdlsdk._IFunction_definition_DS_MATRIX_DIAG_CONSTRUCTOR
            r""" The matrix diagonal constructor."""
            DS_INVALID_REF_CONSTRUCTOR = _pymdlsdk._IFunction_definition_DS_INVALID_REF_CONSTRUCTOR
            r""" The invalid reference constructor."""
            DS_DEFAULT_STRUCT_CONSTRUCTOR = _pymdlsdk._IFunction_definition_DS_DEFAULT_STRUCT_CONSTRUCTOR
            r""" The default constructor for a struct."""
            DS_TEXTURE_CONSTRUCTOR = _pymdlsdk._IFunction_definition_DS_TEXTURE_CONSTRUCTOR
            r""" The texture constructor."""
            DS_CONV_OPERATOR = _pymdlsdk._IFunction_definition_DS_CONV_OPERATOR
            r""" The type conversion operator."""
            DS_COPY_CONSTRUCTOR = _pymdlsdk._IFunction_definition_DS_COPY_CONSTRUCTOR
            r""" The copy constructor."""
            DS_BITWISE_COMPLEMENT = _pymdlsdk._IFunction_definition_DS_BITWISE_COMPLEMENT
            r""" The bitwise complement operator."""
            DS_UNARY_FIRST = _pymdlsdk._IFunction_definition_DS_UNARY_FIRST
            DS_OPERATOR_FIRST = _pymdlsdk._IFunction_definition_DS_OPERATOR_FIRST
            DS_LOGICAL_NOT = _pymdlsdk._IFunction_definition_DS_LOGICAL_NOT
            r""" The unary logical negation operator."""
            DS_POSITIVE = _pymdlsdk._IFunction_definition_DS_POSITIVE
            r""" The unary arithmetic positive operator."""
            DS_NEGATIVE = _pymdlsdk._IFunction_definition_DS_NEGATIVE
            r""" The unary arithmetic negation operator."""
            DS_PRE_INCREMENT = _pymdlsdk._IFunction_definition_DS_PRE_INCREMENT
            r""" The pre-increment operator."""
            DS_PRE_DECREMENT = _pymdlsdk._IFunction_definition_DS_PRE_DECREMENT
            r""" The pre-decrement operator."""
            DS_POST_INCREMENT = _pymdlsdk._IFunction_definition_DS_POST_INCREMENT
            r""" The post-increment operator."""
            DS_POST_DECREMENT = _pymdlsdk._IFunction_definition_DS_POST_DECREMENT
            r""" The post-decrement operator."""
            DS_CAST = _pymdlsdk._IFunction_definition_DS_CAST
            r"""  The cast operator. See 'mi_neuray_mdl_cast_operator'."""
            DS_UNARY_LAST = _pymdlsdk._IFunction_definition_DS_UNARY_LAST
            DS_SELECT = _pymdlsdk._IFunction_definition_DS_SELECT
            r""" The select operator."""
            DS_BINARY_FIRST = _pymdlsdk._IFunction_definition_DS_BINARY_FIRST
            DS_ARRAY_INDEX = _pymdlsdk._IFunction_definition_DS_ARRAY_INDEX
            r""" The array index operator. See 'mi_neuray_mdl_array_index_operator'."""
            DS_MULTIPLY = _pymdlsdk._IFunction_definition_DS_MULTIPLY
            r""" The multiplication operator."""
            DS_DIVIDE = _pymdlsdk._IFunction_definition_DS_DIVIDE
            r""" The division operator."""
            DS_MODULO = _pymdlsdk._IFunction_definition_DS_MODULO
            r""" The modulus operator."""
            DS_PLUS = _pymdlsdk._IFunction_definition_DS_PLUS
            r""" The addition operator."""
            DS_MINUS = _pymdlsdk._IFunction_definition_DS_MINUS
            r""" The subtraction operator."""
            DS_SHIFT_LEFT = _pymdlsdk._IFunction_definition_DS_SHIFT_LEFT
            r""" The shift-left operator."""
            DS_SHIFT_RIGHT = _pymdlsdk._IFunction_definition_DS_SHIFT_RIGHT
            r""" The arithmetic shift-right operator."""
            DS_UNSIGNED_SHIFT_RIGHT = _pymdlsdk._IFunction_definition_DS_UNSIGNED_SHIFT_RIGHT
            r""" The unsigned shift-right operator."""
            DS_LESS = _pymdlsdk._IFunction_definition_DS_LESS
            r""" The less operator."""
            DS_LESS_OR_EQUAL = _pymdlsdk._IFunction_definition_DS_LESS_OR_EQUAL
            r""" The less-or-equal operator."""
            DS_GREATER_OR_EQUAL = _pymdlsdk._IFunction_definition_DS_GREATER_OR_EQUAL
            r""" The greater-or-equal operator."""
            DS_GREATER = _pymdlsdk._IFunction_definition_DS_GREATER
            r""" The greater operator."""
            DS_EQUAL = _pymdlsdk._IFunction_definition_DS_EQUAL
            r""" The equal operator."""
            DS_NOT_EQUAL = _pymdlsdk._IFunction_definition_DS_NOT_EQUAL
            r""" The not-equal operator."""
            DS_BITWISE_AND = _pymdlsdk._IFunction_definition_DS_BITWISE_AND
            r""" The bitwise and operator."""
            DS_BITWISE_XOR = _pymdlsdk._IFunction_definition_DS_BITWISE_XOR
            r""" The bitwise xor operator."""
            DS_BITWISE_OR = _pymdlsdk._IFunction_definition_DS_BITWISE_OR
            r""" The bitwise or operator."""
            DS_LOGICAL_AND = _pymdlsdk._IFunction_definition_DS_LOGICAL_AND
            r""" The logical and operator."""
            DS_LOGICAL_OR = _pymdlsdk._IFunction_definition_DS_LOGICAL_OR
            r""" The logical or operator."""
            DS_ASSIGN = _pymdlsdk._IFunction_definition_DS_ASSIGN
            r""" The assign operator."""
            DS_MULTIPLY_ASSIGN = _pymdlsdk._IFunction_definition_DS_MULTIPLY_ASSIGN
            r""" The multiplication-assign operator."""
            DS_DIVIDE_ASSIGN = _pymdlsdk._IFunction_definition_DS_DIVIDE_ASSIGN
            r""" The division-assign operator."""
            DS_MODULO_ASSIGN = _pymdlsdk._IFunction_definition_DS_MODULO_ASSIGN
            r""" The modulus-assign operator."""
            DS_PLUS_ASSIGN = _pymdlsdk._IFunction_definition_DS_PLUS_ASSIGN
            r""" The plus-assign operator."""
            DS_MINUS_ASSIGN = _pymdlsdk._IFunction_definition_DS_MINUS_ASSIGN
            r""" The minus-assign operator."""
            DS_SHIFT_LEFT_ASSIGN = _pymdlsdk._IFunction_definition_DS_SHIFT_LEFT_ASSIGN
            r""" The shift-left-assign operator."""
            DS_SHIFT_RIGHT_ASSIGN = _pymdlsdk._IFunction_definition_DS_SHIFT_RIGHT_ASSIGN
            r""" The arithmetic shift-right-assign operator."""
            DS_UNSIGNED_SHIFT_RIGHT_ASSIGN = _pymdlsdk._IFunction_definition_DS_UNSIGNED_SHIFT_RIGHT_ASSIGN
            r""" The unsigned shift-right-assign operator."""
            DS_BITWISE_OR_ASSIGN = _pymdlsdk._IFunction_definition_DS_BITWISE_OR_ASSIGN
            r""" The bitwise or-assign operator."""
            DS_BITWISE_XOR_ASSIGN = _pymdlsdk._IFunction_definition_DS_BITWISE_XOR_ASSIGN
            r""" The bitwise xor-assign operator."""
            DS_BITWISE_AND_ASSIGN = _pymdlsdk._IFunction_definition_DS_BITWISE_AND_ASSIGN
            r""" The bitwise and-assign operator."""
            DS_BINARY_LAST = _pymdlsdk._IFunction_definition_DS_BINARY_LAST
            DS_TERNARY = _pymdlsdk._IFunction_definition_DS_TERNARY
            r""" The ternary operator (conditional). See 'mi_neuray_mdl_ternary_operator'."""
            DS_OPERATOR_LAST = _pymdlsdk._IFunction_definition_DS_OPERATOR_LAST
            DS_INTRINSIC_MATH_ABS = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_ABS
            r""" The %math::abs() intrinsic function."""
            DS_INTRINSIC_MATH_FIRST = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_FIRST
            DS_INTRINSIC_MATH_ACOS = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_ACOS
            r""" The %math::acos() intrinsic function."""
            DS_INTRINSIC_MATH_ALL = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_ALL
            r""" The %math::all() intrinsic function."""
            DS_INTRINSIC_MATH_ANY = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_ANY
            r""" The %math::any() intrinsic function."""
            DS_INTRINSIC_MATH_ASIN = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_ASIN
            r""" The %math::asin() intrinsic function."""
            DS_INTRINSIC_MATH_ATAN = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_ATAN
            r""" The %math::atan() intrinsic function."""
            DS_INTRINSIC_MATH_ATAN2 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_ATAN2
            r""" The %math::atan2() intrinsic function."""
            DS_INTRINSIC_MATH_AVERAGE = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_AVERAGE
            r""" The %math::average() intrinsic function."""
            DS_INTRINSIC_MATH_CEIL = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_CEIL
            r""" The %math::ceil() intrinsic function."""
            DS_INTRINSIC_MATH_CLAMP = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_CLAMP
            r""" The %math::clamp() intrinsic function."""
            DS_INTRINSIC_MATH_COS = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_COS
            r""" The %math::cos() intrinsic function."""
            DS_INTRINSIC_MATH_CROSS = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_CROSS
            r""" The %math::cross() intrinsic function."""
            DS_INTRINSIC_MATH_DEGREES = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_DEGREES
            r""" The %math::degrees() intrinsic function."""
            DS_INTRINSIC_MATH_DISTANCE = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_DISTANCE
            r""" The %math::distance() intrinsic function."""
            DS_INTRINSIC_MATH_DOT = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_DOT
            r""" The %math::dot() intrinsic function."""
            DS_INTRINSIC_MATH_EVAL_AT_WAVELENGTH = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_EVAL_AT_WAVELENGTH
            r""" The %math::eval_at_wavelength() intrinsic"""
            DS_INTRINSIC_MATH_EXP = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_EXP
            r"""  function. The %math::exp() intrinsic function."""
            DS_INTRINSIC_MATH_EXP2 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_EXP2
            r""" The %math::exp2() intrinsic function."""
            DS_INTRINSIC_MATH_FLOOR = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_FLOOR
            r""" The %math::floor() intrinsic function."""
            DS_INTRINSIC_MATH_FMOD = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_FMOD
            r""" The %math::fmod() intrinsic function."""
            DS_INTRINSIC_MATH_FRAC = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_FRAC
            r""" The %math::frac() intrinsic function."""
            DS_INTRINSIC_MATH_ISNAN = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_ISNAN
            r""" The %math::isnan() intrinsic function."""
            DS_INTRINSIC_MATH_ISFINITE = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_ISFINITE
            r""" The %math::isfinite() intrinsic function."""
            DS_INTRINSIC_MATH_LENGTH = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_LENGTH
            r""" The %math::length() intrinsic function."""
            DS_INTRINSIC_MATH_LERP = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_LERP
            r""" The %math::lerp() intrinsic function."""
            DS_INTRINSIC_MATH_LOG = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_LOG
            r""" The %math::log() intrinsic function."""
            DS_INTRINSIC_MATH_LOG2 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_LOG2
            r""" The %math::log2() intrinsic function."""
            DS_INTRINSIC_MATH_LOG10 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_LOG10
            r""" The %math::log10() intrinsic function."""
            DS_INTRINSIC_MATH_LUMINANCE = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_LUMINANCE
            r""" The %math::luminance() intrinsic function."""
            DS_INTRINSIC_MATH_MAX = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_MAX
            r""" The %math::max() intrinsic function."""
            DS_INTRINSIC_MATH_MAX_VALUE = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_MAX_VALUE
            r""" The %math::max_value() intrinsic function."""
            DS_INTRINSIC_MATH_MAX_VALUE_WAVELENGTH = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_MAX_VALUE_WAVELENGTH
            r""" The %math::max_value_wavelength() intrinsic"""
            DS_INTRINSIC_MATH_MIN = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_MIN
            r"""  function. The %math::min() intrinsic function."""
            DS_INTRINSIC_MATH_MIN_VALUE = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_MIN_VALUE
            r""" The %math::min_value() intrinsic function."""
            DS_INTRINSIC_MATH_MIN_VALUE_WAVELENGTH = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_MIN_VALUE_WAVELENGTH
            r""" The %math::min_value_wavelength() intrinsic"""
            DS_INTRINSIC_MATH_MODF = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_MODF
            r"""  function. The %math::modf() intrinsic function."""
            DS_INTRINSIC_MATH_NORMALIZE = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_NORMALIZE
            r""" The %math::normalize() intrinsic function."""
            DS_INTRINSIC_MATH_POW = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_POW
            r""" The %math::pow() intrinsic function."""
            DS_INTRINSIC_MATH_RADIANS = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_RADIANS
            r""" The %math::radians() intrinsic function."""
            DS_INTRINSIC_MATH_ROUND = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_ROUND
            r""" The %math::round() intrinsic function."""
            DS_INTRINSIC_MATH_RSQRT = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_RSQRT
            r""" The %math::rsqrt() intrinsic function."""
            DS_INTRINSIC_MATH_SATURATE = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_SATURATE
            r""" The %math::saturate() intrinsic function."""
            DS_INTRINSIC_MATH_SIGN = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_SIGN
            r""" The %math::sign() intrinsic function."""
            DS_INTRINSIC_MATH_SIN = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_SIN
            r""" The %math::sin() intrinsic function."""
            DS_INTRINSIC_MATH_SINCOS = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_SINCOS
            r""" The %math::sincos() intrinsic function."""
            DS_INTRINSIC_MATH_SMOOTHSTEP = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_SMOOTHSTEP
            r""" The %math::smoothstep() intrinsic function."""
            DS_INTRINSIC_MATH_SQRT = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_SQRT
            r""" The %math::sqrt() intrinsic function."""
            DS_INTRINSIC_MATH_STEP = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_STEP
            r""" The %math::step() intrinsic function."""
            DS_INTRINSIC_MATH_TAN = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_TAN
            r""" The %math::tan() intrinsic function."""
            DS_INTRINSIC_MATH_TRANSPOSE = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_TRANSPOSE
            r""" The %math::transpose() intrinsic function."""
            DS_INTRINSIC_MATH_BLACKBODY = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_BLACKBODY
            r""" The %math::blackbody() intrinsic function."""
            DS_INTRINSIC_MATH_EMISSION_COLOR = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_EMISSION_COLOR
            r""" The %math::emission_color() intrinsic"""
            DS_INTRINSIC_MATH_COSH = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_COSH
            r"""  function. The %math::cosh() intrinsic function."""
            DS_INTRINSIC_MATH_SINH = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_SINH
            r""" The %math::sinh() intrinsic function."""
            DS_INTRINSIC_MATH_TANH = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_TANH
            r""" The %math::tanh() intrinsic function."""
            DS_INTRINSIC_MATH_INT_BITS_TO_FLOAT = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_INT_BITS_TO_FLOAT
            r""" The %math::int_bits_to_float() intrinsic"""
            DS_INTRINSIC_MATH_FLOAT_BITS_TO_INT = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_FLOAT_BITS_TO_INT
            r"""  function. The %math::float_bits_to_int() intrinsic"""
            DS_INTRINSIC_MATH_DX = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_DX
            r"""  function. The %math::DX() intrinsic function."""
            DS_INTRINSIC_MATH_DY = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_DY
            r""" The %math::DY() intrinsic function."""
            DS_INTRINSIC_MATH_LAST = _pymdlsdk._IFunction_definition_DS_INTRINSIC_MATH_LAST
            DS_INTRINSIC_STATE_POSITION = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_POSITION
            r"""  The %state::position() function."""
            DS_INTRINSIC_STATE_FIRST = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_FIRST
            DS_INTRINSIC_STATE_NORMAL = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_NORMAL
            r""" The %state::normal() function."""
            DS_INTRINSIC_STATE_GEOMETRY_NORMAL = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_GEOMETRY_NORMAL
            r""" The %state::geometry_normal() function."""
            DS_INTRINSIC_STATE_MOTION = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_MOTION
            r""" The %state::motion() function."""
            DS_INTRINSIC_STATE_TEXTURE_SPACE_MAX = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_TEXTURE_SPACE_MAX
            r""" The %state::texture_space_max() function."""
            DS_INTRINSIC_STATE_TEXTURE_COORDINATE = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_TEXTURE_COORDINATE
            r""" The %state::texture_coordinate() function."""
            DS_INTRINSIC_STATE_TEXTURE_TANGENT_U = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_TEXTURE_TANGENT_U
            r""" The %state::texture_tangent_u() function."""
            DS_INTRINSIC_STATE_TEXTURE_TANGENT_V = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_TEXTURE_TANGENT_V
            r""" The %state::texture_tangent_v() function."""
            DS_INTRINSIC_STATE_TANGENT_SPACE = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_TANGENT_SPACE
            r""" The %state::tangent_space() function."""
            DS_INTRINSIC_STATE_GEOMETRY_TANGENT_U = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_GEOMETRY_TANGENT_U
            r""" The %state::geometry_tangent_u() function."""
            DS_INTRINSIC_STATE_GEOMETRY_TANGENT_V = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_GEOMETRY_TANGENT_V
            r""" The %state::geometry_tangent_v() function."""
            DS_INTRINSIC_STATE_DIRECTION = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_DIRECTION
            r""" The %state::direction() function."""
            DS_INTRINSIC_STATE_ANIMATION_TIME = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_ANIMATION_TIME
            r""" The %state::animation_time() function."""
            DS_INTRINSIC_STATE_WAVELENGTH_BASE = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_WAVELENGTH_BASE
            r""" The %state::wavelength_base() function."""
            DS_INTRINSIC_STATE_TRANSFORM = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_TRANSFORM
            r""" The %state::transform() function."""
            DS_INTRINSIC_STATE_TRANSFORM_POINT = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_TRANSFORM_POINT
            r""" The %state::transform_point() function."""
            DS_INTRINSIC_STATE_TRANSFORM_VECTOR = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_TRANSFORM_VECTOR
            r""" The %state::transform_vector() function."""
            DS_INTRINSIC_STATE_TRANSFORM_NORMAL = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_TRANSFORM_NORMAL
            r""" The %state::transform_normal() function."""
            DS_INTRINSIC_STATE_TRANSFORM_SCALE = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_TRANSFORM_SCALE
            r""" The %state::transform_scale() function."""
            DS_INTRINSIC_STATE_ROUNDED_CORNER_NORMAL = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_ROUNDED_CORNER_NORMAL
            r""" The %state::rounded_corner_normal() function."""
            DS_INTRINSIC_STATE_METERS_PER_SCENE_UNIT = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_METERS_PER_SCENE_UNIT
            r""" The %state::meters_per_scene_unit() function."""
            DS_INTRINSIC_STATE_SCENE_UNITS_PER_METER = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_SCENE_UNITS_PER_METER
            r""" The %state::scene_units_per_meter() function."""
            DS_INTRINSIC_STATE_OBJECT_ID = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_OBJECT_ID
            r""" The %state::object_id() function."""
            DS_INTRINSIC_STATE_WAVELENGTH_MIN = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_WAVELENGTH_MIN
            r""" The %state::wavelength_min() function."""
            DS_INTRINSIC_STATE_WAVELENGTH_MAX = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_WAVELENGTH_MAX
            r""" The %state::wavelength_max() function."""
            DS_INTRINSIC_STATE_LAST = _pymdlsdk._IFunction_definition_DS_INTRINSIC_STATE_LAST
            DS_INTRINSIC_TEX_WIDTH = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_WIDTH
            r""" The tex::width() function."""
            DS_INTRINSIC_TEX_FIRST = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_FIRST
            DS_INTRINSIC_TEX_HEIGHT = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_HEIGHT
            r""" The tex::height() function."""
            DS_INTRINSIC_TEX_DEPTH = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_DEPTH
            r""" The tex::depth() function."""
            DS_INTRINSIC_TEX_LOOKUP_FLOAT = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_LOOKUP_FLOAT
            r""" The tex::lookup_float() function."""
            DS_INTRINSIC_TEX_LOOKUP_FLOAT2 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_LOOKUP_FLOAT2
            r""" The tex::lookup_float2() function."""
            DS_INTRINSIC_TEX_LOOKUP_FLOAT3 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_LOOKUP_FLOAT3
            r""" The tex::lookup_float3() function."""
            DS_INTRINSIC_TEX_LOOKUP_FLOAT4 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_LOOKUP_FLOAT4
            r""" The tex::lookup_float4() function."""
            DS_INTRINSIC_TEX_LOOKUP_COLOR = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_LOOKUP_COLOR
            r""" The tex::lookup_color() function."""
            DS_INTRINSIC_TEX_TEXEL_FLOAT = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_TEXEL_FLOAT
            r""" The tex::texel_float() function."""
            DS_INTRINSIC_TEX_TEXEL_FLOAT2 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_TEXEL_FLOAT2
            r""" The tex::texel_float2() function."""
            DS_INTRINSIC_TEX_TEXEL_FLOAT3 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_TEXEL_FLOAT3
            r""" The tex::texel_float3() function."""
            DS_INTRINSIC_TEX_TEXEL_FLOAT4 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_TEXEL_FLOAT4
            r""" The tex::texel_float4() function."""
            DS_INTRINSIC_TEX_TEXEL_COLOR = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_TEXEL_COLOR
            r""" The tex::texel_color() function."""
            DS_INTRINSIC_TEX_TEXTURE_ISVALID = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_TEXTURE_ISVALID
            r""" The tex::texture_isvalid() function."""
            DS_INTRINSIC_TEX_WIDTH_OFFSET = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_WIDTH_OFFSET
            r""" The tex::width_offset() function."""
            DS_INTRINSIC_TEX_HEIGHT_OFFSET = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_HEIGHT_OFFSET
            r""" The tex::height_offset() function."""
            DS_INTRINSIC_TEX_DEPTH_OFFSET = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_DEPTH_OFFSET
            r""" The tex::depth_offset() function."""
            DS_INTRINSIC_TEX_FIRST_FRAME = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_FIRST_FRAME
            r""" The tex::first_frame() function."""
            DS_INTRINSIC_TEX_LAST_FRAME = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_LAST_FRAME
            r""" The tex::last_frame() function."""
            DS_INTRINSIC_TEX_GRID_TO_OBJECT_SPACE = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_GRID_TO_OBJECT_SPACE
            r""" The tex::grid_to_object_space() function."""
            DS_INTRINSIC_TEX_LAST = _pymdlsdk._IFunction_definition_DS_INTRINSIC_TEX_LAST
            DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF
            r""" The df::diffuse_reflection_bsdf() function."""
            DS_INTRINSIC_DF_FIRST = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_FIRST
            DS_INTRINSIC_DF_DUSTY_DIFFUSE_REFLECTION_BSDF = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_DUSTY_DIFFUSE_REFLECTION_BSDF
            r""" The df::dusty_diffuse_reflection_bsdf() function."""
            DS_INTRINSIC_DF_DIFFUSE_TRANSMISSION_BSDF = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_DIFFUSE_TRANSMISSION_BSDF
            r""" The df::diffuse_transmission_bsdf() function."""
            DS_INTRINSIC_DF_SPECULAR_BSDF = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_SPECULAR_BSDF
            r""" The df::specular_bsdf() function."""
            DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF
            r""" The df::simple_glossy_bsdf() function."""
            DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF
            r""" The df::backscattering_glossy_reflection_bsdf() function."""
            DS_INTRINSIC_DF_MEASURED_BSDF = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_MEASURED_BSDF
            r""" The df::measured_bsdf() function."""
            DS_INTRINSIC_DF_DIFFUSE_EDF = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_DIFFUSE_EDF
            r""" The df::diffuse_edf() function."""
            DS_INTRINSIC_DF_MEASURED_EDF = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_MEASURED_EDF
            r""" The df::measured_edf() function."""
            DS_INTRINSIC_DF_SPOT_EDF = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_SPOT_EDF
            r""" The df::spot_edf() function."""
            DS_INTRINSIC_DF_ANISOTROPIC_VDF = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_ANISOTROPIC_VDF
            r""" The df::anisotropic_vdf() function."""
            DS_INTRINSIC_DF_FOG_VDF = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_FOG_VDF
            r""" The df::fog_vdf() function."""
            DS_INTRINSIC_DF_NORMALIZED_MIX = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_NORMALIZED_MIX
            r""" The df::normalized_mix() function."""
            DS_INTRINSIC_DF_CLAMPED_MIX = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_CLAMPED_MIX
            r""" The df::clamped_mix() function."""
            DS_INTRINSIC_DF_WEIGHTED_LAYER = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_WEIGHTED_LAYER
            r""" The df::weighted_layer() function."""
            DS_INTRINSIC_DF_FRESNEL_LAYER = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_FRESNEL_LAYER
            r""" The df::fresnel_layer() function."""
            DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_CUSTOM_CURVE_LAYER
            r""" The df::custom_curve_layer() function."""
            DS_INTRINSIC_DF_MEASURED_CURVE_LAYER = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_MEASURED_CURVE_LAYER
            r""" The df::measured_curve_layer() function."""
            DS_INTRINSIC_DF_THIN_FILM = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_THIN_FILM
            r""" The df::thin_film() function."""
            DS_INTRINSIC_DF_TINT = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_TINT
            r""" The df::tint() function."""
            DS_INTRINSIC_DF_DIRECTIONAL_FACTOR = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_DIRECTIONAL_FACTOR
            r""" The df::directional_factor() function."""
            DS_INTRINSIC_DF_MEASURED_CURVE_FACTOR = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_MEASURED_CURVE_FACTOR
            r""" The df::measured_curve_factor() function."""
            DS_INTRINSIC_DF_LIGHT_PROFILE_POWER = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_LIGHT_PROFILE_POWER
            r""" The df::light_profile_power() function."""
            DS_INTRINSIC_DF_LIGHT_PROFILE_MAXIMUM = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_LIGHT_PROFILE_MAXIMUM
            r""" The df::light_profile_maximum() function."""
            DS_INTRINSIC_DF_LIGHT_PROFILE_ISVALID = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_LIGHT_PROFILE_ISVALID
            r""" The df::light_profile_isvalid() function."""
            DS_INTRINSIC_DF_BSDF_MEASUREMENT_ISVALID = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_BSDF_MEASUREMENT_ISVALID
            r""" The df::bsdf_measurement_is_valid() function."""
            DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF
            r""" The df::microfacet_beckmann_smith_bsdf() function."""
            DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF
            r""" The df::microfacet_ggx_smith_bsdf() function."""
            DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF
            r""" The df::microfacet_beckmann_vcavities() function."""
            DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF
            r""" The df::microfacet_ggx_vcavities() function."""
            DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF
            r""" The df::ward_geisler_moroder_bsdf() function."""
            DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_COLOR_NORMALIZED_MIX
            r""" The df::color_normalized_mix() function."""
            DS_INTRINSIC_DF_COLOR_CLAMPED_MIX = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_COLOR_CLAMPED_MIX
            r""" The df::color_clamped_mix() function."""
            DS_INTRINSIC_DF_COLOR_WEIGHTED_LAYER = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_COLOR_WEIGHTED_LAYER
            r""" The df::color_weighted_layer() function."""
            DS_INTRINSIC_DF_COLOR_FRESNEL_LAYER = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_COLOR_FRESNEL_LAYER
            r""" The df::color_fresnel_layer() function."""
            DS_INTRINSIC_DF_COLOR_CUSTOM_CURVE_LAYER = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_COLOR_CUSTOM_CURVE_LAYER
            r""" The df::color_custom_curve_layer() function."""
            DS_INTRINSIC_DF_COLOR_MEASURED_CURVE_LAYER = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_COLOR_MEASURED_CURVE_LAYER
            r""" The df::color_measured_curve_layer() function."""
            DS_INTRINSIC_DF_FRESNEL_FACTOR = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_FRESNEL_FACTOR
            r""" The df::fresnel_factor() function."""
            DS_INTRINSIC_DF_MEASURED_FACTOR = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_MEASURED_FACTOR
            r""" The df::measured_factor() function."""
            DS_INTRINSIC_DF_CHIANG_HAIR_BSDF = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_CHIANG_HAIR_BSDF
            r""" The df::chiang_hair_bsdf() function."""
            DS_INTRINSIC_DF_SHEEN_BSDF = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_SHEEN_BSDF
            r""" The df::sheen_bsdf() function."""
            DS_INTRINSIC_DF_UNBOUNDED_MIX = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_UNBOUNDED_MIX
            r""" The df::unbounded_mix() function."""
            DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_COLOR_UNBOUNDED_MIX
            r""" The df::color_unbounded() function."""
            DS_INTRINSIC_DF_LAST = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DF_LAST
            DS_INTRINSIC_SCENE_DATA_ISVALID = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_DATA_ISVALID
            r""" The scene::data_is_valid() function."""
            DS_INTRINSIC_SCENE_FIRST = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_FIRST
            DS_INTRINSIC_SCENE_DATA_LOOKUP_INT = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_DATA_LOOKUP_INT
            r""" The scene::data_lookup_int() function."""
            DS_INTRINSIC_SCENE_DATA_LOOKUP_INT2 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_DATA_LOOKUP_INT2
            r""" The scene::data_lookup_int2() function."""
            DS_INTRINSIC_SCENE_DATA_LOOKUP_INT3 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_DATA_LOOKUP_INT3
            r""" The scene::data_lookup_int3() function."""
            DS_INTRINSIC_SCENE_DATA_LOOKUP_INT4 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_DATA_LOOKUP_INT4
            r""" The scene::data_lookup_int4() function."""
            DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT
            r""" The scene::data_lookup_float() function."""
            DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT2 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT2
            r""" The scene::data_lookup_float2() function."""
            DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT3 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT3
            r""" The scene::data_lookup_float3() function."""
            DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT4 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT4
            r""" The scene::data_lookup_float4() function."""
            DS_INTRINSIC_SCENE_DATA_LOOKUP_COLOR = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_DATA_LOOKUP_COLOR
            r""" The scene::data_lookup_color() function."""
            DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT
            r""" The scene::data_lookup_uniform_int() function."""
            DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT2 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT2
            r""" The scene::data_lookup_uniform_int2() function."""
            DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT3 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT3
            r""" The scene::data_lookup_uniform_int3() function."""
            DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT4 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_INT4
            r""" The scene::data_lookup_uniform_int4() function."""
            DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT
            r""" The scene::data_lookup_uniform_float() function."""
            DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT2 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT2
            r""" The scene::data_lookup_uniform_float2() function."""
            DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT3 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT3
            r""" The scene::data_lookup_uniform_float3() function."""
            DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT4 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT4
            r""" The scene::data_lookup_uniform_float4() function."""
            DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_COLOR = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_COLOR
            r""" The scene::data_lookup_uniform_color() function."""
            DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT4X4 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_DATA_LOOKUP_FLOAT4X4
            r""" The scene::data_lookup_float4x4() function."""
            DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT4X4 = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_DATA_LOOKUP_UNIFORM_FLOAT4X4
            r""" The scene::data_lookup_uniform_float4x4() function."""
            DS_INTRINSIC_SCENE_LAST = _pymdlsdk._IFunction_definition_DS_INTRINSIC_SCENE_LAST
            DS_INTRINSIC_DEBUG_BREAKPOINT = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DEBUG_BREAKPOINT
            r""" The debug::breakpoint() function."""
            DS_INTRINSIC_DEBUG_FIRST = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DEBUG_FIRST
            DS_INTRINSIC_DEBUG_ASSERT = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DEBUG_ASSERT
            r""" The debug::assert() function."""
            DS_INTRINSIC_DEBUG_PRINT = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DEBUG_PRINT
            r""" The debug::print() function."""
            DS_INTRINSIC_DEBUG_LAST = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DEBUG_LAST
            DS_INTRINSIC_DAG_FIELD_ACCESS = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DAG_FIELD_ACCESS
            r""" The structure field access function."""
            DS_INTRINSIC_DAG_FIRST = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DAG_FIRST
            DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR
            r""" The array constructor. See 'mi_neuray_mdl_array_constructor'."""
            DS_INTRINSIC_DAG_ARRAY_LENGTH = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DAG_ARRAY_LENGTH
            r""" The array length operator. See 'mi_neuray_mdl_array_length_operator'."""
            DS_INTRINSIC_DAG_DECL_CAST = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DAG_DECL_CAST
            r"""  The decl_cast operator. See 'mi_neuray_mdl_decl_cast_operator'."""
            DS_INTRINSIC_DAG_LAST = _pymdlsdk._IFunction_definition_DS_INTRINSIC_DAG_LAST
            DS_FORCE_32_BIT = _pymdlsdk._IFunction_definition_DS_FORCE_32_BIT
    }
}

// unwrap all usages of the enum values in arguments
// - NONE-

// Handle all functions that return this enum type
WRAP_ENUM_RETURN(mi::neuraylib, IFunction_definition, get_semantic, IFunction_definition.Semantics)




// special handling for: mi::neuraylib::IMaterial_instance::Compilation_options
// ----------------------------------------------------------------------------

// We manually define the enums in the correct proxy class
%extend SmartPtr<mi::neuraylib::IAnnotation_definition> {
    %pythoncode{

        @post_swig_add_type_hint_mapping("mi::neuraylib::IAnnotation_definition::Semantics", "IAnnotation_definition.Semantics")
        class Semantics(Enum) :
            r"""
            All known semantics of annotation definitions.

            Note, do not rely on the numeric values of the enumerators since they may change without further notice.
            """

            AS_UNKNOWN = _pymdlsdk._IAnnotation_definition_AS_UNKNOWN
            r""" Unknown semantics."""
            AS_INTRINSIC_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_INTRINSIC_ANNOTATION
            r""" This is the internal intrinsic() annotation."""
            AS_ANNOTATION_FIRST = _pymdlsdk._IAnnotation_definition_AS_ANNOTATION_FIRST
            AS_THROWS_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_THROWS_ANNOTATION
            r""" This is the internal throws() annotation."""
            AS_SINCE_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_SINCE_ANNOTATION
            r""" This is the internal since() annotation."""
            AS_REMOVED_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_REMOVED_ANNOTATION
            r""" This is the internal removed() annotation."""
            AS_CONST_EXPR_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_CONST_EXPR_ANNOTATION
            r""" This is the internal const_expr() annotation."""
            AS_DERIVABLE_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_DERIVABLE_ANNOTATION
            r""" This is the internal derivable() annotation."""
            AS_NATIVE_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_NATIVE_ANNOTATION
            r""" This is the internal native() annotation."""
            AS_UNUSED_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_UNUSED_ANNOTATION
            r""" This is the unused() annotation."""
            AS_NOINLINE_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_NOINLINE_ANNOTATION
            r""" This is the noinline() annotation."""
            AS_SOFT_RANGE_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_SOFT_RANGE_ANNOTATION
            r""" This is the soft_range() annotation."""
            AS_HARD_RANGE_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_HARD_RANGE_ANNOTATION
            r""" This is the hard_range() annotation."""
            AS_HIDDEN_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_HIDDEN_ANNOTATION
            r""" This is the hidden() annotation."""
            AS_DEPRECATED_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_DEPRECATED_ANNOTATION
            r""" This is the deprecated() annotation."""
            AS_VERSION_NUMBER_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_VERSION_NUMBER_ANNOTATION
            r""" This is the (old) version_number() annotation."""
            AS_VERSION_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_VERSION_ANNOTATION
            r""" This is the version() annotation."""
            AS_DEPENDENCY_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_DEPENDENCY_ANNOTATION
            r""" This is the dependency() annotation."""
            AS_UI_ORDER_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_UI_ORDER_ANNOTATION
            r""" This is the ui_order() annotation."""
            AS_USAGE_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_USAGE_ANNOTATION
            r""" This is the usage() annotation."""
            AS_ENABLE_IF_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_ENABLE_IF_ANNOTATION
            r""" This is the enable_if() annotation."""
            AS_THUMBNAIL_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_THUMBNAIL_ANNOTATION
            r""" This is the thumbnail() annotation."""
            AS_DISPLAY_NAME_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_DISPLAY_NAME_ANNOTATION
            r""" This is the display_name() annotation."""
            AS_IN_GROUP_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_IN_GROUP_ANNOTATION
            r""" This is the in_group() annotation."""
            AS_DESCRIPTION_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_DESCRIPTION_ANNOTATION
            r""" This is the description() annotation."""
            AS_AUTHOR_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_AUTHOR_ANNOTATION
            r""" This is the author() annotation."""
            AS_CONTRIBUTOR_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_CONTRIBUTOR_ANNOTATION
            r""" This is the contributor() annotation."""
            AS_COPYRIGHT_NOTICE_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_COPYRIGHT_NOTICE_ANNOTATION
            r""" This is the copyright_notice() annotation."""
            AS_CREATED_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_CREATED_ANNOTATION
            r""" This is the created() annotation."""
            AS_MODIFIED_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_MODIFIED_ANNOTATION
            r""" This is the modified() annotation."""
            AS_KEYWORDS_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_KEYWORDS_ANNOTATION
            r""" This is the key_words() annotation."""
            AS_ORIGIN_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_ORIGIN_ANNOTATION
            r""" This is the origin() annotation."""
            AS_NODE_OUTPUT_PORT_DEFAULT_ANNOTATION = _pymdlsdk._IAnnotation_definition_AS_NODE_OUTPUT_PORT_DEFAULT_ANNOTATION
            r""" This is the node_output_port_default()"""
            AS_ANNOTATION_LAST = _pymdlsdk._IAnnotation_definition_AS_ANNOTATION_LAST
            r"""  annotation."""
            AS_FORCE_32_BIT = _pymdlsdk._IAnnotation_definition_AS_FORCE_32_BIT
    }
}

// unwrap all usages of the enum values in arguments
// - NONE-

// Handle all functions that return this enum type
WRAP_ENUM_RETURN(mi::neuraylib, IAnnotation_definition, get_semantic, IAnnotation_definition.Semantics)



// special handling for: mi::neuraylib::IType::Kind
// ----------------------------------------------------------------------------
%extend SmartPtr<mi::neuraylib::IType> {
    %pythoncode{

        @post_swig_add_type_hint_mapping("mi::neuraylib::IType::Kind", "IType.Kind")
        class Kind(Enum):
            r"""The possible kinds of types."""

            TK_ALIAS = _pymdlsdk._IType_TK_ALIAS
            r""" An alias for another type, aka typedef. See #mi::neuraylib::IType_alias."""
            TK_BOOL = _pymdlsdk._IType_TK_BOOL
            r""" The ``boolean`` type. See #mi::neuraylib::IType_bool."""
            TK_INT = _pymdlsdk._IType_TK_INT
            r""" The ``integer`` type. See #mi::neuraylib::IType_int."""
            TK_ENUM = _pymdlsdk._IType_TK_ENUM
            r""" An ``enum`` type. See #mi::neuraylib::IType_enum."""
            TK_FLOAT = _pymdlsdk._IType_TK_FLOAT
            r""" The ``float`` type. See #mi::neuraylib::IType_float."""
            TK_DOUBLE = _pymdlsdk._IType_TK_DOUBLE
            r""" The ``double`` type. See #mi::neuraylib::IType_double."""
            TK_STRING = _pymdlsdk._IType_TK_STRING
            r"""  The ``string`` type. See #mi::neuraylib::IType_string."""
            TK_VECTOR = _pymdlsdk._IType_TK_VECTOR
            r""" A vector type. See #mi::neuraylib::IType_vector."""
            TK_MATRIX = _pymdlsdk._IType_TK_MATRIX
            r""" A matrix type. See #mi::neuraylib::IType_matrix."""
            TK_COLOR = _pymdlsdk._IType_TK_COLOR
            r""" The color type. See #mi::neuraylib::IType_color."""
            TK_ARRAY = _pymdlsdk._IType_TK_ARRAY
            r""" An array type. See #mi::neuraylib::IType_array."""
            TK_STRUCT = _pymdlsdk._IType_TK_STRUCT
            r""" A struct type. See #mi::neuraylib::IType_struct."""
            TK_TEXTURE = _pymdlsdk._IType_TK_TEXTURE
            r""" A texture type. See #mi::neuraylib::IType_texture."""
            TK_LIGHT_PROFILE = _pymdlsdk._IType_TK_LIGHT_PROFILE
            r""" The ``light_profile`` type. See #mi::neuraylib::IType_light_profile."""
            TK_BSDF_MEASUREMENT = _pymdlsdk._IType_TK_BSDF_MEASUREMENT
            r""" The ``bsdf_measurement`` type. See #mi::neuraylib::IType_bsdf_measurement."""
            TK_BSDF = _pymdlsdk._IType_TK_BSDF
            r""" The ``bsdf`` type. See #mi::neuraylib::IType_bsdf."""
            TK_HAIR_BSDF = _pymdlsdk._IType_TK_HAIR_BSDF
            r""" The ``hair_bsdf`` type. See #mi::neuraylib::IType_hair_bsdf."""
            TK_EDF = _pymdlsdk._IType_TK_EDF
            r""" The ``edf`` type. See #mi::neuraylib::IType_edf."""
            TK_VDF = _pymdlsdk._IType_TK_VDF
            r""" The ``vdf`` type. See #mi::neuraylib::IType_vdf."""
            TK_FORCE_32_BIT = _pymdlsdk._IType_TK_FORCE_32_BIT

        @post_swig_add_type_hint_mapping("mi::neuraylib::IType::Modifier", "IType.Modifier")
        class Modifier(Enum):
            r"""The possible kinds of type modifiers."""

            MK_NONE = _pymdlsdk._IType_MK_NONE
            r""" No type modifier (mutable, auto-typed)."""
            MK_UNIFORM = _pymdlsdk._IType_MK_UNIFORM
            r""" A uniform type."""
            MK_VARYING = _pymdlsdk._IType_MK_VARYING
            r""" A varying type."""
            MK_FORCE_32_BIT = _pymdlsdk._IType_MK_FORCE_32_BIT
    }
}

// unwrap all usages of the enum values in arguments
%feature("pythonprepend") mi::neuraylib::IMdl_module_builder::add_function %{
    frequency_qualifier = frequency_qualifier.value  # unwrap python enum and pass the integer value
%}

// Handle all functions that return this enum type
WRAP_ENUM_RETURN(mi::neuraylib, IType, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_alias, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_atomic, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_bool, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_int, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_enumeration, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_float, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_double, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_string, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_compound, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_vector, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_matrix, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_color, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_array, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_structure, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_reference, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_resource, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_texture, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_light_profile, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_bsdf_measurement, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_df, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_bsdf, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_hair_bsdf, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_edf, get_kind, IType.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IType_vdf, get_kind, IType.Kind)

// unwrap all usages of the enum values in arguments
%feature("pythonprepend") mi::neuraylib::IMdl_configuration::set_material_ior_frequency%{
    frequency_qualifier = frequency_qualifier.value  # unwrap python enum and pass the integer value
%}
// Handle all functions that return this enum type
WRAP_ENUM_RETURN(mi::neuraylib, IMdl_configuration, get_material_ior_frequency, IType.Modifier)



// special handling for: mi::neuraylib::IType_texture::Shape
// ----------------------------------------------------------------------------

%extend SmartPtr<mi::neuraylib::IType_texture> {
    %pythoncode{

        @post_swig_add_type_hint_mapping("mi::neuraylib::IType_texture::Shape", "IType_texture.Shape")
        class Shape(Enum):
            r"""The possible texture shapes."""

            TS_2D = _pymdlsdk._IType_texture_TS_2D
            r""" Two-dimensional texture."""
            TS_3D = _pymdlsdk._IType_texture_TS_3D
            r""" Three-dimensional texture."""
            TS_CUBE = _pymdlsdk._IType_texture_TS_CUBE
            r""" Cube map texture."""
            TS_PTEX = _pymdlsdk._IType_texture_TS_PTEX
            r""" PTEX texture."""
            TS_BSDF_DATA = _pymdlsdk._IType_texture_TS_BSDF_DATA
            r""" Three-dimensional texture representing a BSDF data table."""
            TS_FORCE_32_BIT = _pymdlsdk._IType_texture_TS_FORCE_32_BIT
    }
}

// unwrap all usages of the enum values
%feature("pythonprepend") mi::neuraylib::IType_factory::create_texture %{
    shape = shape.value  # unwrap python enum and pass the integer value
%}
%feature("pythonprepend") mi::neuraylib::IMdl_factory::create_texture %{
    shape = shape.value  # unwrap python enum and pass the integer value
%}

// Handle all functions that return this enum type
WRAP_ENUM_RETURN(mi::neuraylib, IType_texture, get_shape, IType_texture.Shape)



// special handling for: mi::neuraylib::IStruct_category::Predefined_id
// ----------------------------------------------------------------------------

%extend SmartPtr<mi::neuraylib::IStruct_category> {
    %pythoncode{

        @post_swig_add_type_hint_mapping("mi::neuraylib::IStruct_category::Predefined_id", "IStruct_category.Predefined_id")
        class Predefined_id(Enum) :
            r"""Identifiers of struct categories."""

            CID_USER = _pymdlsdk._IStruct_category_CID_USER
            r"""A user-defined struct category."""
            CID_MATERIAL_CATEGORY = _pymdlsdk._IStruct_category_CID_MATERIAL_CATEGORY
            r"""The ``"::material_category"`` struct category."""
            CID_FORCE_32_BIT = _pymdlsdk._IStruct_category_CID_FORCE_32_BIT
    }
}

// unwrap all usages of the predefined struct categories
%feature("pythonprepend") mi::neuraylib::IType_factory::get_predefined_struct_category %{
    id = id.value  # unwrap python enum and pass the integer value
%}

// Handle all functions that return this struct category
WRAP_ENUM_RETURN(mi::neuraylib, IStruct_category, get_predefined_id, IStruct_category.Predefined_id)



// special handling for: mi::neuraylib::IType_enum::Predefined_id
// ----------------------------------------------------------------------------

%extend SmartPtr<mi::neuraylib::IType_enumeration> {
    %pythoncode{

        @post_swig_add_type_hint_mapping("mi::neuraylib::IType_enumeration::Predefined_id", "IType_enumeration.Predefined_id")
        class Predefined_id(Enum):
            r"""TIDs to distinguish predefined enum types."""

            EID_USER = _pymdlsdk._IType_enum_EID_USER
            r""" A user-defined enum type."""
            EID_TEX_GAMMA_MODE = _pymdlsdk._IType_enum_EID_TEX_GAMMA_MODE
            r""" The ``"::tex::gamma_mode"`` enum type."""
            EID_INTENSITY_MODE = _pymdlsdk._IType_enum_EID_INTENSITY_MODE
            r""" The ``"::intensity_mode"`` enum type."""
            EID_FORCE_32_BIT = _pymdlsdk._IType_enum_EID_FORCE_32_BIT
    }
}

// unwrap all usages of the enum values
%feature("pythonprepend") mi::neuraylib::IType_factory::get_predefined_enum %{
    id = id.value  # unwrap python enum and pass the integer value
%}

// Handle all functions that return this enum type
WRAP_ENUM_RETURN(mi::neuraylib, IType_enumeration, get_predefined_id, IType_enumeration.Predefined_id)




// special handling for: mi::neuraylib::IType_struct::Predefined_id
// ----------------------------------------------------------------------------

%extend SmartPtr<mi::neuraylib::IType_structure> {
    %pythoncode{

        @post_swig_add_type_hint_mapping("mi::neuraylib::IType_structure::Predefined_id", "IType_structure.Predefined_id")
        class Predefined_id(Enum) :
            r"""TIDs to distinguish predefined struct types."""

            SID_USER = _pymdlsdk._IType_struct_SID_USER
            r""" A user-defined struct type."""
            SID_MATERIAL_EMISSION = _pymdlsdk._IType_struct_SID_MATERIAL_EMISSION
            r""" The ``"::material_emission"`` struct type."""
            SID_MATERIAL_SURFACE = _pymdlsdk._IType_struct_SID_MATERIAL_SURFACE
            r""" The ``"::material_surface"`` struct type."""
            SID_MATERIAL_VOLUME = _pymdlsdk._IType_struct_SID_MATERIAL_VOLUME
            r""" The ``"::material_volume"`` struct type."""
            SID_MATERIAL_GEOMETRY = _pymdlsdk._IType_struct_SID_MATERIAL_GEOMETRY
            r""" The ``"::material_geometry"`` struct type."""
            SID_MATERIAL = _pymdlsdk._IType_struct_SID_MATERIAL
            r""" The ``"::material"`` struct type."""
            SID_FORCE_32_BIT = _pymdlsdk._IType_struct_SID_FORCE_32_BIT
    }
}

// unwrap all usages of the enum values
%feature("pythonprepend") mi::neuraylib::IType_factory::get_predefined_struct %{
    id = id.value  # unwrap python enum and pass the integer value
%}

// Handle all functions that return this enum type
WRAP_ENUM_RETURN(mi::neuraylib, IType_structure, get_predefined_id, IType_structure.Predefined_id)




// special handling for: mi::neuraylib::IValue::Kind
// ----------------------------------------------------------------------------
%extend SmartPtr<mi::neuraylib::IValue> {
    %pythoncode{

        @post_swig_add_type_hint_mapping("mi::neuraylib::IValue::Kind", "IValue.Kind")
        class Kind(Enum) :
            r"""The possible kinds of values."""

            VK_BOOL = _pymdlsdk._IValue_VK_BOOL
            r""" A boolean value. See #mi::neuraylib::IValue_bool."""
            VK_INT = _pymdlsdk._IValue_VK_INT
            r""" An integer value. See #mi::neuraylib::IValue_int."""
            VK_ENUM = _pymdlsdk._IValue_VK_ENUM
            r""" An enum value. See #mi::neuraylib::IValue_enum."""
            VK_FLOAT = _pymdlsdk._IValue_VK_FLOAT
            r""" A float value. See #mi::neuraylib::IValue_float."""
            VK_DOUBLE = _pymdlsdk._IValue_VK_DOUBLE
            r""" A double value. See #mi::neuraylib::IValue_double."""
            VK_STRING = _pymdlsdk._IValue_VK_STRING
            r""" A string value. See #mi::neuraylib::IValue_string."""
            VK_VECTOR = _pymdlsdk._IValue_VK_VECTOR
            r""" A vector value. See #mi::neuraylib::IValue_vector."""
            VK_MATRIX = _pymdlsdk._IValue_VK_MATRIX
            r""" A matrix value. See #mi::neuraylib::IValue_matrix."""
            VK_COLOR = _pymdlsdk._IValue_VK_COLOR
            r""" A color value. See #mi::neuraylib::IValue_color."""
            VK_ARRAY = _pymdlsdk._IValue_VK_ARRAY
            r""" An array value. See #mi::neuraylib::IValue_array."""
            VK_STRUCT = _pymdlsdk._IValue_VK_STRUCT
            r""" A struct value. See #mi::neuraylib::IValue_struct."""
            VK_INVALID_DF = _pymdlsdk._IValue_VK_INVALID_DF
            r""" An invalid distribution function value. See #mi::neuraylib::IValue_invalid_df."""
            VK_TEXTURE = _pymdlsdk._IValue_VK_TEXTURE
            r""" A texture value. See #mi::neuraylib::IValue_texture."""
            VK_LIGHT_PROFILE = _pymdlsdk._IValue_VK_LIGHT_PROFILE
            r""" A light_profile value. See #mi::neuraylib::IValue_light_profile."""
            VK_BSDF_MEASUREMENT = _pymdlsdk._IValue_VK_BSDF_MEASUREMENT
            r""" A bsdf_measurement value. See #mi::neuraylib::IValue_bsdf_measurement."""
            VK_FORCE_32_BIT = _pymdlsdk._IValue_VK_FORCE_32_BIT
    }
}

// unwrap all usages of the enum values
// - NONE-

// Handle all functions that return this enum type
WRAP_ENUM_RETURN(mi::neuraylib, IValue, get_kind, IValue.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IValue_bool, get_kind, IValue.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IValue_int, get_kind, IValue.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IValue_enumeration, get_kind, IValue.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IValue_float, get_kind, IValue.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IValue_double, get_kind, IValue.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IValue_string, get_kind, IValue.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IValue_vector, get_kind, IValue.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IValue_matrix, get_kind, IValue.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IValue_color, get_kind, IValue.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IValue_array, get_kind, IValue.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IValue_structure, get_kind, IValue.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IValue_invalid_df, get_kind, IValue.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IValue_texture, get_kind, IValue.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IValue_light_profile, get_kind, IValue.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IValue_bsdf_measurement, get_kind, IValue.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IValue_atomic, get_kind, IValue.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IValue_string_localized, get_kind, IValue.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IValue_compound, get_kind, IValue.Kind)
WRAP_ENUM_RETURN(mi::neuraylib, IValue_resource, get_kind, IValue.Kind)




// special handling for: mi::neuraylib::Lightprofile_flags
// ----------------------------------------------------------------------------

%pythoncode{

    @post_swig_add_type_hint_mapping("mi::neuraylib::Lightprofile_flags", "Lightprofile_flags")
    class Lightprofile_flags(Enum):
        r"""
        Ordering of horizontal angles in a light profile

        The flags can be used to override the horizontal sample order in an IES file
        [IES02]. There are two IES file types in common use, type B and type C. The IES
        standard defines that samples are stored in counter-clockwise order. Type C files conform
        to this standard, but about 30% of the type B files deviate from the standard and store
        samples in clockwise order, without giving any indication in the IES file that could be
        used to switch the order. (Sometimes there is an informal comment.) Type A IES files are
        not supported.

        See #mi::neuraylib::ILightprofile::reset_file(), #mi::neuraylib::ILightprofile::get_flags().
        """

        LIGHTPROFILE_CLOCKWISE = _pymdlsdk.LIGHTPROFILE_CLOCKWISE
        r""" Clockwise order, contrary to the IES standard for these (incorrect) type B files."""
        LIGHTPROFILE_COUNTER_CLOCKWISE = _pymdlsdk.LIGHTPROFILE_COUNTER_CLOCKWISE
        r""" Counter-clockwise, standard-conforming order (default)."""
        LIGHTPROFILE_ROTATE_TYPE_B = _pymdlsdk.LIGHTPROFILE_ROTATE_TYPE_B
        r""" For 3dsmax"""
        LIGHTPROFILE_ROTATE_TYPE_C_90_270 = _pymdlsdk.LIGHTPROFILE_ROTATE_TYPE_C_90_270
        r""" For 3dsmax"""
        LIGHTPROFILE_FLAGS_FORCE_32_BIT = _pymdlsdk.LIGHTPROFILE_FLAGS_FORCE_32_BIT
}

// unwrap all usages of the enum values
// - NONE-

// Handle all functions that return this enum type
WRAP_ENUM_RETURN(mi::neuraylib, ILightprofile, get_flags, Lightprofile_flags)




// special handling for: mi::neuraylib::Lightprofile_degree
// ----------------------------------------------------------------------------

%pythoncode{

    @post_swig_add_type_hint_mapping("mi::neuraylib::Lightprofile_degree", "Lightprofile_degree")
    class Lightprofile_degree(Enum):
        r"""
        Degree of hermite interpolation.

        Currently only linear (hermite 1) and cubic (hermite 3) degree are supported
        (see also [DH05]).

        See #mi::neuraylib::ILightprofile::reset_file(), #mi::neuraylib::ILightprofile::get_degree().
        """

        LIGHTPROFILE_HERMITE_BASE_1 = _pymdlsdk.LIGHTPROFILE_HERMITE_BASE_1
        r""" Degree 1 = linear interpolation"""
        LIGHTPROFILE_HERMITE_BASE_3 = _pymdlsdk.LIGHTPROFILE_HERMITE_BASE_3
        r""" Degree 3 = cubic interpolation"""
        LIGHTPROFILE_DEGREE_FORCE_32_BIT = _pymdlsdk.LIGHTPROFILE_DEGREE_FORCE_32_BIT
}

// unwrap all usages of the enum values
%feature("pythonprepend") mi::neuraylib::ILightprofile::reset_file %{
    argc: int = len(args)
    if argc == 5:
        args = (args[0], args[1], args[2], args[3].value, args[4].value)
    elif argc == 4:
        args = (args[0], args[1], args[2], args[3].value, Lightprofile_flags.LIGHTPROFILE_COUNTER_CLOCKWISE.value)
    elif argc == 3:
        args = (args[0], args[1], args[2], Lightprofile_degree.LIGHTPROFILE_HERMITE_BASE_1.value, Lightprofile_flags.LIGHTPROFILE_COUNTER_CLOCKWISE.value)
    elif argc == 2:
        args = (args[0], args[1], 0, Lightprofile_degree.LIGHTPROFILE_HERMITE_BASE_1.value, Lightprofile_flags.LIGHTPROFILE_COUNTER_CLOCKWISE.value)
    elif argc == 1:
        args = (args[0], 0, 0, Lightprofile_degree.LIGHTPROFILE_HERMITE_BASE_1.value, Lightprofile_flags.LIGHTPROFILE_COUNTER_CLOCKWISE.value)
%}
%feature("pythonprepend") mi::neuraylib::ILightprofile::reset_reader %{
    argc: int = len(args)
    if argc == 5:
        args = (args[0], args[1], args[2], args[3].value, args[4].value)
    elif argc == 4:
        args = (args[0], args[1], args[2], args[3].value, Lightprofile_flags.LIGHTPROFILE_COUNTER_CLOCKWISE.value)
    elif argc == 3:
        args = (args[0], args[1], args[2], Lightprofile_degree.LIGHTPROFILE_HERMITE_BASE_1.value, Lightprofile_flags.LIGHTPROFILE_COUNTER_CLOCKWISE.value)
    elif argc == 2:
        args = (args[0], args[1], 0, Lightprofile_degree.LIGHTPROFILE_HERMITE_BASE_1.value, Lightprofile_flags.LIGHTPROFILE_COUNTER_CLOCKWISE.value)
    elif argc == 1:
        args = (args[0], 0, 0, Lightprofile_degree.LIGHTPROFILE_HERMITE_BASE_1.value, Lightprofile_flags.LIGHTPROFILE_COUNTER_CLOCKWISE.value)
%}

// Handle all functions that return this enum type
WRAP_ENUM_RETURN(mi::neuraylib, ILightprofile, get_degree, Lightprofile_degree)




// special handling for: mi::neuraylib::Texture_compression
// ----------------------------------------------------------------------------

%pythoncode{

    @post_swig_add_type_hint_mapping("mi::neuraylib::Texture_compression", "Texture_compression")
    class Texture_compression(Enum):
        r"""Texture compression method."""

        TEXTURE_NO_COMPRESSION = _pymdlsdk.TEXTURE_NO_COMPRESSION
        r""" no compression"""
        TEXTURE_MEDIUM_COMPRESSION = _pymdlsdk.TEXTURE_MEDIUM_COMPRESSION
        r""" medium compression ratio"""
        TEXTURE_HIGH_COMPRESSION = _pymdlsdk.TEXTURE_HIGH_COMPRESSION
        r""" high compression ratio"""
        TEXTURE_COMPRESSION_FORCE_32_BIT = _pymdlsdk.TEXTURE_COMPRESSION_FORCE_32_BIT
}

// unwrap all usages of the enum values
%feature("pythonprepend") mi::neuraylib::ITexture::set_compression%{
    compression = compression.value  # unwrap python enum and pass the integer value
%}

// Handle all functions that return this enum type
WRAP_ENUM_RETURN(mi::neuraylib, ITexture, get_compression, Texture_compression)




// special handling for: mi::neuraylib::Filter_type
// ----------------------------------------------------------------------------

%pythoncode {

    @post_swig_add_type_hint_mapping("mi::neuraylib::Filter_type", "Filter_type")
    class Filter_type(Enum):
        r"""
        Supported filter types.

        The filter type (or filter kernel) specifies how multiple samples are to be combined into a single pixel value.
        """

        FILTER_BOX = _pymdlsdk.FILTER_BOX
        r""" box filter"""
        FILTER_TRIANGLE = _pymdlsdk.FILTER_TRIANGLE
        r""" triangle filter"""
        FILTER_GAUSS = _pymdlsdk.FILTER_GAUSS
        r""" Gaussian filter"""
        FILTER_CMITCHELL = _pymdlsdk.FILTER_CMITCHELL
        r""" clipped Mitchell filter"""
        FILTER_CLANCZOS = _pymdlsdk.FILTER_CLANCZOS
        r""" clipped Lanczos filter"""
        FILTER_FAST = _pymdlsdk.FILTER_FAST
        r""" a fast filter, could be GPU anti-aliasing, or any"""
        FILTER_FORCE_32_BIT = _pymdlsdk.FILTER_FORCE_32_BIT
}

// unwrap all usages of the enum values
// - NONE-

// Handle all functions that return this enum type
// - NONE-




// special handling for: mi::neuraylib::Baker_resource
// ----------------------------------------------------------------------------

%pythoncode {

    @post_swig_add_type_hint_mapping("mi::neuraylib::Baker_resource", "Baker_resource")
    class Baker_resource(Enum):
        r"""
        Identifies the resource(s) to be used by a baker.

        See #mi::neuraylib::IMdl_distiller_api::create_baker().
        """

        BAKE_ON_CPU = _pymdlsdk.BAKE_ON_CPU
        r""" Use only the CPU for texture baking."""
        BAKE_ON_GPU = _pymdlsdk.BAKE_ON_GPU
        r""" Use only the GPU for texture baking."""
        BAKE_ON_GPU_WITH_CPU_FALLBACK = _pymdlsdk.BAKE_ON_GPU_WITH_CPU_FALLBACK
        r""" Prefer using the GPU for texture baking, use the CPU as fallback."""
        BAKER_RESOURCE_FORCE_32_BIT = _pymdlsdk.BAKER_RESOURCE_FORCE_32_BIT
}

// unwrap all usages of the enum values
// unwrap all usages of the enum values
%feature("pythonprepend") mi::neuraylib::IMdl_distiller_api::create_baker %{
    argc: int = len(args)
    if argc == 4:
        args = (args[0], args[1], args[2].value, args[3])
    if argc == 3:
        args = (args[0], args[1], args[2].value, 0)
    if argc == 2:
        args = (args[0], args[1], Baker_resource.BAKE_ON_CPU.value, 0)
%}

// Handle all functions that return this enum type
// - NONE-


// special handling for: mi::neuraylib::Mdl_repair_options
// ----------------------------------------------------------------------------

%pythoncode {

    @post_swig_add_type_hint_mapping("mi::neuraylib::Mdl_repair_options", "Mdl_repair_options")
    class Mdl_repair_options(Enum):
        r"""
        Options for repairing function calls.

        See #mi::neuraylib::IFunction_call::repair().
        """

        MDL_REPAIR_DEFAULT = _pymdlsdk.MDL_REPAIR_DEFAULT
        r""" Default mode, do not alter any inputs."""
        MDL_REMOVE_INVALID_ARGUMENTS = _pymdlsdk.MDL_REMOVE_INVALID_ARGUMENTS
        r""" Remove an invalid call attached to an argument."""
        MDL_REPAIR_INVALID_ARGUMENTS = _pymdlsdk.MDL_REPAIR_INVALID_ARGUMENTS
        r""" Attempt to repair invalid calls attached to an argument."""
        MDL_REPAIR_OPTIONS_FORCE_32_BIT = _pymdlsdk.MDL_REPAIR_OPTIONS_FORCE_32_BIT
}

// unwrap all usages of the enum values
// - NONE-

// Handle all functions that return this enum type
// - NONE-



// special handling for: mi::neuraylib::Bsdf_type
// ----------------------------------------------------------------------------

%pythoncode{

    @post_swig_add_type_hint_mapping("mi::neuraylib::Bsdf_type", "Bsdf_type")
    class Bsdf_type(Enum) :
        r"""The BSDF type."""

        BSDF_SCALAR = _pymdlsdk.BSDF_SCALAR
        r""" One scalar per grid value."""
        BSDF_RGB = _pymdlsdk.BSDF_RGB
        r""" Three scalars (RGB) per grid value."""

        BSDF_TYPES_FORCE_32_BIT = _pymdlsdk.BSDF_TYPES_FORCE_32_BIT
    }

// unwrap all usages of the enum values
// - NONE-

// Handle all functions that return this enum type
WRAP_ENUM_RETURN(mi::neuraylib, IBsdf_isotropic_data, get_type, Bsdf_type)
