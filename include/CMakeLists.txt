# name of the target and the resulting library
set(PROJECT_NAME include-mi)

# add dummy-source to create a dummy-lib
set(DUMMY_CPP ${CMAKE_CURRENT_BINARY_DIR}/generated/empty.cpp)
if(NOT EXISTS ${DUMMY_CPP})
    file(WRITE ${DUMMY_CPP} "")
endif()

# collect sources
set(PROJECT_HEADERS_BASE
    "${MDL_INCLUDE_FOLDER}/mi/base/assert.h"
    "${MDL_INCLUDE_FOLDER}/mi/base/atom.h"
    "${MDL_INCLUDE_FOLDER}/mi/base/condition.h"
    "${MDL_INCLUDE_FOLDER}/mi/base/config.h"
    "${MDL_INCLUDE_FOLDER}/mi/base/default_allocator.h"
    "${MDL_INCLUDE_FOLDER}/mi/base/enums.h"
    "${MDL_INCLUDE_FOLDER}/mi/base/handle.h"
    "${MDL_INCLUDE_FOLDER}/mi/base/iallocator.h"
    "${MDL_INCLUDE_FOLDER}/mi/base/iinterface.h"
    "${MDL_INCLUDE_FOLDER}/mi/base/ilogger.h"
    "${MDL_INCLUDE_FOLDER}/mi/base/interface_declare.h"
    "${MDL_INCLUDE_FOLDER}/mi/base/interface_implement.h"
    "${MDL_INCLUDE_FOLDER}/mi/base/interface_merger.h"
    "${MDL_INCLUDE_FOLDER}/mi/base/lock.h"
    "${MDL_INCLUDE_FOLDER}/mi/base/miwindows.h"
    "${MDL_INCLUDE_FOLDER}/mi/base/plugin.h"
    "${MDL_INCLUDE_FOLDER}/mi/base/std_allocator.h"
    "${MDL_INCLUDE_FOLDER}/mi/base/types.h"
    "${MDL_INCLUDE_FOLDER}/mi/base/uuid.h"
    "${MDL_INCLUDE_FOLDER}/mi/base/version.h"
    )

set(PROJECT_HEADERS_MATH
    "${MDL_INCLUDE_FOLDER}/mi/math/assert.h"
    "${MDL_INCLUDE_FOLDER}/mi/math/bbox.h"
    "${MDL_INCLUDE_FOLDER}/mi/math/color.h"
    "${MDL_INCLUDE_FOLDER}/mi/math/function.h"
    "${MDL_INCLUDE_FOLDER}/mi/math/matrix.h"
    "${MDL_INCLUDE_FOLDER}/mi/math/spectrum.h"
    "${MDL_INCLUDE_FOLDER}/mi/math/vector.h"
    "${MDL_INCLUDE_FOLDER}/mi/math/version.h"
    )

set(PROJECT_HEADERS_MDL
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_annotations.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_archiver.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_assert.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_code_generators.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_comparator.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_declarations.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_definitions.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_entity_resolver.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_expressions.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_fatal.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_generated_code.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_generated_dag.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_generated_executable.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_iowned.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_mdl.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_messages.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_modules.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_names.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_options.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_positions.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_printers.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_serializer.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_statements.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_stdlib_types.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_streams.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_symbols.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_target_types.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_thread_context.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_types.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_values.h"
    )

set(PROJECT_HEADERS_NEURAYLIB
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/annotation_wrapper.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/argument_editor.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/assert.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/bsdf_isotropic_data.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/definition_wrapper.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/factory.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iallocator.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iarray.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iattribute_container.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iattribute_set.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ibbox.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ibsdf_isotropic_data.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ibsdf_measurement.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ibuffer.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/icanvas.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/icolor.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/icompiled_material.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/icompound.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/idata.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/idatabase.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/idebug_configuration.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ideserializer.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/idynamic_array.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ienum.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ienum_decl.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iexport_result.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iexporter.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iexpression.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iextension_api.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ifactory.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ifunction_call.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ifunction_definition.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iimage.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iimage_api.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iimage_plugin.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iimpexp_base.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iimpexp_state.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iimport_result.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iimporter.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ilightprofile.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ilogging_configuration.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imap.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imaterial_definition.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imaterial_instance.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imatrix.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_archive_api.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_compatibility_api.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_compiler.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_discovery_api.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_evaluator_api.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_execution_context.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_factory.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_i18n_configuration.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imodule.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ineuray.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/inumber.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iplugin_api.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ipointer.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ireader.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ireader_writer_base.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iref.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iscene_element.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iscope.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iserializer.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ispectrum.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/istream_position.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/istring.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/istructure.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/istructure_decl.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/itexture.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/itile.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/itransaction.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/itype.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iuser_class.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iuser_class_factory.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iuuid.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ivalue.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ivector.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iversion.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iwriter.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/set_get.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/target_code_types.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/type_traits.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/typedefs.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/version.h"
    )

set(PROJECT_SOURCES
    "${MDL_INCLUDE_FOLDER}/mi/base.h"
    "${MDL_INCLUDE_FOLDER}/mi/math.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl_sdk.h"
     ${PROJECT_HEADERS_BASE}
     ${PROJECT_HEADERS_MATH}
     ${PROJECT_HEADERS_MDL}
     ${PROJECT_HEADERS_NEURAYLIB}
     ${DUMMY_CPP}
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    )

# override position and name in the folder hierarchy
set_target_properties(${PROJECT_NAME} PROPERTIES 
    FOLDER          "include"               # hierarchy
    PROJECT_LABEL   "mi"                    # project name
    )

# group files corresponding to the folder structure
source_group("base" FILES ${PROJECT_HEADERS_BASE})
source_group("math" FILES ${PROJECT_HEADERS_MATH})
source_group("mdl" FILES ${PROJECT_HEADERS_MDL})
source_group("neuraylib" FILES ${PROJECT_HEADERS_NEURAYLIB})
