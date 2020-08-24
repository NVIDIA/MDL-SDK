#*****************************************************************************
# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#*****************************************************************************

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
    "${MDL_INCLUDE_FOLDER}/mi/base.h"
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
    "${MDL_INCLUDE_FOLDER}/mi/math.h"
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
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_encapsulator.h"
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
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_module_transformer.h"
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
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_translator_plugin.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_types.h"
    "${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_values.h"
    )

set(PROJECT_HEADERS_NEURAYLIB
    "${MDL_INCLUDE_FOLDER}/mi/mdl_sdk.h"
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
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iexporter.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iexport_result.h"
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
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iimporter.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iimport_result.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ilightprofile.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ilogging_configuration.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imap.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imaterial_definition.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imaterial_instance.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imatrix.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdle_api.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_archive_api.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_backend.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_backend_api.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_compatibility_api.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_compiler.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_configuration.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_discovery_api.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_entity_resolver.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_evaluator_api.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_execution_context.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_factory.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_i18n_configuration.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_impexp_api.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_loading_wait_handle.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imdl_module_transformer.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/imodule.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/ineuray.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/inumber.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iplugin_api.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/iplugin_configuration.h"
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
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/matrix_typedefs.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/set_get.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/target_code_types.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/typedefs.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/type_traits.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/vector_typedefs.h"
    "${MDL_INCLUDE_FOLDER}/mi/neuraylib/version.h"
    )

set(PROJECT_SOURCES
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
