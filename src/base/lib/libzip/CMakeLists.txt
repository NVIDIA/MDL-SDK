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
set(PROJECT_NAME base-lib-libzip)

# collect sources
set(PROJECT_HEADERS
    "compat.h"
    "config.h"
    "zip.h"
    "zipconf.h"
    "zipint.h"
    )

set(PROJECT_SOURCES 
    "zip_add.c"
    "zip_add_dir.c"
    "zip_add_entry.c"
    "zip_algorithm_deflate.c"
    "zip_buffer.c"
    "zip_close.c"
    "zip_delete.c"
    "zip_dir_add.c"
    "zip_dirent.c"
    "zip_discard.c"
    "zip_entry.c"
    "zip_err_str.c"
    "zip_error.c"
    "zip_error_clear.c"
    "zip_error_get.c"
    "zip_error_get_sys_type.c"
    "zip_error_strerror.c"
    "zip_error_to_str.c"
    "zip_extra_field.c"
    "zip_extra_field_api.c"
    "zip_fclose.c"
    "zip_fdopen.c"
    "zip_file_add.c"
    "zip_file_error_clear.c"
    "zip_file_error_get.c"
    "zip_file_get_comment.c"
    "zip_file_get_external_attributes.c"
    "zip_file_get_offset.c"
    "zip_file_rename.c"
    "zip_file_replace.c"
    "zip_file_set_comment.c"
    "zip_file_set_encryption.c"
    "zip_file_set_external_attributes.c"
    "zip_file_set_mtime.c"
    "zip_file_strerror.c"
    "zip_filerange_crc.c"
    "zip_fopen.c"
    "zip_fopen_encrypted.c"
    "zip_fopen_index.c"
    "zip_fopen_index_encrypted.c"
    "zip_fread.c"
    "zip_fseek.c"
    "zip_ftell.c"
    "zip_get_archive_comment.c"
    "zip_get_archive_flag.c"
    "zip_get_encryption_implementation.c"
    "zip_get_file_comment.c"
    "zip_get_name.c"
    "zip_get_num_entries.c"
    "zip_get_num_files.c"
    "zip_hash.c"
    "zip_io_util.c"
    "zip_libzip_version.c"
    "zip_memdup.c"
    "zip_name_locate.c"
    "zip_new.c"
    "zip_open.c"
    "zip_progress.c"
    "zip_rename.c"
    "zip_replace.c"
    "zip_set_archive_comment.c"
    "zip_set_archive_flag.c"
    "zip_set_default_password.c"
    "zip_set_file_comment.c"
    "zip_set_file_compression.c"
    "zip_set_name.c"
    "zip_source_begin_write.c"
    "zip_source_begin_write_cloning.c"
    "zip_source_buffer.c"
    "zip_source_call.c"
    "zip_source_close.c"
    "zip_source_commit_write.c"
    "zip_source_compress.c"
    "zip_source_crc.c"
    "zip_source_error.c"
    "zip_source_filep.c"
    "zip_source_free.c"
    "zip_source_function.c"
    "zip_source_get_compression_flags.c"
    "zip_source_is_deleted.c"
    "zip_source_layered.c"
    "zip_source_open.c"
    "zip_source_pkware.c"
    "zip_source_read.c"
    "zip_source_remove.c"
    "zip_source_rollback_write.c"
    "zip_source_seek.c"
    "zip_source_seek_write.c"
    "zip_source_stat.c"
    "zip_source_supports.c"
    "zip_source_tell.c"
    "zip_source_tell_write.c"
    "zip_source_window.c"
    "zip_source_write.c"
    "zip_source_zip.c"
    "zip_source_zip_new.c"
    "zip_stat.c"
    "zip_stat_index.c"
    "zip_stat_init.c"
    "zip_strerror.c"
    "zip_string.c"
    "zip_unchange.c"
    "zip_unchange_all.c"
    "zip_unchange_archive.c"
    "zip_unchange_data.c"
    "zip_utf-8.c"
    ${PROJECT_HEADERS}
    )

# platform depenedent sources
if(WIN32)
    list(APPEND PROJECT_SOURCES
        "zip_source_win32handle.c"
        "zip_source_win32utf8.c"
        "zip_source_win32w.c"
        "mkstemp.c"
        "zipwin32.h"
        )
else()
    list(APPEND PROJECT_SOURCES
        "zip_source_file.c"
        )
endif()

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    ADDITIONAL_INCLUDE_DIRS 
        ${CMAKE_CURRENT_SOURCE_DIR}/../zlib
    )

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS 
        boost
    )

# additional defines
target_compile_definitions(${PROJECT_NAME} 
    PRIVATE
        "HAVE_CONFIG_H"
    )
            