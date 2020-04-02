/******************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/
/// \file
/// \brief      List of all modules.
///
/// This file contains the \c MI::SYSTEM::Module_id enum. This enum lists all
/// neuray modules.

#ifndef BASE_LIB_MAIN_MODULE_ID_H
#define BASE_LIB_MAIN_MODULE_ID_H

namespace MI {
namespace SYSTEM {


/** \brief List of all modules in the system.

 When adding a module which has no module
 class then the name of the module must be added at the appropriate place
 in module_names in module.cpp!
*/
enum Module_id {
    M_MAIN,		// base/system/main
    M_ATTR,		// base/data/attr
    M_CACHE_MANAGER,	// base/data/cache_manager
    M_CLUSTER,		// base/data/cluster
    M_DATA,		// base/data/data
    M_DB,		// base/data/db
    M_DBNET,		// base/data/dbnet
    M_HTTP,		// base/data/http
    M_NET,		// base/data/net
    M_DISK_CACHE,       // base/data/diskcache
    M_DISCNET,		// base/data/discnet
    M_GPU_MANAGER,	// base/data/gpu_manager
    M_NODEMGR,		// base/data/nodemgr
    M_PAGER,		// base/data/pager
    M_RDMA,		// base/data/rdma
    M_SCH,		// base/data/sched
    M_SERIAL,		// base/data/serial
    M_REMOTE,		// base/data/remote
    M_BRIDGE_VIDEO,	// base/data/bridge_video
    M_STREAMS,  	// base/data/streams
    M_RTMP,		// base/data/rtmp
    M_TCPNET,		// base/data/tcpnet
    M_THREAD_POOL,	// base/data/thread_pool
    M_CUDA,             // base/hal/cuda
    M_DISK,		// base/hal/disk
    M_GPU,              // base/hal/gpu
    M_HAL,		// base/hal/hal
    M_HOST,		// base/hal/host
    M_LINK,		// base/hal/link
    M_MSG,		// base/hal/msg
    M_SSL,		// base/hal/ssl
    M_THREAD,		// base/hal/thread
    M_TIME,		// base/hal/time
    M_BBOX,		// base/lib/bbox
    M_BVH,              // base/lib/bvh
    M_COLOR,		// base/lib/color
    M_CONFIG,		// base/lib/config
    M_CONT,		// base/lib/cont
    M_DIFFEQ,		// base/lib/diffeq
    M_ENTY,           	// base/lib/entropy
    M_KDTREE,		// base/lib/kdtree
    M_LICENSE,		// base/lib/license
    M_LALG,		// base/lib/linalg
    M_LOG,		// base/lib/log
    M_MATH,		// base/lib/math
    M_MEM,		// base/lib/mem
    M_PATH,		// base/lib/path
    M_PLUG,		// base/lib/plug
    M_RESTRICTIONS,	// base/lib/restrictions
    M_SPML,		// base/lib/spml
    M_STREAM,           // base/lib/stream
    M_STRING,		// base/lib/string
    M_STRINGID_MOD,     // base/lib/stringid_module
    M_ZLIB,		// base/lib/zlib
    M_I18N,          // mdl/integration/i18n
    M_MDLC,             // mdl/integration/mdlnr
    M_IMAGE,		// io/image/image
    M_XML,		// io/parser/xml
    M_BSDF_MEASUREMENT,	// io/scene/bsdf_measurement
    M_CAMERA,		// io/scene/camera
    M_DBIMAGE,		// io/scene/dbimage
    M_GROUP,		// io/scene/group
    M_PROXY,		// io/scene/proxy
    M_INSTANCE,		// io/scene/instance
    M_LIGHT,		// io/scene/light
    M_LIGHTPROFILE,	// io/scene/lightprofile
    M_OPTIONS,		// io/scene/options
    M_SCENE,		// io/scene/scene
    M_TEXTURE,		// io/scene/texture
    M_DISPLACE,		// geometry/gap/displace
    M_FFS,		// geometry/ffs/ffs
    M_FFSCONV,		// geometry/ffs/ffsconv
    M_FFSCURVE,		// geometry/ffs/ffscurve
    M_FFSREG,		// geometry/ffs/ffsreg
    M_FFSTESS,		// geometry/ffs/ffstess
    M_GAP,		// geometry/gap/gap
    M_GAPLIB,		// geometry/gap/gaplib
    M_TRIANG,		// geometry/poly/triangulate
    M_TRIQTESS,		// geometry/poly/triqtess
    M_TRIQREG,		// geometry/poly/triqreg
    M_SDSREG,           // geometry/sds/sdsreg
    M_SDSTESS,		// geometry/sds/sdstess
    M_GDIFF,          	// geometry/diffgeo/gdiffusion
    M_GEOLIB,          	// geometry/geolib/geolib
    M_TOPO,		// geometry/geolib/topology
    M_GPRIM,          	// geometry/geolib/primitives
    M_CACHEOPT,		// geometry/mesh/cacheopt
    M_GCMP,		// geometry/mesh/gcompress
    M_PHT,		// geometry/mesh/phongtess
    M_SIMP,		// geometry/mesh/simplify
    M_STITCH,          	// geometry/mesh/stitch
    M_WELD,          	// geometry/mesh/weld
    M_ICMAP,		// render/particlemap/irradcache
    M_PMAP,		// render/particlemap/particlemap
    M_BACKENDS,		// render/mdl/backends
    M_ASSEMBLY,		// render/render/assembly
    M_BRDF,		// render/render/brdf
    M_FRAMEBUFFER,	// render/render/framebuffer
    M_LPEXPR,           // render/render/lpexpr
    M_MATCONV,          // render/render/matconv
    M_MRMDL,          	// render/render/mrmetasl (mr only)
    M_RAL,		// render/render/ral
    M_RDIFF,		// render/render/rdiff
    M_RENDER,		// render/render/render
    M_SAMP,		// render/render/samp
    M_SHADER,		// render/render/shader
    M_SOFTSHADER,	// render/render/softshader
    M_TRAVERSE,		// render/render/traverse
    M_TRAVERSE_API,     // render/render/traverse_api
    M_IRAY,             // render/iray/...
    M_BSP,		// some mental ray code
    M_SWRCLIB,		// render/swrc/swrclib
    M_SWRCLIGHT,	// render/swrc/swrclight
    M_NEURAY_API,	// api/api/neuray
    M_MESH,		// prod/lib/mentalmesh
    M_DIST,             // mdl/distiller/dist
    M_BAKER,            // mdl/distiller/baker
    M_POSTPROCESSING,	// render/postprocessing
    NUM_OF_MODULES	// number of modules
};



}

using namespace SYSTEM;

}

#endif //BASE_LIB_MAIN_MODULE_ID_H
