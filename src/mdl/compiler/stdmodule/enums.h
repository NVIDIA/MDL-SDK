#ifndef MDL_COMPILER_STDMODULE_ENUMS_H
#define MDL_COMPILER_STDMODULE_ENUMS_H

// This file replicates the enums from the standard modules for use in C++ code.

namespace mi {
namespace mdl {

namespace df {

enum scatter_mode {
    scatter_reflect,
    scatter_transmit,
    scatter_reflect_transmit
};

}

namespace state {

enum coordinate_space {
    coordinate_internal,
    coordinate_object,
    coordinate_world
};

}

namespace tex {

enum gamma_mode {
    gamma_default,
    gamma_linear,
    gamma_srgb
};

enum wrap_mode {
    wrap_clamp,
    wrap_repeat,
    wrap_mirrored_repeat,
    wrap_clip
};

}

} // namespace mdl
} // namespace mi

#endif // MDL_COMPILER_STDMODULE_ENUMS_H
