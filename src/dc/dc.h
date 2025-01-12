#ifndef DC_H
#define DC_H

#include "mem_dc.h"

#if !defined(DREAMCAST)

#ifdef __cplusplus
#define FLOOR_REAL std::floor
#define CEIL_REAL std::ceil
#define MIN_REAL std::min
#define MAX_REAL std::max
#define MIN_REAL_INT std::min
#define MAX_REAL_INT std::max

#define ABS_REAL std::abs
#define SIN_REAL std::sin
#define COS_REAL std::cos
#else
#define FLOOR_REAL floor
#define CEIL_REAL ceil
#define MIN_REAL_INT min
#define MAX_REAL_INT max
#define MIN_REAL min
#define MAX_REAL max
#define ABS_REAL abs
#define SIN_REAL sin
#define COS_REAL cos
#endif

#define SQRTF_REAL sqrtf
#define DIVIDE_REAL(a,b) (a / b)
#define MEMSET_REAL memset
#define MEMCPY_REAL memcpy
#define SUPER_MEMCPY_REAL memcpy

#define FSSRA_REAL(a) (1.f/sqrt(a))

#define FMAC(a, b, c) ((a) * (b) + (c))
#define FMAC_DEC(a, b, c) ((a) * (b) - (c))

#define DEFAULT_TO_FASTEST int

#else
#include "sh4_math.h"
#include <dc/fmath.h>
#define FLOOR_REAL MATH_Fast_Floorf
// MATH_Very_Fast_Floorf doesn't work properly with Dusk Child

#define CEIL_REAL MATH_Fast_Ceilf
#define MIN_REAL MATH_Fast_Fminf
#define MAX_REAL MATH_Fast_Fmaxf

#define MIN_REAL_INT MATH_Fast_Fminf
#define MAX_REAL_INT MATH_Fast_Fmaxf

#define ABS_REAL MATH_fabs
#define SQRTF_REAL MATH_Fast_Sqrt

#define FSSRA_REAL(a) MATH_fsrra(a)

#define SIN_REAL fsin
#define COS_REAL fcos
	
#define DIVIDE_REAL(a,b) MATH_Fast_Divide(a, b)
#define MEMSET_REAL memsetasm
#define MEMCPY_REAL memcpy6
#define SUPER_MEMCPY_REAL bit64_sq_cpy

// FMAC ((a) * (b) + (c))
// FMAC_DEC ((a) * (b) - (c))

#define FMAC(a, b, c) MATH_fmac(a,b,c)
#define FMAC_DEC(a, b, c) MATH_fmac_Dec(a,b,c)

#define DEFAULT_TO_FASTEST float

#endif


#endif
