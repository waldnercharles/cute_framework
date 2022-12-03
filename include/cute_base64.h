/*
	Cute Framework
	Copyright (C) 2019 Randy Gaul https://randygaul.net

	This software is provided 'as-is', without any express or implied
	warranty.  In no event will the authors be held liable for any damages
	arising from the use of this software.

	Permission is granted to anyone to use this software for any purpose,
	including commercial applications, and to alter it and redistribute it
	freely, subject to the following restrictions:

	1. The origin of this software must not be misrepresented; you must not
	   claim that you wrote the original software. If you use this software
	   in a product, an acknowledgment in the product documentation would be
	   appreciated but is not required.
	2. Altered source versions must be plainly marked as such, and must not be
	   misrepresented as being the original software.
	3. This notice may not be removed or altered from any source distribution.
*/

#ifndef CUTE_BASE64_H
#define CUTE_BASE64_H

#include "cute_defines.h"
#include "cute_result.h"

//--------------------------------------------------------------------------------------------------
// C API

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Info about base 64 encoding: https://tools.ietf.org/html/rfc4648

#define CUTE_BASE64_ENCODED_SIZE(size) ((((size) + 2) / 3) * 4)
#define CUTE_BASE64_DECODED_SIZE(size) ((((size) + 3) / 4) * 3)

CUTE_API CF_Result CUTE_CALL cf_base64_encode(void* dst, size_t dst_size, const void* src, size_t src_size);
CUTE_API CF_Result CUTE_CALL cf_base64_decode(void* dst, size_t dst_size, const void* src, size_t src_size);

#ifdef __cplusplus
}
#endif // __cplusplus

//--------------------------------------------------------------------------------------------------
// C++ API

#ifdef CUTE_CPP

namespace Cute
{

CUTE_INLINE Result base64_encode(void* dst, size_t dst_size, const void* src, size_t src_size) { return cf_base64_encode(dst, dst_size, src, src_size); }
CUTE_INLINE Result base64_decode(void* dst, size_t dst_size, const void* src, size_t src_size) { return cf_base64_decode(dst, dst_size, src, src_size); }

}

#endif // CUTE_CPP

#endif // !CUTE_BASE64_H