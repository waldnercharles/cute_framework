/*
	Cute Framework
	Copyright (C) 2023 Randy Gaul https://randygaul.github.io/

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

#ifndef CF_ALLOC_INTERNAL_H
#define CF_ALLOC_INTERNAL_H

#include <cute_defines.h>
#include <cute_alloc.h>

#if !defined(CF_ALLOC) && !defined(CF_FREE)
#	define CF_CALLOC(size) cf_calloc(size, 1)
#	define CF_ALLOC(size) cf_alloc(size)
#	define CF_FREE(ptr) cf_free(ptr)
#	define CF_REALLOC(ptr, size) cf_realloc(ptr, size)
#endif

#endif // CF_ALLOC_INTERNAL_H
