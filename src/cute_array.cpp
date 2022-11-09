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

#include "cute_array.h"
#include "cute_math.h"

using namespace cute;

void* cf_agrow(const void* a, int new_size, size_t element_size)
{
	CF_ACANARY(a);
	CUTE_ASSERT(acap(a) <= (SIZE_MAX - 1)/2);
	int new_capacity = max(2 * acap(a), max(new_size, 16));
	CUTE_ASSERT(new_size <= new_capacity);
	CUTE_ASSERT(new_capacity <= (SIZE_MAX - sizeof(cf_ahdr_t)) / element_size);
	size_t total_size = sizeof(cf_ahdr_t) + new_capacity * element_size;
	cf_ahdr_t* hdr;
	if (a) {
		if (!CF_AHDR(a)->is_static) {
			hdr = (cf_ahdr_t*)CUTE_REALLOC(CF_AHDR(a), total_size);
		} else {
			hdr = (cf_ahdr_t*)CUTE_ALLOC(total_size);
			hdr->size = asize(a);
			hdr->cookie = CF_ACOOKIE;
		}
	} else {
		hdr = (cf_ahdr_t*)CUTE_ALLOC(total_size);
		hdr->size = 0;
		hdr->cookie = CF_ACOOKIE;
	}
	hdr->capacity = new_capacity;
	hdr->is_static = false;
	hdr->data = (char*)(hdr + 1); // For debugging convenience.
	return (void*)(hdr + 1);
}

void* cf_astatic(const void* a, int buffer_size, size_t element_size)
{
	cf_ahdr_t* hdr = (cf_ahdr_t*)a;
	hdr->size = 0;
	hdr->cookie = CF_ACOOKIE;
	if (sizeof(cf_ahdr_t) <= element_size) {
		hdr->capacity = buffer_size / (int)element_size - 1;
	} else {
		int elements_taken = sizeof(cf_ahdr_t) / (int)element_size + (sizeof(cf_ahdr_t) % (int)element_size > 0);
		hdr->capacity = buffer_size / (int)element_size - elements_taken;
	}
	hdr->data = (char*)(hdr + 1); // For debugging convenience.
	hdr->is_static = true;
	return (void*)(hdr + 1);
}

void* cf_aset(const void* a, const void* b, size_t element_size)
{
	CF_ACANARY(a);
	CF_ACANARY(b);
	if (acap(a) < asize(b)) {
		int len = asize(b);
		a = cf_agrow(a, asize(b), element_size);
	}
	CUTE_MEMCPY((void*)a, b, asize(b) * element_size);
	alen(a) = asize(b);
	return (void*)a;
}
