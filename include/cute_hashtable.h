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

#ifndef CUTE_HASHTABLE_H
#define CUTE_HASHTABLE_H

#include "cute_defines.h"

//--------------------------------------------------------------------------------------------------
// C API

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


typedef struct cf_hashtable_slot_t
{
	uint64_t key_hash;
	int item_index;
	int base_count;
} cf_hashtable_slot_t;

typedef struct cf_hashtable_t
{
	int count;
	int slot_capacity;
	cf_hashtable_slot_t* slots;

	int key_size;
	int item_size;
	int item_capacity;
	void* items_key;
	int* items_slot_index;
	void* items_data;

	void* temp_key;
	void* temp_item;
	void* mem_ctx;
} cf_hashtable_t;

CUTE_API int CUTE_CALL cf_hashtable_init(cf_hashtable_t* table, int key_size, int item_size, int capacity, void* mem_ctx);
CUTE_API void CUTE_CALL cf_hashtable_cleanup(cf_hashtable_t* table);

CUTE_API void* CUTE_CALL cf_hashtable_insert(cf_hashtable_t* table, const void* key, const void* item);
CUTE_API void CUTE_CALL cf_hashtable_remove(cf_hashtable_t* table, const void* key);
CUTE_API void CUTE_CALL cf_hashtable_clear(cf_hashtable_t* table);
CUTE_API void* CUTE_CALL cf_hashtable_find(const cf_hashtable_t* table, const void* key);
CUTE_API int CUTE_CALL cf_hashtable_count(const cf_hashtable_t* table);
CUTE_API void* CUTE_CALL cf_hashtable_items(const cf_hashtable_t* table);
CUTE_API void* CUTE_CALL cf_hashtable_keys(const cf_hashtable_t* table);
CUTE_API void CUTE_CALL cf_hashtable_swap(cf_hashtable_t* table, int index_a, int index_b);

#ifdef __cplusplus
}
#endif // __cplusplus

//--------------------------------------------------------------------------------------------------
// C++ API

#ifdef CUTE_CPP

namespace cute
{

using hashtable_slot_t = cf_hashtable_slot_t;

using hashtable_t = cf_hashtable_t;

CUTE_INLINE int hashtable_init(hashtable_t* table, int key_size, int item_size, int capacity, void* mem_ctx) { return cf_hashtable_init(table,key_size,item_size,capacity,mem_ctx); }
CUTE_INLINE void hashtable_cleanup(hashtable_t* table) { cf_hashtable_cleanup(table); }
CUTE_INLINE void* hashtable_insert(hashtable_t* table, const void* key, const void* item) { return cf_hashtable_insert(table,key,item); }
CUTE_INLINE void hashtable_remove(hashtable_t* table, const void* key) { cf_hashtable_remove(table,key); }
CUTE_INLINE void hashtable_clear(hashtable_t* table) { cf_hashtable_clear(table); }
CUTE_INLINE void* hashtable_find(const hashtable_t* table, const void* key) { return cf_hashtable_find(table,key); }
CUTE_INLINE int hashtable_count(const hashtable_t* table) { return cf_hashtable_count(table); }
CUTE_INLINE void* hashtable_items(const hashtable_t* table) { return cf_hashtable_items(table); }
CUTE_INLINE void* hashtable_keys(const hashtable_t* table) { return cf_hashtable_keys(table); }
CUTE_INLINE void hashtable_swap(hashtable_t* table, int index_a, int index_b) { cf_hashtable_swap(table,index_a,index_b); }

}

#endif // CUTE_CPP

#endif // CUTE_HASHTABLE_H
