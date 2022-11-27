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

#ifndef CUTE_ASEPRITE_CACHE_H
#define CUTE_ASEPRITE_CACHE_H

#include "cute_defines.h"
#include "cute_result.h"
#include "cute_batch.h"

#include "cute/cute_aseprite.h"

//--------------------------------------------------------------------------------------------------
// C++ API

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * The aseprite cache is used to load ase files from disk in order to make sprites.
 *
 * You will mostly just care about these three functions.
 *
 *     cf_aseprite_cache_load
 *     cf_aseprite_cache_unload
 *     cf_make_aseprite_cache
 *
 * It's a cache, which means it actually caches aseprites loaded in RAM, so subsequent
 * calls to `aseprite_cache_load` won't have to fetch the image off of disk, as long as
 * the image is currently cached in RAM.
 */
typedef struct cf_aseprite_cache_t cf_aseprite_cache_t;
typedef struct cf_sprite_t cf_sprite_t;

/**
 * Constructs a new aseprite cache. Destroy it with `cf_destroy_aseprite_cache` when done with it.
 */
CUTE_API cf_aseprite_cache_t* CUTE_CALL cf_make_aseprite_cache();

/**
 * Destroys a aseprite cache previously made with `cf_make_aseprite_cache`.
 */
CUTE_API void CUTE_CALL cf_destroy_aseprite_cache(cf_aseprite_cache_t* cache);

/**
 * Returns a sprite from the cache. If it does not exist in the cache, it is loaded from disk
 * and placed into the cache.
 */
CUTE_API cf_result_t CUTE_CALL cf_aseprite_cache_load(cf_aseprite_cache_t* cache, const char* aseprite_path, cf_sprite_t* sprite);

/**
 * Removes a sprite from the cache. This will cause the next call to `cf_aseprite_cache_load` to fetch from disk.
 */
CUTE_API void CUTE_CALL cf_aseprite_cache_unload(cf_aseprite_cache_t* cache, const char* aseprite_path);

/**
 * A low-level function used to return an `ase_t` from the cache. If it does not exist within the cache
 * it is loaded from disk.
 *
 * This function is typically not necessary to call. You might be looking for `cf_aseprite_cache_load`
 * instead.
 *
 * Only call this function if you know what you're doing.
 */
CUTE_API cf_result_t CUTE_CALL cf_aseprite_cache_load_ase(cf_aseprite_cache_t* cache, const char* aseprite_path, ase_t** ase);

#ifdef __cplusplus
}
#endif // __cplusplus

//--------------------------------------------------------------------------------------------------
// C API

#ifdef CUTE_CPP

namespace cute
{

struct sprite_t;
using aseprite_cache_t = cf_aseprite_cache_t;

CUTE_INLINE aseprite_cache_t* cmake_aseprite_cache() { return cf_make_aseprite_cache(); }
CUTE_INLINE void destroy_aseprite_cache(aseprite_cache_t* cache) { cf_destroy_aseprite_cache(cache); }
CUTE_INLINE result_t aseprite_cache_load(aseprite_cache_t* cache, const char* aseprite_path, sprite_t* sprite) { return cf_aseprite_cache_load(cache, aseprite_path, (cf_sprite_t*)sprite); }
CUTE_INLINE void aseprite_cache_unload(aseprite_cache_t* cache, const char* aseprite_path) { cf_aseprite_cache_unload(cache, aseprite_path); }
CUTE_INLINE result_t aseprite_cache_load_ase(aseprite_cache_t* cache, const char* aseprite_path, ase_t** ase) { return  cf_aseprite_cache_load_ase(cache, aseprite_path, ase); }

}

#endif // CUTE_CPP

#endif // CUTE_ASEPRITE_CACHE_H
