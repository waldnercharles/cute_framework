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

#ifndef CUTE_TIMER_H
#define CUTE_TIMER_H

#include "cute_defines.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Calculates the time, in seconds, since the last time this function was called.
 * No special care is taken to handle multi-threading (this function uses static memory).
 * Returns 0 on the first call.
 * 
 * For more fine-grained measuring of time, try using `timer_t`.
 */
CUTE_API float CUTE_CALL cf_calc_dt();

typedef struct cf_timer_t
{
	double inv_freq;
	uint64_t prev;
} cf_timer_t;

/**
 * Initializes a new `timer_t` on the stack.
 */
CUTE_API cf_timer_t CUTE_CALL cf_timer_init();

/**
 * Returns the time elapsed since the last call to `timer_dt` was made.
 */
CUTE_API float CUTE_CALL cf_timer_dt(cf_timer_t* timer);

/**
 * Returns the time elapsed since the last call to `timer_dt` was made. Use this function
 * to repeatedly measure the time since the last `timer_dt` call.
 */
CUTE_API float CUTE_CALL cf_timer_elapsed(cf_timer_t* timer);

#ifdef __cplusplus
}
#endif // __cplusplus

#ifdef  CUTE_CPP

namespace cute
{

using timer_t = cf_timer_t;

CUTE_INLINE float calc_dt() { return cf_calc_dt(); }
CUTE_INLINE timer_t timer_init() { return cf_timer_init(); }
CUTE_INLINE float timer_dt(timer_t* timer) { return cf_timer_dt(timer); }
CUTE_INLINE float timer_elapsed(timer_t* timer) { return cf_timer_elapsed(timer); }

}

#endif //  CUTE_CPP

#endif // CUTE_TIMER_H
