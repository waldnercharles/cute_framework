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

#include <cute.h>

using namespace cute;

void coroutine_func(cf_coroutine_t* co)
{
	int a, b;
	cf_coroutine_pop(co, &a, sizeof(a));
	cf_coroutine_pop(co, &b, sizeof(b));
	cf_coroutine_yield(co, NULL);

	int c = a * b;
	cf_coroutine_push(co, &c, sizeof(c));
}

void coroutine_wait_func(cf_coroutine_t* co)
{
	cf_coroutine_wait(co, 1.0f);

	int a = 3;
	cf_coroutine_push(co, &a, sizeof(a));
}

CUTE_TEST_CASE(test_coroutine, "Call some coroutine functions or whatever.");
int test_coroutine()
{
	cf_coroutine_t* co = cf_coroutine_make(coroutine_func, 0, NULL);
	int a = 5;
	int b = 10;
	int c = 0;
	cf_coroutine_push(co, &a, sizeof(a));
	cf_coroutine_push(co, &b, sizeof(b));
	cf_coroutine_resume(co, 0);
	CUTE_TEST_ASSERT(c == 0);
	cf_coroutine_resume(co, 0);
	cf_coroutine_pop(co, &c, sizeof(c));
	CUTE_TEST_ASSERT(c == 50);
	cf_coroutine_destroy(co);

	co = cf_coroutine_make(coroutine_func, 0, NULL);
	a = 5;
	b = 10;
	c = 0;
	cf_coroutine_push(co, &a, sizeof(a));
	cf_coroutine_push(co, &b, sizeof(b));
	cf_coroutine_resume(co, 0);
	CUTE_TEST_ASSERT(c == 0);
	cf_coroutine_resume(co, 0);
	cf_coroutine_pop(co, &c, sizeof(c));
	CUTE_TEST_ASSERT(c == 50);
	cf_coroutine_destroy(co);

	co = cf_coroutine_make(coroutine_wait_func, 0, NULL);
	cf_coroutine_resume(co, 0);
	size_t bytes = cf_coroutine_bytes_pushed(co);
	CUTE_TEST_ASSERT(bytes == 0);
	cf_coroutine_resume(co, 0);
	cf_coroutine_resume(co, 0);
	cf_coroutine_resume(co, 0);
	bytes = cf_coroutine_bytes_pushed(co);
	CUTE_TEST_ASSERT(bytes == 0);
	cf_coroutine_resume(co, 0.5f);
	bytes = cf_coroutine_bytes_pushed(co);
	CUTE_TEST_ASSERT(bytes == 0);
	cf_coroutine_resume(co, 0.5f);
	bytes = cf_coroutine_bytes_pushed(co);
	CUTE_TEST_ASSERT(bytes == sizeof(int));
	cf_coroutine_pop(co, &a, sizeof(a));
	CUTE_TEST_ASSERT(a == 3);
	CUTE_TEST_ASSERT(cf_coroutine_state(co) == CF_COROUTINE_STATE_DEAD);
	cf_coroutine_destroy(co);

	return 0;
}
