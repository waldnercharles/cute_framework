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
#include <cute_alloc.h>
#include <SDL.h>
#include <glad.h>

namespace cute
{

struct cute_t
{
	int running;
	SDL_Window* window;
	void* mem_ctx;
};

cute_t* cute_make(const char* window_title, int x, int y, int w, int h, uint32_t options, void* user_allocator_context)
{
	cute_t* cute = (cute_t*)CUTE_ALLOC(sizeof(cute_t), user_allocator_context);

	if (!(options & CUTE_OPTIONS_NO_GFX)) {
		SDL_InitSubSystem(SDL_INIT_VIDEO);
	}

	if (!(options & CUTE_OPTIONS_NO_AUDIO)) {
		SDL_InitSubSystem(SDL_INIT_AUDIO);
	}

	Uint32 flags = 0;
	if (options & CUTE_OPTIONS_GFX_GL) flags |= SDL_WINDOW_OPENGL;
	if (options & CUTE_OPTIONS_GFX_GLES) flags |= SDL_WINDOW_OPENGL;
	if (options & CUTE_OPTIONS_FULLSCREEN) flags |= SDL_WINDOW_FULLSCREEN;
	if (options & CUTE_OPTIONS_RESIZABLE) flags |= SDL_WINDOW_RESIZABLE;
	SDL_Window* window = SDL_CreateWindow(window_title, x, y, w, h, flags);
	cute->running = 1;
	cute->window = window;
	cute->mem_ctx = user_allocator_context;

	if (options & CUTE_OPTIONS_GFX_GL) {
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	}

	if (options & CUTE_OPTIONS_GFX_GLES) {
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
	}

	if ((options & CUTE_OPTIONS_GFX_GL) | (options & CUTE_OPTIONS_GFX_GLES)) {
		SDL_GL_SetSwapInterval(0);
		SDL_GL_CreateContext(window);
		gladLoadGLLoader(SDL_GL_GetProcAddress);
	}

	return cute;
}

void cute_destroy(cute_t* cute)
{
	SDL_DestroyWindow(cute->window);
	SDL_Quit();
	CUTE_FREE(cute, cute->mem_ctx);
}

int is_running(cute_t* cute)
{
	return cute->running;
}

void stop_running(cute_t* cute)
{
	cute->running = 0;
}

}
