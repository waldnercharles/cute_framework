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

#ifndef CUTE_GFX_H
#define CUTE_GFX_H

#include "cute_defines.h"
#include "cute_error.h"
#include "cute_app.h"
#include "cute_color.h"

#include "sokol/sokol_gfx.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef uint64_t cf_texture_t;

CUTE_API cf_texture_t CUTE_CALL cf_texture_make(cf_pixel_t* pixels, int w, int h);
CUTE_API cf_texture_t CUTE_CALL cf_texture_make2(cf_pixel_t* pixels, int w, int h, sg_wrap mode /* = SG_WRAP_REPEAT */, sg_filter filter /* = SG_FILTER_NEAREST */);
CUTE_API void CUTE_CALL cf_texture_destroy(cf_texture_t texture);

typedef struct cf_matrix_t
{
	float data[16];
} cf_matrix_t;

CUTE_API cf_matrix_t CUTE_CALL cf_matrix_identity();
CUTE_API cf_matrix_t CUTE_CALL cf_matrix_ortho_2d(float w, float h, float x, float y);

typedef struct cf_gfx_buffer_t
{
	int stride;
	int buffer_number;
	int offset;
	sg_buffer buffer[3];
} cf_gfx_buffer_t;

typedef struct cf_triple_buffer_t
{
	cf_gfx_buffer_t vbuf;
	cf_gfx_buffer_t ibuf;

	#ifdef CUTE_CPP
	CUTE_INLINE void advance();
	CUTE_INLINE sg_bindings bind();
	#endif // CUTE_CPP

} cf_triple_buffer_t;

CUTE_API CUTE_INLINE void CUTE_CALL cf_triple_buffer_advance(cf_triple_buffer_t* buffer)
{
	++buffer->vbuf.buffer_number; buffer->vbuf.buffer_number %= 3;
	++buffer->ibuf.buffer_number; buffer->ibuf.buffer_number %= 3;
}

CUTE_API CUTE_INLINE sg_bindings CUTE_CALL cf_triple_buffer_bind(cf_triple_buffer_t* buffer)
{
	sg_bindings bind = { 0 };
	bind.vertex_buffers[0] = buffer->vbuf.buffer[buffer->vbuf.buffer_number];
	bind.vertex_buffer_offsets[0] = buffer->vbuf.offset;
	bind.index_buffer = buffer->ibuf.buffer[buffer->ibuf.buffer_number];
	bind.index_buffer_offset = buffer->ibuf.offset;
	return bind;
}


CUTE_API cf_triple_buffer_t CUTE_CALL cf_triple_buffer_make(int vertex_data_size, int vertex_stride, int index_count /* = 0 */, int index_stride /* = 0 */);
CUTE_API cf_error_t CUTE_CALL cf_triple_buffer_append(cf_triple_buffer_t* buffer, int vertex_count, const void* vertices, int index_count /* = 0 */, const void* indices /* = NULL */);

#ifdef __cplusplus
}
#endif // __cplusplus

#ifdef CUTE_CPP

CUTE_INLINE void cf_triple_buffer_t::advance()
{
	cf_triple_buffer_advance(this);
}

CUTE_INLINE sg_bindings cf_triple_buffer_t::bind()
{
	return cf_triple_buffer_bind(this);
}


namespace cute
{
using texture_t = uint64_t;


CUTE_INLINE texture_t  texture_make(pixel_t* pixels, int w, int h, sg_wrap mode = SG_WRAP_REPEAT, sg_filter filter = SG_FILTER_NEAREST) { return cf_texture_make2(pixels, w, h, mode, filter); }
CUTE_INLINE void  texture_destroy(texture_t texture) { cf_texture_destroy(texture); }

using matrix_t = cf_matrix_t;

CUTE_INLINE matrix_t  matrix_identity() { return cf_matrix_identity(); }
CUTE_INLINE matrix_t  matrix_ortho_2d(float w, float h, float x, float y) { return cf_matrix_ortho_2d(w, h, x, y); }

using triple_buffer_t = cf_triple_buffer_t;

CUTE_INLINE triple_buffer_t  triple_buffer_make(int vertex_data_size, int vertex_stride, int index_count = 0, int index_stride = 0)
{
	return cf_triple_buffer_make(vertex_data_size, vertex_stride, index_count, index_stride);
}
CUTE_INLINE error_t triple_buffer_append(triple_buffer_t* buffer, int vertex_count, const void* vertices, int index_count = 0, const void* indices = NULL)
{
	return cf_triple_buffer_append(buffer, vertex_count, vertices, index_count, indices);
}

}

#endif // CUTE_CPP


#endif // CUTE_GFX_H
