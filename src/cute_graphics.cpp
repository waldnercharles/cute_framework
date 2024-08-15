/*
	Cute Framework
	Copyright (C) 2024 Randy Gaul https://randygaul.github.io/

	This software is dual-licensed with zlib or Unlicense, check LICENSE.txt for more info
*/

#include <cute_defines.h>
#include <cute_c_runtime.h>
#include <cute_graphics.h>
#include <cute_file_system.h>

#include <internal/cute_alloc_internal.h>
#include <internal/cute_app_internal.h>
#include <internal/cute_graphics_internal.h>

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <SPIRV/GlslangToSpv.h>
#define SDL_GPU_SHADERCROSS_IMPLEMENTATION
#define SDL_GPU_SHADERCROSS_STATIC
#include <SDL_gpu_shadercross/SDL_gpu_shadercross.h>
#include <SDL_gpu_shadercross/spirv.h>
#include <SPIRV-Reflect/spirv_reflect.h>

struct CF_CanvasInternal;
static CF_CanvasInternal* s_canvas = NULL;
static CF_CanvasInternal* s_default_canvas = NULL;

#include <float.h>

using namespace Cute;

const char* s_gamma = R"(
vec4 gamma(vec4 c)
{
	return vec4(pow(abs(c.rgb), vec3(1.0/2.2)), c.a);
}

vec4 de_gamma(vec4 c)
{
	return vec4(pow(abs(c.rgb), vec3(2.2)), c.a);
}
)";

const char* s_blend = R"(
// HSV <-> RGB from : http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
// And https://www.shadertoy.com/view/MsS3Wc

vec3 rgb_to_hsv(vec3 c)
{
	vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
	vec4 p = c.g < c.b ? vec4(c.bg, K.wz) : vec4(c.gb, K.xy);
	vec4 q = c.r < p.x ? vec4(p.xyw, c.r) : vec4(c.r, p.yzx);
	float d = q.x - min(q.w, q.y);
	float e = 1.0e-10;
	return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv_to_rgb(vec3 c)
{
	vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0);
	rgb = rgb*rgb*(3.0-2.0*rgb);
	return c.z * mix(vec3(1.0), rgb, c.y);
}

vec3 hue(vec3 base, vec3 tint)
{
	base = rgb_to_hsv(base);
	tint = rgb_to_hsv(tint);
	return hsv_to_rgb(vec3(tint.r, base.gb));
}

vec4 hue(vec4 base, vec4 tint)
{
	return vec4(hue(base.rgb, tint.rgb), base.a);
}

float overlay(float base, float blend)
{
	return (base <= 0.5) ? 2*base * blend : 1-2*(1-base) * (1-blend);
}

vec3 overlay(vec3 base, vec3 blend)
{
	return vec3(overlay(base.r, blend.r), overlay(base.g, blend.g), overlay(base.b, blend.b));
}

vec4 overlay(vec4 base, vec4 blend)
{
	return vec4(overlay(base.rgb, blend.rgb), base.a);
}

float softlight(float base, float blend)
{
	if (blend <= 0.5) return base - (1-2*blend)*base*(1-base);
	else return base + (2.0 * blend - 1) * (((base <= 0.25) ? ((16.0 * base - 12.0) * base + 4.0) * base : sqrt(base)) - base);
}

vec3 softlight(vec3 base, vec3 blend)
{
	return vec3(softlight(base.r, blend.r), softlight(base.g, blend.g), softlight(base.b, blend.b));
}

vec4 softlight(vec4 base, vec4 blend)
{
	return vec4(softlight(base.rgb, blend.rgb), base.a);
}
)";

const char* s_distance = R"(
float safe_div(float a, float b)
{
	return b == 0.0 ? 0.0 : a / b;
}

float safe_len(vec2 v)
{
	float d = dot(v,v);
	return d == 0.0 ? 0.0 : sqrt(d);
}

vec2 safe_norm(vec2 v, float l)
{
	return mix(vec2(0), v / l, l == 0.0 ? 0.0 : 1.0);
}

vec2 skew(vec2 v)
{
	return vec2(-v.y, v.x);
}

float det2(vec2 a, vec2 b)
{
	return a.x * b.y - a.y * b.x;
}

float sdf_stroke(float d)
{
	return abs(d) - v_stroke;
}

float sdf_intersect(float a, float b)
{
	return max(a, b);
}

float sdf_union(float a, float b)
{
	return min(a, b);
}

float sdf_subtract(float d0, float d1)
{
	return max(d0, -d1);
}

float dd(float d)
{
	return length(vec2(dFdx(d), dFdy(d)));
}

vec4 sdf(vec4 a, vec4 b, float d)
{
	float wire_d = sdf_stroke(d);
	vec4 stroke_aa = mix(b, a, smoothstep(0.0, v_aa, wire_d));
	vec4 stroke_no_aa = wire_d <= 0.0 ? b : a;

	vec4 fill_aa = mix(b, a, smoothstep(0.0, v_aa, d));
	vec4 fill_no_aa = clamp(d, -1.0, 1.0) <= 0.0 ? b : a;

	vec4 stroke = mix(stroke_aa, stroke_aa, v_aa > 0.0 ? 1.0 : 0.0);
	vec4 fill = mix(fill_no_aa, fill_aa, v_aa > 0.0 ? 1.0 : 0.0);

	result = mix(stroke, fill, v_fill);
	return result;
}

float distance_aabb(vec2 p, vec2 he)
{
	vec2 d = abs(p) - he;
	return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float distance_box(vec2 p, vec2 c, vec2 he, vec2 u)
{
	mat2 m = transpose(mat2(u, skew(u)));
	p = p - c;
	p = m * p;
	return distance_aabb(p, he);
}

// Referenced from: https://www.shadertoy.com/view/3tdSDj
float distance_segment(vec2 p, vec2 a, vec2 b)
{
	vec2 n = b - a;
	vec2 pa = p - a;
	float d = safe_div(dot(pa,n), dot(n,n));
	float h = clamp(d, 0.0, 1.0);
	return safe_len(pa - h * n);
}

// Referenced from: https://www.shadertoy.com/view/XsXSz4
float distance_triangle(vec2 p, vec2 a, vec2 b, vec2 c)
{
	vec2 e0 = b - a;
	vec2 e1 = c - b;
	vec2 e2 = a - c;

	vec2 v0 = p - a;
	vec2 v1 = p - b;
	vec2 v2 = p - c;

	vec2 pq0 = v0 - e0 * clamp(safe_div(dot(v0, e0), dot(e0, e0)), 0.0, 1.0);
	vec2 pq1 = v1 - e1 * clamp(safe_div(dot(v1, e1), dot(e1, e1)), 0.0, 1.0);
	vec2 pq2 = v2 - e2 * clamp(safe_div(dot(v2, e2), dot(e2, e2)), 0.0, 1.0);

	float s = det2(e0, e2);
	vec2 d = min(min(vec2(dot(pq0, pq0), s * det2(v0, e0)),
						vec2(dot(pq1, pq1), s * det2(v1, e1))),
						vec2(dot(pq2, pq2), s * det2(v2, e2)));

	return -sqrt(d.x) * sign(d.y);
}
)";

const char* s_smooth_uv = R"(
vec2 smooth_uv(vec2 uv, vec2 texture_size)
{
	vec2 pixel = uv * texture_size;
	vec2 seam = floor(pixel + 0.5);
	pixel = seam + clamp((pixel - seam) / fwidth(pixel), -0.5, 0.5);
	return pixel / texture_size;
}
)";

const char* s_shader_stub = R"(
vec4 shader(vec4 color, vec2 pos, vec2 atlas_uv, vec2 screen_uv, vec4 params)
{
	return color;
}
)";

const char* s_draw_vs = R"(
layout (location = 0) in vec2 in_pos;
layout (location = 1) in vec2 in_posH;
layout (location = 2) in vec2 in_a;
layout (location = 3) in vec2 in_b;
layout (location = 4) in vec2 in_c;
layout (location = 5) in vec2 in_uv;
layout (location = 6) in vec4 in_col;
layout (location = 7) in float in_radius;
layout (location = 8) in float in_stroke;
layout (location = 9) in float in_aa;
layout (location = 10) in vec4 in_params;
layout (location = 11) in vec4 in_user_params;

layout (location = 0) out vec2 v_pos;
layout (location = 1) out vec2 v_a;
layout (location = 2) out vec2 v_b;
layout (location = 3) out vec2 v_c;
layout (location = 4) out vec2 v_uv;
layout (location = 5) out vec4 v_col;
layout (location = 6) out float v_radius;
layout (location = 7) out float v_stroke;
layout (location = 8) out float v_aa;
layout (location = 9) out float v_type;
layout (location = 10) out float v_alpha;
layout (location = 11) out float v_fill;
layout (location = 12) out vec2 v_posH;
layout (location = 13) out vec4 v_user;

void main()
{
	v_pos = in_pos;
	v_a = in_a;
	v_b = in_b;
	v_c = in_c;
	v_uv = in_uv;
	v_col = in_col;
	v_radius = in_radius;
	v_stroke = in_stroke;
	v_aa = in_aa;
	v_type = in_params.r;
	v_alpha = in_params.g;
	v_fill = in_params.b;
	// = in_params.a;

	vec4 posH = vec4(in_posH, 0, 1);
	gl_Position = posH;
	v_posH = in_posH;
	v_user = in_user_params;
}
)";

const char* s_draw_fs = R"(
layout (location = 0) in vec2 v_pos;
layout (location = 1) in vec2 v_a;
layout (location = 2) in vec2 v_b;
layout (location = 3) in vec2 v_c;
layout (location = 4) in vec2 v_uv;
layout (location = 5) in vec4 v_col;
layout (location = 6) in float v_radius;
layout (location = 7) in float v_stroke;
layout (location = 8) in float v_aa;
layout (location = 9) in float v_type;
layout (location = 10) in float v_alpha;
layout (location = 11) in float v_fill;
layout (location = 12) in vec2 v_posH;
layout (location = 13) in vec4 v_user;

out vec4 result;

layout (set = 2, binding = 0) uniform sampler2D u_image;

layout (set = 3, binding = 0) uniform uniform_block {
	vec2 u_texture_size;
};

#include "blend.shd"
#include "gamma.shd"
#include "smooth_uv.shd"
#include "distance.shd"
#include "shader_stub.shd"

void main()
{
	bool is_sprite  = v_type >= (0.0/255.0) && v_type < (0.5/255.0);
	bool is_text    = v_type >  (0.5/255.0) && v_type < (1.5/255.0);
	bool is_box     = v_type >  (1.5/255.0) && v_type < (2.5/255.0);
	bool is_seg     = v_type >  (2.5/255.0) && v_type < (3.5/255.0);
	bool is_tri     = v_type >  (3.5/255.0) && v_type < (4.5/255.0);
	bool is_tri_sdf = v_type >  (4.5/255.0) && v_type < (5.5/255.0);

	// Traditional sprite/text/tri cases.
	vec4 c = vec4(0);
	c = !(is_sprite && is_text) ? de_gamma(texture(u_image, smooth_uv(v_uv, u_texture_size))) : c;
	c = is_sprite ? gamma(overlay(c, v_col)) : c;
	c = is_text ? v_col * c.a : c;
	c = is_tri ? v_col : c;

	// SDF cases.
	float d = 0;
	if (is_box) {
		d = distance_box(v_pos, v_a, v_b, v_c);
	} else if (is_seg) {
		d = distance_segment(v_pos, v_a, v_b);
		d = min(d, distance_segment(v_pos, v_b, v_c));
	} else if (is_tri_sdf) {
		d = distance_triangle(v_pos, v_a, v_b, v_c);
	}
	c = (!is_sprite && !is_text && !is_tri) ? sdf(c, v_col, d - v_radius) : c;

	c *= v_alpha;
	vec2 screen_position = (v_posH + vec2(1,1)) * 0.5;
	c = shader(c, v_pos, v_uv, screen_position, v_user);
	if (c.a == 0) discard;
	result = c;
}
)";

const char* s_base_vs = R"(
layout (location = 0) in vec2 in_posH;

void main()
{
	vec4 posH = vec4(in_posH, 0, 1);
	gl_Position = posH;
}
)";

const char* s_base_fs = R"(
layout (location = 0) out vec4 result;

void main()
{
	result = vec4(1);
}
)";

typedef enum CF_ShaderInputFormat
{
	CF_SHADER_INPUT_FORMAT_UNKNOWN,
	CF_SHADER_INPUT_FORMAT_UINT,
	CF_SHADER_INPUT_FORMAT_INT,
	CF_SHADER_INPUT_FORMAT_FLOAT,
	CF_SHADER_INPUT_FORMAT_UVEC2,
	CF_SHADER_INPUT_FORMAT_IVEC2,
	CF_SHADER_INPUT_FORMAT_VEC2,
	CF_SHADER_INPUT_FORMAT_UVEC3,
	CF_SHADER_INPUT_FORMAT_IVEC3,
	CF_SHADER_INPUT_FORMAT_VEC3,
	CF_SHADER_INPUT_FORMAT_UVEC4,
	CF_SHADER_INPUT_FORMAT_IVEC4,
	CF_SHADER_INPUT_FORMAT_VEC4,
} CF_ShaderInputFormat;

static bool s_is_compatible(CF_ShaderInputFormat input_format, CF_VertexFormat vertex_format)
{
	switch (input_format)
	{
	case CF_SHADER_INPUT_FORMAT_UINT:
		return vertex_format == CF_VERTEX_FORMAT_UINT;

	case CF_SHADER_INPUT_FORMAT_FLOAT:
		return vertex_format == CF_VERTEX_FORMAT_FLOAT;

	case CF_SHADER_INPUT_FORMAT_VEC2:
		return vertex_format == CF_VERTEX_FORMAT_FLOAT2;

	case CF_SHADER_INPUT_FORMAT_VEC3:
		return vertex_format == CF_VERTEX_FORMAT_FLOAT3;

	case CF_SHADER_INPUT_FORMAT_VEC4:
		return vertex_format == CF_VERTEX_FORMAT_FLOAT4 || vertex_format == CF_VERTEX_FORMAT_UBYTE4N || vertex_format == CF_VERTEX_FORMAT_UBYTE4;

	case CF_SHADER_INPUT_FORMAT_UVEC4:
		return vertex_format == CF_VERTEX_FORMAT_UBYTE4N || vertex_format == CF_VERTEX_FORMAT_UBYTE4;

	case CF_SHADER_INPUT_FORMAT_IVEC4:
		return vertex_format == CF_VERTEX_FORMAT_SHORT4 || vertex_format == CF_VERTEX_FORMAT_SHORT4N;

	case CF_SHADER_INPUT_FORMAT_IVEC2:
		return vertex_format == CF_VERTEX_FORMAT_SHORT2 || vertex_format == CF_VERTEX_FORMAT_SHORT2N;

	case CF_SHADER_INPUT_FORMAT_UVEC2:
		return vertex_format == CF_VERTEX_FORMAT_HALFVECTOR2;

	// Not supported.
	case CF_SHADER_INPUT_FORMAT_UVEC3:
	case CF_SHADER_INPUT_FORMAT_IVEC3:
	case CF_SHADER_INPUT_FORMAT_UNKNOWN:
	default:
		return false;
	}
}


static SDL_GpuVertexElementFormat s_wrap(CF_VertexFormat format)
{
	switch (format) {
	default: return SDL_GPU_VERTEXELEMENTFORMAT_UINT;
	case CF_VERTEX_FORMAT_UINT: return SDL_GPU_VERTEXELEMENTFORMAT_UINT;
	case CF_VERTEX_FORMAT_FLOAT: return SDL_GPU_VERTEXELEMENTFORMAT_FLOAT;
	case CF_VERTEX_FORMAT_FLOAT2: return SDL_GPU_VERTEXELEMENTFORMAT_VECTOR2;
	case CF_VERTEX_FORMAT_FLOAT3: return SDL_GPU_VERTEXELEMENTFORMAT_VECTOR3;
	case CF_VERTEX_FORMAT_FLOAT4: return SDL_GPU_VERTEXELEMENTFORMAT_VECTOR4;
	case CF_VERTEX_FORMAT_UBYTE4N: return SDL_GPU_VERTEXELEMENTFORMAT_COLOR;
	case CF_VERTEX_FORMAT_UBYTE4: return SDL_GPU_VERTEXELEMENTFORMAT_BYTE4;
	case CF_VERTEX_FORMAT_SHORT2: return SDL_GPU_VERTEXELEMENTFORMAT_SHORT2;
	case CF_VERTEX_FORMAT_SHORT4: return SDL_GPU_VERTEXELEMENTFORMAT_SHORT4;
	case CF_VERTEX_FORMAT_SHORT2N: return SDL_GPU_VERTEXELEMENTFORMAT_NORMALIZEDSHORT2;
	case CF_VERTEX_FORMAT_SHORT4N: return SDL_GPU_VERTEXELEMENTFORMAT_NORMALIZEDSHORT4;
	case CF_VERTEX_FORMAT_HALFVECTOR2: return SDL_GPU_VERTEXELEMENTFORMAT_HALFVECTOR2;
	case CF_VERTEX_FORMAT_HALFVECTOR4: return SDL_GPU_VERTEXELEMENTFORMAT_HALFVECTOR4;
	}
}

static int s_uniform_size(CF_UniformType type)
{
	switch (type) {
	case CF_UNIFORM_TYPE_FLOAT:  return 4;
	case CF_UNIFORM_TYPE_FLOAT2: return 8;
	case CF_UNIFORM_TYPE_FLOAT4: return 16;
	case CF_UNIFORM_TYPE_INT:    return 4;
	case CF_UNIFORM_TYPE_INT2:   return 8;
	case CF_UNIFORM_TYPE_INT4:   return 16;
	case CF_UNIFORM_TYPE_MAT4:   return 64;
	default:                     return 0;
	}
}

#define CF_MAX_SHADER_INPUTS (32)

struct CF_UniformBlockMember
{
	const char* name;
	CF_UniformType type;
	int array_element_count;
	int size; // In bytes. If an array, it's the size in bytes of the whole array.
	int offset;
};

struct CF_Uniform
{
	const char* name;
	CF_UniformType type;
	int array_length;
	void* data;
	int size;
};

struct CF_MaterialTex
{
	const char* name;
	CF_Texture handle;
};

struct CF_MaterialState
{
	Array<CF_Uniform> uniforms;
	Array<CF_MaterialTex> textures;
};

struct CF_MaterialInternal
{
	bool dirty = false;
	CF_RenderState state;
	CF_MaterialState vs;
	CF_MaterialState fs;
	CF_Arena uniform_arena;
	CF_Arena block_arena;
};

struct CF_Pipeline
{
	CF_MaterialInternal* material = NULL;
	SDL_GpuGraphicsPipeline* pip = NULL;
	CF_MeshInternal* mesh = NULL;
};

struct CF_ShaderInternal
{
	SDL_GpuShader* vs;
	SDL_GpuShader* fs;
	int input_count;
	const char* input_names[CF_MAX_SHADER_INPUTS];
	int input_locations[CF_MAX_SHADER_INPUTS];
	CF_ShaderInputFormat input_formats[CF_MAX_SHADER_INPUTS];
	int vs_block_size;
	int fs_block_size;
	Array<CF_UniformBlockMember> fs_uniform_block_members;
	Array<CF_UniformBlockMember> vs_uniform_block_members;
	Array<const char*> image_names;
	Array<CF_Pipeline> pip_cache;

	CF_INLINE int get_input_index(const char* name)
	{
		for (int i = 0; i < input_count; ++i) {
			if (input_names[i] == name) return i;
		}
		return -1;
	}

	CF_INLINE int fs_index(const char* name)
	{
		for (int i = 0; i < fs_uniform_block_members.size(); ++i) {
			if (fs_uniform_block_members[i].name == name) return i;
		}
		return -1;
	}

	CF_INLINE int vs_index(const char* name)
	{
		for (int i = 0; i < vs_uniform_block_members.size(); ++i) {
			if (vs_uniform_block_members[i].name == name) return i;
		}
		return -1;
	}
};

struct CF_Buffer
{
	int element_count;
	int size;
	//int offset; // @TODO Unused for now as SDL_Gpu has no way to stream map'd portions of buffers.
	int stride;
	SDL_GpuBuffer* buffer;
	SDL_GpuTransferBuffer* transfer_buffer;
};

struct CF_MeshInternal
{
	CF_Buffer vertices;
	CF_Buffer indices;
	int attribute_count;
	CF_VertexAttribute attributes[CF_MESH_MAX_VERTEX_ATTRIBUTES];
};

CF_BackendType cf_query_backend()
{
	return BACKEND_TYPE_D3D11;
}

CF_TextureParams cf_texture_defaults(int w, int h)
{
	CF_TextureParams params;
	params.pixel_format = CF_PIXEL_FORMAT_R8G8B8A8;
	params.filter = CF_FILTER_NEAREST;
	params.usage = CF_TEXTURE_USAGE_SAMPLER_BIT;
	params.wrap_u = CF_WRAP_MODE_REPEAT;
	params.wrap_v = CF_WRAP_MODE_REPEAT;
	params.width = w;
	params.height = h;
	params.stream = false;
	return params;
}

static SDL_GpuTextureFormat s_wrap(CF_PixelFormat format)
{
	switch (format) {
	case CF_PIXEL_FORMAT_INVALID:              return SDL_GPU_TEXTUREFORMAT_INVALID;
	case CF_PIXEL_FORMAT_R8G8B8A8:             return SDL_GPU_TEXTUREFORMAT_R8G8B8A8;
	case CF_PIXEL_FORMAT_B8G8R8A8:             return SDL_GPU_TEXTUREFORMAT_B8G8R8A8;
	case CF_PIXEL_FORMAT_B5G6R5:               return SDL_GPU_TEXTUREFORMAT_B5G6R5;
	case CF_PIXEL_FORMAT_B5G5R5A1:             return SDL_GPU_TEXTUREFORMAT_B5G5R5A1;
	case CF_PIXEL_FORMAT_B4G4R4A4:             return SDL_GPU_TEXTUREFORMAT_B4G4R4A4;
	case CF_PIXEL_FORMAT_R10G10B10A2:          return SDL_GPU_TEXTUREFORMAT_R10G10B10A2;
	case CF_PIXEL_FORMAT_R16G16:               return SDL_GPU_TEXTUREFORMAT_R16G16;
	case CF_PIXEL_FORMAT_R16G16B16A16:         return SDL_GPU_TEXTUREFORMAT_R16G16B16A16;
	case CF_PIXEL_FORMAT_R8:                   return SDL_GPU_TEXTUREFORMAT_R8;
	case CF_PIXEL_FORMAT_A8:                   return SDL_GPU_TEXTUREFORMAT_A8;
	case CF_PIXEL_FORMAT_R8_UINT:              return SDL_GPU_TEXTUREFORMAT_R8_UINT;
	case CF_PIXEL_FORMAT_R8G8_UINT:            return SDL_GPU_TEXTUREFORMAT_R8G8_UINT;
	case CF_PIXEL_FORMAT_R8G8B8A8_UINT:        return SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UINT;
	case CF_PIXEL_FORMAT_R16_UINT:             return SDL_GPU_TEXTUREFORMAT_R16_UINT;
	case CF_PIXEL_FORMAT_R16G16_UINT:          return SDL_GPU_TEXTUREFORMAT_R16G16_UINT;
	case CF_PIXEL_FORMAT_R16G16B16A16_UINT:    return SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UINT;
	case CF_PIXEL_FORMAT_BC1:                  return SDL_GPU_TEXTUREFORMAT_BC1;
	case CF_PIXEL_FORMAT_BC2:                  return SDL_GPU_TEXTUREFORMAT_BC2;
	case CF_PIXEL_FORMAT_BC3:                  return SDL_GPU_TEXTUREFORMAT_BC3;
	case CF_PIXEL_FORMAT_BC7:                  return SDL_GPU_TEXTUREFORMAT_BC7;
	case CF_PIXEL_FORMAT_R8G8_SNORM:           return SDL_GPU_TEXTUREFORMAT_R8G8_SNORM;
	case CF_PIXEL_FORMAT_R8G8B8A8_SNORM:       return SDL_GPU_TEXTUREFORMAT_R8G8B8A8_SNORM;
	case CF_PIXEL_FORMAT_R16_SFLOAT:           return SDL_GPU_TEXTUREFORMAT_R16_SFLOAT;
	case CF_PIXEL_FORMAT_R16G16_SFLOAT:        return SDL_GPU_TEXTUREFORMAT_R16G16_SFLOAT;
	case CF_PIXEL_FORMAT_R16G16B16A16_SFLOAT:  return SDL_GPU_TEXTUREFORMAT_R16G16B16A16_SFLOAT;
	case CF_PIXEL_FORMAT_R32_SFLOAT:           return SDL_GPU_TEXTUREFORMAT_R32_SFLOAT;
	case CF_PIXEL_FORMAT_R32G32_SFLOAT:        return SDL_GPU_TEXTUREFORMAT_R32G32_SFLOAT;
	case CF_PIXEL_FORMAT_R32G32B32A32_SFLOAT:  return SDL_GPU_TEXTUREFORMAT_R32G32B32A32_SFLOAT;
	case CF_PIXEL_FORMAT_R8G8B8A8_SRGB:        return SDL_GPU_TEXTUREFORMAT_R8G8B8A8_SRGB;
	case CF_PIXEL_FORMAT_B8G8R8A8_SRGB:        return SDL_GPU_TEXTUREFORMAT_B8G8R8A8_SRGB;
	case CF_PIXEL_FORMAT_BC3_SRGB:             return SDL_GPU_TEXTUREFORMAT_BC3_SRGB;
	case CF_PIXEL_FORMAT_BC7_SRGB:             return SDL_GPU_TEXTUREFORMAT_BC7_SRGB;
	case CF_PIXEL_FORMAT_D16_UNORM:            return SDL_GPU_TEXTUREFORMAT_D16_UNORM;
	case CF_PIXEL_FORMAT_D24_UNORM:            return SDL_GPU_TEXTUREFORMAT_D24_UNORM;
	case CF_PIXEL_FORMAT_D32_SFLOAT:           return SDL_GPU_TEXTUREFORMAT_D32_SFLOAT;
	case CF_PIXEL_FORMAT_D24_UNORM_S8_UINT:    return SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT;
	case CF_PIXEL_FORMAT_D32_SFLOAT_S8_UINT:   return SDL_GPU_TEXTUREFORMAT_D32_SFLOAT_S8_UINT;
	default:                                   return SDL_GPU_TEXTUREFORMAT_INVALID;
	}
}

static SDL_GpuFilter s_wrap(CF_Filter filter)
{
	switch (filter) {
	default: return SDL_GPU_FILTER_NEAREST;
	case CF_FILTER_NEAREST: return SDL_GPU_FILTER_NEAREST;
	case CF_FILTER_LINEAR: return SDL_GPU_FILTER_LINEAR;
	}
}

static SDL_GpuSamplerAddressMode s_wrap(CF_WrapMode mode)
{
	switch (mode)
	{
	case CF_WRAP_MODE_REPEAT:           return SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
	case CF_WRAP_MODE_CLAMP_TO_EDGE:    return SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
	case CF_WRAP_MODE_MIRRORED_REPEAT:  return SDL_GPU_SAMPLERADDRESSMODE_MIRRORED_REPEAT;
	default:                            return SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
	}
}

CF_Texture cf_make_texture(CF_TextureParams params)
{
	SDL_GpuTextureCreateInfo tex_info = SDL_GpuTextureCreateInfoDefaults(params.width, params.height);
	tex_info.width = (Uint32)params.width;
	tex_info.height = (Uint32)params.height;
	tex_info.format = s_wrap(params.pixel_format);
	tex_info.usageFlags = params.usage;
	SDL_GpuTexture* tex = SDL_GpuCreateTexture(app->device, &tex_info);
	if (!tex) return { 0 };

	SDL_GpuSamplerCreateInfo sampler_info = SDL_GpuSamplerCreateInfoDefaults();
	sampler_info.minFilter = s_wrap(params.filter);
	sampler_info.magFilter = s_wrap(params.filter);
	sampler_info.addressModeU = s_wrap(params.wrap_u);
	sampler_info.addressModeV = s_wrap(params.wrap_v);
	SDL_GpuSampler* sampler = SDL_GpuCreateSampler(app->device, &sampler_info);
	if (!sampler) {
		SDL_GpuReleaseTexture(app->device, tex);
		return { 0 };
	}

	SDL_GpuTransferBuffer* buf = NULL;
	if (params.stream) {
		int texel_size = (int)SDL_GpuTextureFormatTexelBlockSize(tex_info.format);
		SDL_GpuTransferBuffer* buf = SDL_GpuCreateTransferBuffer(app->device, SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, texel_size * params.width * params.height);
	}

	CF_TextureInternal* tex_internal = CF_NEW(CF_TextureInternal);
	tex_internal->w = params.width;
	tex_internal->h = params.height;
	tex_internal->filter = sampler_info.minFilter;
	tex_internal->tex = tex;
	tex_internal->buf = buf;
	tex_internal->sampler = sampler;
	tex_internal->format = tex_info.format;
	CF_Texture result;
	result.id = { (uint64_t)tex_internal };
	return result;
}

void cf_destroy_texture(CF_Texture texture_handle)
{
	CF_TextureInternal* tex = (CF_TextureInternal*)texture_handle.id;
	SDL_GpuReleaseTexture(app->device, tex->tex);
	SDL_GpuReleaseSampler(app->device, tex->sampler);
	if (tex->buf) SDL_GpuReleaseTransferBuffer(app->device, tex->buf);
	CF_FREE(tex);
}

static SDL_GpuTextureLocation SDL_GpuTextureLocationDefaults(CF_TextureInternal* tex, float x, float y)
{
	SDL_GpuTextureLocation location;
	CF_MEMSET(&location, 0, sizeof(location));
	location.textureSlice.texture = tex->tex;
	location.x = (Uint32)(x * tex->w);
	location.y = (Uint32)(y * tex->h);
	return location;
}

void cf_update_texture(CF_Texture texture_handle, void* data, int size)
{
	CF_TextureInternal* tex = (CF_TextureInternal*)texture_handle.id;

	// Copy bytes over to the driver.
	SDL_GpuTransferBuffer* buf = tex->buf;
	if (!buf) {
		buf = SDL_GpuCreateTransferBuffer(app->device, SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, size);
	}
	uint8_t* p;
	SDL_GpuMapTransferBuffer(app->device, buf, tex->buf ? true : false, (void**)&p);
	SDL_memcpy(p, data, size);
	SDL_GpuUnmapTransferBuffer(app->device, buf);

	// Tell the driver to upload the bytes to the GPU.
	SDL_GpuCommandBuffer* cmd = app->cmd;
	SDL_GpuCopyPass* pass = SDL_GpuBeginCopyPass(cmd);
	SDL_GpuTextureTransferInfo src;
	src.transferBuffer = buf;
	src.offset = 0;
	src.imagePitch = tex->w;
	src.imageHeight = tex->h;
	SDL_GpuTextureRegion dst = SDL_GpuTextureRegionDefaults(tex, 0, 0, 1, 1);
	SDL_GpuUploadToTexture(pass, &src, &dst, tex->buf ? true : false);
	SDL_GpuEndCopyPass(pass);
	if (!tex->buf) {
		SDL_GpuReleaseTransferBuffer(app->device, buf);
	}
}

static void s_shader_directory(Path path)
{
	Array<Path> dir = Directory::enumerate(app->shader_directory + path);
	for (int i = 0; i < dir.size(); ++i) {
		Path p = app->shader_directory + path + dir[i];
		if (p.is_directory()) {
			cf_shader_directory(p);
		} else {
			CF_Stat stat;
			fs_stat(p, &stat);
			String ext = p.ext();
			if (ext == ".vs" || ext == ".fs" || ext == ".shd") {
				// Exclude app->shader_directory for easier lookups.
				// e.g. app->shader_directory is "/shaders" and contains
				// "/shaders/my_shader.shd", the user needs to only reference it by:
				// "my_shader.shd".
				CF_ShaderFileInfo info;
				info.stat = stat;
				info.path = sintern(p);
				const char* key = sintern(path + dir[i]);
				app->shader_file_infos.add(key, info);
			}
		}
	}
}

void cf_shader_directory(const char* path)
{
	CF_ASSERT(!app->shader_directory_set);
	if (app->shader_directory_set) return;
	app->shader_directory_set = true;
	app->shader_directory = path;
	s_shader_directory("/");
}

void cf_shader_on_changed(void (*on_changed_fn)(const char* path, void* udata), void* udata)
{
	// @TODO Implement in app.
	app->on_shader_changed_fn = on_changed_fn;
	app->on_shader_changed_udata = udata;
}

const dyna uint8_t* cf_compile_shader_to_bytecode(const char* shader_src, CF_ShaderStage cf_stage)
{
	EShLanguage stage = EShLangVertex;
	switch (cf_stage) {
	default: CF_ASSERT(false); break; // No valid stage provided.
	case CF_SHADER_STAGE_VERTEX: stage = EShLangVertex; break;
	case CF_SHADER_STAGE_FRAGMENT: stage = EShLangFragment; break;
	}

	glslang::TShader shader(stage);

	const char* shader_strings[1];
	shader_strings[0] = shader_src;
	shader.setStrings(shader_strings, 1);

	shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 450);
	shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_2);
	shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_6);
	shader.setEntryPoint("main");
	shader.setSourceEntryPoint("main");
	shader.setAutoMapLocations(true);
	shader.setAutoMapBindings(true);

	if (!shader.parse(GetDefaultResources(), 450, false, EShMsgDefault)) {
		fprintf(stderr, "GLSL parsing failed...\n");
		fprintf(stderr, "%s\n\n%s\n", shader.getInfoLog(), shader.getInfoDebugLog());
		return NULL;
	}

	glslang::TProgram program;
	program.addShader(&shader);

	if (!program.link(EShMsgDefault)) {
		fprintf(stderr, "GLSL linking failed...\n");
		fprintf(stderr, "%s\n\n%s\n", program.getInfoLog(), program.getInfoDebugLog());
		return NULL;
	}

	std::vector<uint32_t> spirv;
	glslang::SpvOptions options;
	options.generateDebugInfo = false;
	options.stripDebugInfo = false;
	options.disableOptimizer = false;
	options.optimizeSize = false;
	options.disassemble = false;
	options.validate = false;
	glslang::GlslangToSpv(*program.getIntermediate(stage), spirv, &options);

	dyna uint8_t* bytecode = NULL;
	int size = (int)(sizeof(uint32_t) * spirv.size());
	afit(bytecode, size);
	CF_MEMCPY(bytecode, spirv.data(), size);
	alen(bytecode) = size;

	return bytecode;
}

static CF_INLINE SDL_GpuShaderStage s_wrap(CF_ShaderStage stage)
{
	switch (stage) {
	case CF_SHADER_STAGE_VERTEX: return SDL_GPU_SHADERSTAGE_VERTEX;
	case CF_SHADER_STAGE_FRAGMENT: return SDL_GPU_SHADERSTAGE_FRAGMENT;
	default: return SDL_GPU_SHADERSTAGE_VERTEX;
	}
}

static CF_ShaderInputFormat s_wrap(SpvReflectFormat format)
{
	switch (format) {
	case SPV_REFLECT_FORMAT_UNDEFINED:           return CF_SHADER_INPUT_FORMAT_UNKNOWN;
	case SPV_REFLECT_FORMAT_R32_UINT:            return CF_SHADER_INPUT_FORMAT_UINT;
	case SPV_REFLECT_FORMAT_R32_SINT:            return CF_SHADER_INPUT_FORMAT_INT;
	case SPV_REFLECT_FORMAT_R32_SFLOAT:          return CF_SHADER_INPUT_FORMAT_FLOAT;
	case SPV_REFLECT_FORMAT_R32G32_UINT:         return CF_SHADER_INPUT_FORMAT_UVEC2;
	case SPV_REFLECT_FORMAT_R32G32_SINT:         return CF_SHADER_INPUT_FORMAT_IVEC2;
	case SPV_REFLECT_FORMAT_R32G32_SFLOAT:       return CF_SHADER_INPUT_FORMAT_VEC2;
	case SPV_REFLECT_FORMAT_R32G32B32_UINT:      return CF_SHADER_INPUT_FORMAT_UVEC3;
	case SPV_REFLECT_FORMAT_R32G32B32_SINT:      return CF_SHADER_INPUT_FORMAT_IVEC3;
	case SPV_REFLECT_FORMAT_R32G32B32_SFLOAT:    return CF_SHADER_INPUT_FORMAT_VEC3;
	case SPV_REFLECT_FORMAT_R32G32B32A32_UINT:   return CF_SHADER_INPUT_FORMAT_UVEC4;
	case SPV_REFLECT_FORMAT_R32G32B32A32_SINT:   return CF_SHADER_INPUT_FORMAT_IVEC4;
	case SPV_REFLECT_FORMAT_R32G32B32A32_SFLOAT: return CF_SHADER_INPUT_FORMAT_VEC4;
	default: return CF_SHADER_INPUT_FORMAT_UNKNOWN;
	}
}

static CF_UniformType s_uniform_type(SpvReflectTypeDescription* type_desc)
{
	switch (type_desc->op) {
	case SpvOpTypeFloat: return CF_UNIFORM_TYPE_FLOAT;
	case SpvOpTypeInt: return CF_UNIFORM_TYPE_INT;
	case SpvOpTypeVector:
		if (type_desc->traits.numeric.scalar.width == 32) {
			if (type_desc->traits.numeric.scalar.signedness == 0) {
				switch (type_desc->traits.numeric.vector.component_count) {
					case 2: return CF_UNIFORM_TYPE_FLOAT2;
					case 4: return CF_UNIFORM_TYPE_FLOAT4;
					default: return CF_UNIFORM_TYPE_UNKNOWN;
				}
			} else {
				switch (type_desc->traits.numeric.vector.component_count) {
					case 2: return CF_UNIFORM_TYPE_INT2;
					case 4: return CF_UNIFORM_TYPE_INT4;
					default: return CF_UNIFORM_TYPE_UNKNOWN;
				}
			}
		}
		break;
	case SpvOpTypeMatrix:
		if (type_desc->traits.numeric.matrix.column_count == 4 && type_desc->traits.numeric.matrix.row_count == 4)
			return CF_UNIFORM_TYPE_MAT4;
		break;
	default:
		return CF_UNIFORM_TYPE_UNKNOWN;
	}
	return CF_UNIFORM_TYPE_UNKNOWN;
}

static SDL_GpuShader* s_compile(CF_ShaderInternal* shader_internal, const dyna uint8_t* bytecode, CF_ShaderStage stage)
{
	bool vs = stage == CF_SHADER_STAGE_VERTEX ? true : false;
	SpvReflectShaderModule module;
	spvReflectCreateShaderModule(asize(bytecode), bytecode, &module);

	// Gather up counts for samplers/textures/buffers.
	// ...SDL_Gpu needs these counts.
	uint32_t binding_count = 0;
	spvReflectEnumerateDescriptorBindings(&module, &binding_count, nullptr);
	dyna SpvReflectDescriptorBinding** bindings = NULL;
	afit(bindings, (int)binding_count);
	if (binding_count) alen(bindings) = binding_count;
	spvReflectEnumerateDescriptorBindings(&module, &binding_count, bindings);
	int sampler_count = 0;
	int storage_texture_count = 0;
	int storage_buffer_count = 0;
	int uniform_buffer_count = 0;
	for (int i = 0; i < (int)binding_count; ++i) {
		SpvReflectDescriptorBinding* binding = bindings[i];

		switch (binding->descriptor_type) {
		case SPV_REFLECT_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
		{
			shader_internal->image_names.add(sintern(binding->name));
		}    // Fall-thru.
		case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER: sampler_count++; break;
		case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE: storage_texture_count++; break;
		case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER: storage_buffer_count++; break;
		case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
		{
			uniform_buffer_count++;

			// Grab information about the uniform block.
			// ...This allows CF_Material to dynamically match uniforms to a shader.
			CF_ASSERT(sequ(binding->type_description->type_name, "uniform_block"));
			if (vs) {
				shader_internal->vs_block_size = binding->block.size;
			} else {
				shader_internal->fs_block_size = binding->block.size;
			}
			for (uint32_t i = 0; i < binding->block.member_count; ++i) {
				const SpvReflectBlockVariable* member = &binding->block.members[i];
				CF_UniformType uniform_type = s_uniform_type(member->type_description);
				CF_ASSERT(uniform_type != CF_UNIFORM_TYPE_UNKNOWN);
				int array_length = 1;
				if (member->type_description->type_flags & SPV_REFLECT_TYPE_FLAG_ARRAY && member->type_description->traits.array.dims_count > 0) {
					array_length = (int)member->type_description->traits.array.dims[0];
				}

				CF_UniformBlockMember block_member;
				block_member.name = sintern(member->name);
				block_member.type = uniform_type;
				block_member.array_element_count = array_length;
				block_member.size = s_uniform_size(block_member.type) * array_length;
				block_member.offset = (int)member->offset;
				if (vs) {
					shader_internal->vs_uniform_block_members.add(block_member);
				} else {
					shader_internal->fs_uniform_block_members.add(block_member);
				}
			}
		} break;
		}
	}
	afree(bindings);

	// Gather up type information on shader inputs.
	if (vs) {
		uint32_t input_count = 0;
		spvReflectEnumerateInputVariables(&module, &input_count, nullptr);
		CF_ASSERT(input_count <= CF_MAX_SHADER_INPUTS); // Increase `CF_MAX_SHADER_INPUTS`, or refactor the shader with less vertex attributes.
		shader_internal->input_count = input_count;
		dyna SpvReflectInterfaceVariable** inputs = NULL;
		afit(inputs, (int)input_count);
		alen(inputs) = (int)input_count;
		spvReflectEnumerateInputVariables(&module, &input_count, inputs);
		for (int i = 0; i < alen(inputs); ++i) {
			SpvReflectInterfaceVariable* input = inputs[i];

			shader_internal->input_names[i] = sintern(input->name);
			shader_internal->input_locations[i] = input->location;
			shader_internal->input_formats[i] = s_wrap(input->format);
		}
		afree(inputs);
	}

	// Create the actual shader.
	SDL_GpuShaderCreateInfo shaderCreateInfo = {};
	shaderCreateInfo.codeSize = asize(bytecode);
	shaderCreateInfo.code = bytecode;
	shaderCreateInfo.entryPointName = "main";
	shaderCreateInfo.format = SDL_GPU_SHADERFORMAT_SPIRV;
	shaderCreateInfo.stage = s_wrap(stage);
	shaderCreateInfo.samplerCount = sampler_count;
	shaderCreateInfo.storageTextureCount = storage_texture_count;
	shaderCreateInfo.storageBufferCount = storage_buffer_count;
	shaderCreateInfo.uniformBufferCount = uniform_buffer_count;
	SDL_GpuShader* sdl_shader = NULL;
	if (SDL_GpuGetDriver(app->device) == SDL_GPU_DRIVER_VULKAN) {
		sdl_shader = (SDL_GpuShader*)SDL_GpuCreateShader(app->device, &shaderCreateInfo);
	} else {
		sdl_shader = (SDL_GpuShader*)SDL_CompileFromSPIRV(app->device, &shaderCreateInfo, false);
	}
	afree(bytecode);
	return sdl_shader;
}

CF_Shader cf_make_shader_from_bytecode(const dyna uint8_t* vertex_bytecode, const dyna uint8_t* fragment_bytecode)
{
	CF_ShaderInternal* shader_internal = CF_NEW(CF_ShaderInternal);
	CF_MEMSET(shader_internal, 0, sizeof(*shader_internal));

	shader_internal->vs = s_compile(shader_internal, vertex_bytecode, CF_SHADER_STAGE_VERTEX);
	shader_internal->fs = s_compile(shader_internal, fragment_bytecode, CF_SHADER_STAGE_FRAGMENT);
	CF_ASSERT(shader_internal->vs);
	CF_ASSERT(shader_internal->fs);

	CF_Shader result;
	result.id = { (uint64_t)shader_internal };
	return result;
}

// Return the index of the first #include substring that's not in a comment.
static int s_find_first_include(const char *src)
{
	const char *in = src;
	while (*in) {
		if (*in == '/' && *(in + 1) == '/') {
			in += 2;
			while (*in && *in != '\n') {
				in++;
			}
		} else if (*in == '/' && *(in + 1) == '*') {
			in += 2;
			while (*in && !(*in == '*' && *(in + 1) == '/')) {
				in++;
			}
			if (*in) {
				in += 2;
			}
		} else if (*in == '#' && !sequ(in, "#include")) {
			return (int)(in - src);
		} else {
			in++;
		}
	}
	return -1;
}

// Recursively apply #include directives in shaders.
// ...A cache is used to protect against multiple includes and infinite loops.
static String s_include_recurse(Map<const char*, const char*>& incl_protection, String shd, bool builtin, const char* user_shd)
{
	while (1) {
		int idx = s_find_first_include(shd);
		if (idx < 0) break;

		// Cut out the #include substring, and record the path.
		char* s = shd + idx + 8;
		int n = 0;
		while (*s++ != '\n') ++n;
		++n; // skip '\n'
		String path = String(shd + idx + 8, s).trim();
		shd.erase(idx, n + 8);
		path.replace("\"", "");
		path.replace("'", "");

		// Search for the shader to include.
		if (builtin || fs_file_exists(path)) {
			String ext = Path(path).ext();
			if (ext == ".vs" || ext == ".fs" || ext == ".shd") {
				String incl;
				bool found = false;
				if (builtin) {
					if (sequ(path.c_str(), "shader_stub.shd")) {
						// Inject the user shader if-applicable, stub if not.
						incl = user_shd ? user_shd : s_shader_stub;
						found = true;
					} else {
						// Builtin shaders can include other builtin shaders.
						const char* result = app->builtin_shaders.find(sintern(path));
						if (result) {
							incl = result;
							found = true;
						}
					}
				}

				// Wasn't a builtin shader, try including a user shader.
				if (!found) {
					// Processing a user-include shader.
					const char* result = fs_read_entire_file_to_memory_and_nul_terminate(path);
					if (result) {
						incl = result;
						found = true;
						cf_free((void*)result);
					}
				}

				if (found) {
					// Prevent infinite include loops.
					const char* incl_path = sintern(path);
					if (!incl_protection.has(incl_path)) {
						incl_protection.add(incl_path);
						incl = s_include_recurse(incl_protection, incl, builtin, user_shd);
						
						// Perform the actual string splice + inclusion.
						shd = String(shd, shd + idx)
							.append("// -- begin include ")
							.append(path)
							.append(" --\n")
							.append(incl)
							.append("// -- end include ")
							.append(path)
							.append(" --\n")
							.append(shd + idx);
					}
				}
			}
		}
	}
	return shd;
}

// Parse + perform include directives across shaders.
static String s_include(String shd, bool builtin, const char* user_shd)
{
	Map<const char*, const char*> incl_protection;
	return s_include_recurse(incl_protection, shd, builtin, user_shd);
}

static CF_Shader s_compile(const char* vs_src, const char* fs_src, bool builtin = false, const char* user_shd = NULL)
{
	// Support #include directives.
	String vs = s_include(vs_src, builtin, NULL);
	String fs = s_include(fs_src, builtin, user_shd);

#if 1
	printf(vs.c_str());
	printf("---\n");
	printf(fs.c_str());
	printf("---\n");
#endif

	// Compile to bytecode.
	const char* vertex = vs.c_str();
	const char* fragment = fs.c_str();
	const dyna uint8_t* vs_bytecode = cf_compile_shader_to_bytecode(vertex, CF_SHADER_STAGE_VERTEX);
	if (!vs_bytecode) {
		CF_Shader result = { 0 };
		return result;
	}
	const dyna uint8_t* fs_bytecode = cf_compile_shader_to_bytecode(fragment, CF_SHADER_STAGE_FRAGMENT);
	if (!fs_bytecode) {
		afree(vs_bytecode);
		CF_Shader result = { 0 };
		return result;
	}

	// Create the actual shader object.
	return cf_make_shader_from_bytecode(vs_bytecode, fs_bytecode);
}

void cf_load_internal_shaders()
{
#ifdef CF_RUNTIME_SHADER_COMPILATION
	glslang::InitializeProcess();
#endif

	// Map out all the builtin includable shaders.
	app->builtin_shaders.add(sintern("shader_stub.shd"), s_shader_stub);
	app->builtin_shaders.add(sintern("gamma.shd"), s_gamma);
	app->builtin_shaders.add(sintern("distance.shd"), s_distance);
	app->builtin_shaders.add(sintern("smooth_uv.shd"), s_smooth_uv);
	app->builtin_shaders.add(sintern("blend.shd"), s_blend);
	app->draw_shader = s_compile(s_draw_vs, s_draw_fs, true, NULL);
	app->basic = s_compile(s_base_vs, s_base_fs, true, NULL);
}

void cf_unload_shader_compiler()
{
#ifdef CF_RUNTIME_SHADER_COMPIILATION
	glslang::FinalizeProcess();
#endif CF_RUNTIME_SHADER_COMPIILATION
}

CF_Shader cf_make_draw_shader_internal(const char* path)
{
	Path p = Path("/") + path;
	const char* path_s = sintern(p);
	CF_ShaderFileInfo info = app->shader_file_infos.find(path_s);
	if (!info.path) return { 0 };
	const char* shd = fs_read_entire_file_to_memory_and_nul_terminate(info.path);
	if (!shd) return { 0 };
	return s_compile(s_draw_vs, s_draw_fs, true, shd);
}

CF_Shader cf_make_shader(const char* vertex_path, const char* fragment_path)
{
	// Make sure each file can be found.
	const char* vs = fs_read_entire_file_to_memory_and_nul_terminate(vertex_path);
	const char* fs = fs_read_entire_file_to_memory_and_nul_terminate(fragment_path);
	CF_ASSERT(vs);
	CF_ASSERT(fs);
	return s_compile(vs, fs);
}

CF_Shader cf_make_shader_from_source(const char* vertex_src, const char* fragment_src)
{
	return s_compile(vertex_src, fragment_src);
}

void cf_destroy_shader(CF_Shader shader_handle)
{
	CF_ShaderInternal* shd = (CF_ShaderInternal*)shader_handle.id;
	SDL_GpuReleaseShader(app->device, shd->vs);
	SDL_GpuReleaseShader(app->device, shd->fs);
	shd->~CF_ShaderInternal();
	CF_FREE(shd);
}

CF_CanvasParams cf_canvas_defaults(int w, int h)
{
	CF_CanvasParams params;
	if (w == 0 || h == 0) {
		params.name = NULL;
		params.target = { };
		params.depth_stencil_target = { };
	} else {
		params.name = NULL;
		params.target = cf_texture_defaults(w, h);
		params.target.usage |= CF_TEXTURE_USAGE_COLOR_TARGET_BIT;
		params.depth_stencil_enable = false;
		params.depth_stencil_target = cf_texture_defaults(w, h);
		params.depth_stencil_target.pixel_format = CF_PIXEL_FORMAT_D24_UNORM_S8_UINT;
		params.depth_stencil_target.usage = CF_TEXTURE_USAGE_DEPTH_STENCIL_TARGET_BIT;
	}
	return params;
}

CF_Canvas cf_make_canvas(CF_CanvasParams params)
{
	CF_CanvasInternal* canvas = (CF_CanvasInternal*)CF_CALLOC(sizeof(CF_CanvasInternal));
	if (params.target.width > 0 && params.target.height > 0) {
		canvas->cf_texture = cf_make_texture(params.target);
		if (canvas->cf_texture.id) {
			canvas->texture = ((CF_TextureInternal*)canvas->cf_texture.id)->tex;
			canvas->sampler = ((CF_TextureInternal*)canvas->cf_texture.id)->sampler;
		}
		if (params.depth_stencil_enable) {
			canvas->cf_depth_stencil = cf_make_texture(params.depth_stencil_target);
			if (canvas->cf_depth_stencil.id) {
				canvas->depth_stencil = ((CF_TextureInternal*)canvas->cf_depth_stencil.id)->tex;
			}
		} else {
			canvas->cf_depth_stencil = { 0 };
		}
	} else {
		return { 0 };
	}
	CF_Canvas result;
	result.id = (uint64_t)canvas;
	return result;
}

void cf_destroy_canvas(CF_Canvas canvas_handle)
{
	CF_CanvasInternal* canvas = (CF_CanvasInternal*)canvas_handle.id;
	cf_destroy_texture(canvas->cf_texture);
	if (canvas->depth_stencil) cf_destroy_texture(canvas->cf_depth_stencil);
	CF_FREE(canvas);
}

void cf_canvas_clear_depth_stencil(CF_Canvas canvas_handle, float depth, uint32_t stencil)
{
	// @TODO.
}

CF_Texture cf_canvas_get_target(CF_Canvas canvas_handle)
{
	CF_CanvasInternal* canvas = (CF_CanvasInternal*)canvas_handle.id;
	return canvas->cf_texture;
}

CF_Texture cf_canvas_get_depth_stencil_target(CF_Canvas canvas_handle)
{
	CF_CanvasInternal* canvas = (CF_CanvasInternal*)canvas_handle.id;
	return canvas->cf_depth_stencil;
}

void cf_canvas_blit(CF_Canvas src_handle, CF_V2 u0, CF_V2 v0, CF_Canvas dst_handle, CF_V2 u1, CF_V2 v1)
{
	CF_CanvasInternal* src = (CF_CanvasInternal*)src_handle.id;
	CF_CanvasInternal* dst = (CF_CanvasInternal*)dst_handle.id;
	CF_TextureInternal* src_tex = (CF_TextureInternal*)src->cf_texture.id;
	CF_TextureInternal* dst_tex = (CF_TextureInternal*)dst->cf_texture.id;
	CF_TextureInternal* src_depth_stencil = (CF_TextureInternal*)src->cf_depth_stencil.id;
	CF_TextureInternal* dst_depth_stencil = (CF_TextureInternal*)dst->cf_depth_stencil.id;

	SDL_GpuCommandBuffer* cmd = app->cmd;
	SDL_GpuTextureRegion src_tex_region = SDL_GpuTextureRegionDefaults(src_tex, u0.x, u0.y, v0.x, v0.y);
	SDL_GpuTextureRegion dst_tex_region = SDL_GpuTextureRegionDefaults(dst_tex, u0.x, u0.y, v0.x, v0.y);
	SDL_GpuBlit(cmd, &src_tex_region, &dst_tex_region, src_tex->filter, true);

	if (src_depth_stencil && dst_depth_stencil) {
		SDL_GpuTextureRegion src_depth_stencil_region = SDL_GpuTextureRegionDefaults(src_depth_stencil, u0.x, u0.y, v0.x, v0.y);
		SDL_GpuTextureRegion dst_depth_stencil_region = SDL_GpuTextureRegionDefaults(dst_depth_stencil, u0.x, u0.y, v0.x, v0.y);
		SDL_GpuBlit(cmd, &src_depth_stencil_region, &dst_depth_stencil_region, src_depth_stencil->filter, true);
	}
}

CF_Mesh cf_make_mesh(int vertex_buffer_size, int index_buffer_size)
{
	CF_MeshInternal* mesh = (CF_MeshInternal*)CF_CALLOC(sizeof(CF_MeshInternal));
	mesh->vertices.size = vertex_buffer_size;
	mesh->indices.size = index_buffer_size;
	mesh->indices.stride = sizeof(uint32_t);
	if (vertex_buffer_size) {
		mesh->vertices.buffer = SDL_GpuCreateBuffer(app->device, SDL_GPU_BUFFERUSAGE_VERTEX_BIT, vertex_buffer_size);
		mesh->vertices.transfer_buffer = SDL_GpuCreateTransferBuffer(app->device, SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, vertex_buffer_size);
	}
	if (index_buffer_size) {
		mesh->indices.buffer = SDL_GpuCreateBuffer(app->device, SDL_GPU_BUFFERUSAGE_INDEX_BIT, index_buffer_size);
		mesh->indices.transfer_buffer = SDL_GpuCreateTransferBuffer(app->device, SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, index_buffer_size);
	}
	CF_Mesh result = { (uint64_t)mesh };
	return result;
}

void cf_destroy_mesh(CF_Mesh mesh_handle)
{
	CF_MeshInternal* mesh = (CF_MeshInternal*)mesh_handle.id;
	if (mesh->vertices.buffer) {
		SDL_GpuReleaseBuffer(app->device, mesh->vertices.buffer);
		SDL_GpuReleaseTransferBuffer(app->device, mesh->vertices.transfer_buffer);
	}
	if (mesh->indices.buffer) {
		SDL_GpuReleaseBuffer(app->device, mesh->indices.buffer);
		SDL_GpuReleaseTransferBuffer(app->device, mesh->indices.transfer_buffer);
	}
	CF_FREE(mesh);
}

void cf_mesh_set_attributes(CF_Mesh mesh_handle, const CF_VertexAttribute* attributes, int attribute_count, int vertex_stride)
{
	CF_MeshInternal* mesh = (CF_MeshInternal*)mesh_handle.id;
	attribute_count = min(attribute_count, CF_MESH_MAX_VERTEX_ATTRIBUTES);
	mesh->attribute_count = attribute_count;
	mesh->vertices.stride = vertex_stride;
	for (int i = 0; i < attribute_count; ++i) {
		mesh->attributes[i] = attributes[i];
		mesh->attributes[i].name = sintern(attributes[i].name);
	}
}

void cf_mesh_update_vertex_data(CF_Mesh mesh_handle, void* data, int count)
{
	// Copy vertices over to the driver.
	CF_MeshInternal* mesh = (CF_MeshInternal*)mesh_handle.id;
	CF_ASSERT(mesh->attribute_count);
	int size = count * mesh->vertices.stride;
	CF_ASSERT(size <= mesh->vertices.size);
	void* p = NULL;
	SDL_GpuMapTransferBuffer(app->device, mesh->vertices.transfer_buffer, true, &p);
	CF_MEMCPY(p, data, size);
	SDL_GpuUnmapTransferBuffer(app->device, mesh->vertices.transfer_buffer);
	mesh->vertices.element_count = count;

	// Submit the upload command to the GPU.
	SDL_GpuCommandBuffer* cmd = app->cmd;
	SDL_GpuCopyPass *pass = SDL_GpuBeginCopyPass(cmd);
	SDL_GpuTransferBufferLocation location;
	location.offset = 0;
	location.transferBuffer = mesh->vertices.transfer_buffer;
	SDL_GpuBufferRegion region;
	region.buffer = mesh->vertices.buffer;
	region.offset = 0;
	region.size = size;
	SDL_GpuUploadToBuffer(pass, &location, &region, true);
	SDL_GpuEndCopyPass(pass);
}

void cf_mesh_update_index_data(CF_Mesh mesh_handle, uint32_t* indices, int count)
{
	//CF_MeshInternal* mesh = (CF_MeshInternal*)mesh_handle.id;
	//CF_ASSERT(mesh->attribute_count);
	//int size = count * mesh->indices.stride;
	//void* p = NULL;
	//SDL_GpuMapTransferBuffer(app->device, mesh->indices.transfer_buffer, true, &p);
	//CF_MEMCPY(p, indices, size);
	//SDL_GpuUnmapTransferBuffer(app->device, mesh->indices.transfer_buffer);
	//mesh->indices.element_count = count;
}

CF_RenderState cf_render_state_defaults()
{
	CF_RenderState state;
	state.blend.enabled = false;
	state.cull_mode = CF_CULL_MODE_NONE;
	state.blend.pixel_format = CF_PIXEL_FORMAT_R8G8B8A8;
	state.blend.write_R_enabled = true;
	state.blend.write_G_enabled = true;
	state.blend.write_B_enabled = true;
	state.blend.write_A_enabled = true;
	state.blend.rgb_op = CF_BLEND_OP_ADD;
	state.blend.rgb_src_blend_factor = CF_BLENDFACTOR_ONE;
	state.blend.rgb_dst_blend_factor = CF_BLENDFACTOR_ZERO;
	state.blend.alpha_op = CF_BLEND_OP_ADD;
	state.blend.alpha_src_blend_factor = CF_BLENDFACTOR_ONE;
	state.blend.alpha_dst_blend_factor = CF_BLENDFACTOR_ZERO;
	state.depth_compare = CF_COMPARE_FUNCTION_ALWAYS;
	state.depth_write_enabled = false;
	state.stencil.enabled = false;
	state.stencil.read_mask = 0;
	state.stencil.write_mask = 0;
	state.stencil.reference = 0;
	state.stencil.front.compare = CF_COMPARE_FUNCTION_ALWAYS;
	state.stencil.front.fail_op = CF_STENCIL_OP_KEEP;
	state.stencil.front.depth_fail_op = CF_STENCIL_OP_KEEP;
	state.stencil.front.pass_op = CF_STENCIL_OP_KEEP;
	state.stencil.back.compare = CF_COMPARE_FUNCTION_ALWAYS;
	state.stencil.back.fail_op = CF_STENCIL_OP_KEEP;
	state.stencil.back.depth_fail_op = CF_STENCIL_OP_KEEP;
	state.stencil.back.pass_op = CF_STENCIL_OP_KEEP;
	return state;
}

CF_Material cf_make_material()
{
	CF_MaterialInternal* material = CF_NEW(CF_MaterialInternal);
	cf_arena_init(&material->uniform_arena, 4, 1024);
	cf_arena_init(&material->block_arena, 4, 1024);
	material->state = cf_render_state_defaults();
	CF_Material result = { (uint64_t)material };
	return result;
}

void cf_destroy_material(CF_Material material_handle)
{
	CF_MaterialInternal* material = (CF_MaterialInternal*)material_handle.id;
	cf_arena_reset(&material->uniform_arena);
	cf_arena_reset(&material->block_arena);
	material->~CF_MaterialInternal();
	CF_FREE(material);
}

void cf_material_set_render_state(CF_Material material_handle, CF_RenderState render_state)
{
	CF_MaterialInternal* material = (CF_MaterialInternal*)material_handle.id;
	if (CF_MEMCMP(&material->state, &render_state, sizeof(material->state))) {
		material->state = render_state;
		material->dirty = true;
	}
}

static void s_material_set_texture(CF_MaterialInternal* material, CF_MaterialState* state, const char* name, CF_Texture texture)
{
	bool found = false;
	for (int i = 0; i < state->textures.count(); ++i) {
		if (state->textures[i].name == name) {
			state->textures[i].handle = texture;
			found = true;
			break;
		}
	}
	if (!found) {
		CF_MaterialTex tex;
		tex.name = name;
		tex.handle = texture;
		state->textures.add(tex);
		material->dirty = true;
	}
}

void cf_material_set_texture_vs(CF_Material material_handle, const char* name, CF_Texture texture)
{
	CF_MaterialInternal* material = (CF_MaterialInternal*)material_handle.id;
	name = sintern(name);
	s_material_set_texture(material, &material->vs, name, texture);
}

void cf_material_set_texture_fs(CF_Material material_handle, const char* name, CF_Texture texture)
{
	CF_MaterialInternal* material = (CF_MaterialInternal*)material_handle.id;
	name = sintern(name);
	s_material_set_texture(material, &material->fs, name, texture);
}

void cf_material_clear_textures(CF_Material material_handle)
{
	CF_MaterialInternal* material = (CF_MaterialInternal*)material_handle.id;
	material->vs.textures.clear();
	material->fs.textures.clear();
	material->dirty = true;
}

static void s_material_set_uniform(CF_Arena* arena, CF_MaterialState* state, const char* name, void* data, CF_UniformType type, int array_length)
{
	if (array_length <= 0) array_length = 1;
	CF_Uniform* uniform = NULL;
	for (int i = 0; i < state->uniforms.count(); ++i) {
		CF_Uniform* u = state->uniforms + i;
		if (u->name == name) {
			uniform = u;
			break;
		}
	}
	int size = s_uniform_size(type) * array_length;
	if (!uniform) {
		uniform = &state->uniforms.add();
		uniform->name = name;
		uniform->data = cf_arena_alloc(arena, size);
		uniform->size = size;
		uniform->type = type;
		uniform->array_length = array_length;
	}
	CF_ASSERT(uniform->type == type);
	CF_ASSERT(uniform->array_length == array_length);
	CF_MEMCPY(uniform->data, data, size);
}

void cf_material_set_uniform_vs(CF_Material material_handle, const char* name, void* data, CF_UniformType type, int array_length)
{
	CF_MaterialInternal* material = (CF_MaterialInternal*)material_handle.id;
	name = sintern(name);
	s_material_set_uniform(&material->uniform_arena, &material->vs, name, data, type, array_length);
}

void cf_material_set_uniform_fs(CF_Material material_handle, const char* name, void* data, CF_UniformType type, int array_length)
{
	CF_MaterialInternal* material = (CF_MaterialInternal*)material_handle.id;
	name = sintern(name);
	s_material_set_uniform(&material->uniform_arena, &material->fs, name, data, type, array_length);
}

void cf_material_clear_uniforms(CF_Material material_handle)
{
	CF_MaterialInternal* material = (CF_MaterialInternal*)material_handle.id;
	arena_reset(&material->uniform_arena);
	material->vs.uniforms.clear();
	material->fs.uniforms.clear();
}

void cf_clear_color(float red, float green, float blue, float alpha)
{
	app->clear_color = make_color(red, green, blue, alpha);
}

void cf_apply_canvas(CF_Canvas canvas_handle, bool clear)
{
	CF_CanvasInternal* canvas = (CF_CanvasInternal*)canvas_handle.id;
	CF_ASSERT(canvas);
	s_canvas = canvas;
	s_canvas->clear = clear;
}

void cf_apply_viewport(int x, int y, int w, int h)
{
	CF_ASSERT(s_canvas);
	CF_ASSERT(s_canvas->pass);
	SDL_GpuViewport viewport;
	viewport.x = (float)x;
	viewport.y = (float)y;
	viewport.w = (float)w;
	viewport.h = (float)h;
	viewport.minDepth = 0;
	viewport.maxDepth = 1;
	SDL_GpuSetViewport(s_canvas->pass, &viewport);
}

void cf_apply_scissor(int x, int y, int w, int h)
{
	CF_ASSERT(s_canvas);
	CF_ASSERT(s_canvas->pass);
	SDL_Rect scissor;
	scissor.x = x;
	scissor.y = y;
	scissor.w = w;
	scissor.y = h;
	SDL_GpuSetScissor(s_canvas->pass, &scissor);
}

void cf_apply_mesh(CF_Mesh mesh_handle)
{
	CF_ASSERT(s_canvas);
	CF_MeshInternal* mesh = (CF_MeshInternal*)mesh_handle.id;
	s_canvas->mesh = mesh;
}

static void s_copy_uniforms(SDL_GpuCommandBuffer* cmd, CF_Arena* arena, CF_ShaderInternal* shd, CF_MaterialState* mstate, bool vs)
{
	// Create any required uniform blocks for all uniforms matching between which uniforms
	// the material has and the shader needs.
	int block_size = vs ? shd->vs_block_size : shd->fs_block_size;
	if (!block_size) return;
	void* block = cf_arena_alloc(arena, block_size);
	for (int i = 0; i < mstate->uniforms.count(); ++i) {
		CF_Uniform uniform = mstate->uniforms[i];
		int idx = vs ? shd->vs_index(uniform.name) : shd->fs_index(uniform.name);
		if (idx >= 0) {
			int offset = vs ? shd->vs_uniform_block_members[idx].offset : shd->fs_uniform_block_members[idx].offset;
			void* dst = (void*)(((uintptr_t)block) + offset);
			CF_MEMCPY(dst, uniform.data, uniform.size);
		}
	}

	// Send uniform data to the GPU.
	if (vs) {
		SDL_GpuPushVertexUniformData(cmd, 0, block, (uint32_t)block_size);
	} else {
		SDL_GpuPushFragmentUniformData(cmd, 0, block, (uint32_t)block_size);
	}

	// @TODO Use a different allocation scheme that caches better.
	cf_arena_reset(arena);
}

static SDL_GpuCompareOp s_wrap(CF_CompareFunction compare_function)
{
	switch (compare_function)
	{
	case CF_COMPARE_FUNCTION_ALWAYS:                return SDL_GPU_COMPAREOP_ALWAYS;
	case CF_COMPARE_FUNCTION_NEVER:                 return SDL_GPU_COMPAREOP_NEVER;
	case CF_COMPARE_FUNCTION_LESS_THAN:             return SDL_GPU_COMPAREOP_LESS;
	case CF_COMPARE_FUNCTION_EQUAL:                 return SDL_GPU_COMPAREOP_EQUAL;
	case CF_COMPARE_FUNCTION_NOT_EQUAL:             return SDL_GPU_COMPAREOP_NOT_EQUAL;
	case CF_COMPARE_FUNCTION_LESS_THAN_OR_EQUAL:    return SDL_GPU_COMPAREOP_LESS_OR_EQUAL;
	case CF_COMPARE_FUNCTION_GREATER_THAN:          return SDL_GPU_COMPAREOP_GREATER;
	case CF_COMPARE_FUNCTION_GREATER_THAN_OR_EQUAL: return SDL_GPU_COMPAREOP_GREATER_OR_EQUAL;
	default:                                        return SDL_GPU_COMPAREOP_ALWAYS;
	}
}

static SDL_GpuStencilOp s_wrap(CF_StencilOp stencil_op)
{
	switch (stencil_op)
	{
	case CF_STENCIL_OP_KEEP:            return SDL_GPU_STENCILOP_KEEP;
	case CF_STENCIL_OP_ZERO:            return SDL_GPU_STENCILOP_ZERO;
	case CF_STENCIL_OP_REPLACE:         return SDL_GPU_STENCILOP_REPLACE;
	case CF_STENCIL_OP_INCREMENT_CLAMP: return SDL_GPU_STENCILOP_INCREMENT_AND_CLAMP;
	case CF_STENCIL_OP_DECREMENT_CLAMP: return SDL_GPU_STENCILOP_DECREMENT_AND_CLAMP;
	case CF_STENCIL_OP_INVERT:          return SDL_GPU_STENCILOP_INVERT;
	case CF_STENCIL_OP_INCREMENT_WRAP:  return SDL_GPU_STENCILOP_INCREMENT_AND_WRAP;
	case CF_STENCIL_OP_DECREMENT_WRAP:  return SDL_GPU_STENCILOP_DECREMENT_AND_WRAP;
	default:                            return SDL_GPU_STENCILOP_KEEP;
	}
}

static SDL_GpuBlendOp s_wrap(CF_BlendOp blend_op)
{
	switch (blend_op)
	{
	case CF_BLEND_OP_ADD:              return SDL_GPU_BLENDOP_ADD;
	case CF_BLEND_OP_SUBTRACT:         return SDL_GPU_BLENDOP_SUBTRACT;
	case CF_BLEND_OP_REVERSE_SUBTRACT: return SDL_GPU_BLENDOP_REVERSE_SUBTRACT;
	case CF_BLEND_OP_MIN:              return SDL_GPU_BLENDOP_MIN;
	case CF_BLEND_OP_MAX:              return SDL_GPU_BLENDOP_MAX;
	default:                           return SDL_GPU_BLENDOP_ADD;
	}
}

SDL_GpuBlendFactor s_wrap(CF_BlendFactor factor)
{
	switch (factor) {
	case CF_BLENDFACTOR_ZERO:                    return SDL_GPU_BLENDFACTOR_ZERO;
	case CF_BLENDFACTOR_ONE:                     return SDL_GPU_BLENDFACTOR_ONE;
	case CF_BLENDFACTOR_SRC_COLOR:               return SDL_GPU_BLENDFACTOR_SRC_COLOR;
	case CF_BLENDFACTOR_ONE_MINUS_SRC_COLOR:     return SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_COLOR;
	case CF_BLENDFACTOR_DST_COLOR:               return SDL_GPU_BLENDFACTOR_DST_COLOR;
	case CF_BLENDFACTOR_ONE_MINUS_DST_COLOR:     return SDL_GPU_BLENDFACTOR_ONE_MINUS_DST_COLOR;
	case CF_BLENDFACTOR_SRC_ALPHA:               return SDL_GPU_BLENDFACTOR_SRC_ALPHA;
	case CF_BLENDFACTOR_ONE_MINUS_SRC_ALPHA:     return SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
	case CF_BLENDFACTOR_DST_ALPHA:               return SDL_GPU_BLENDFACTOR_DST_ALPHA;
	case CF_BLENDFACTOR_ONE_MINUS_DST_ALPHA:     return SDL_GPU_BLENDFACTOR_ONE_MINUS_DST_ALPHA;
	case CF_BLENDFACTOR_CONSTANT_COLOR:          return SDL_GPU_BLENDFACTOR_CONSTANT_COLOR;
	case CF_BLENDFACTOR_ONE_MINUS_CONSTANT_COLOR:return SDL_GPU_BLENDFACTOR_ONE_MINUS_CONSTANT_COLOR;
	case CF_BLENDFACTOR_SRC_ALPHA_SATURATE:      return SDL_GPU_BLENDFACTOR_SRC_ALPHA_SATURATE;
	default:                                     return SDL_GPU_BLENDFACTOR_ZERO;
	}
}

static SDL_GpuGraphicsPipeline* s_build_pipeline(CF_ShaderInternal* shader, CF_RenderState* state, CF_MeshInternal* mesh)
{
	SDL_GpuColorAttachmentDescription color_info;
	CF_MEMSET(&color_info, 0, sizeof(color_info));
	CF_ASSERT(s_canvas->texture);
	color_info.format = ((CF_TextureInternal*)s_canvas->cf_texture.id)->format;
	color_info.blendState.blendEnable = state->blend.enabled;
	color_info.blendState.alphaBlendOp = s_wrap(state->blend.alpha_op);
	color_info.blendState.colorBlendOp = s_wrap(state->blend.rgb_op);
	color_info.blendState.srcColorBlendFactor = s_wrap(state->blend.rgb_src_blend_factor);
	color_info.blendState.srcAlphaBlendFactor = s_wrap(state->blend.alpha_src_blend_factor);
	color_info.blendState.dstColorBlendFactor = s_wrap(state->blend.rgb_dst_blend_factor);
	color_info.blendState.dstAlphaBlendFactor = s_wrap(state->blend.alpha_dst_blend_factor);
	int mask_r = (int)state->blend.write_R_enabled << 0;
	int mask_g = (int)state->blend.write_G_enabled << 1;
	int mask_b = (int)state->blend.write_B_enabled << 2;
	int mask_a = (int)state->blend.write_A_enabled << 3;
	color_info.blendState.colorWriteMask = (uint32_t)(mask_r | mask_g | mask_b | mask_a);

	SDL_GpuGraphicsPipelineCreateInfo pip_info;
	CF_MEMSET(&pip_info, 0, sizeof(pip_info));
	pip_info.attachmentInfo.colorAttachmentCount = 1;
	pip_info.attachmentInfo.colorAttachmentDescriptions = &color_info;
	pip_info.vertexShader = shader->vs;
	pip_info.fragmentShader = shader->fs;
	pip_info.attachmentInfo.hasDepthStencilAttachment = state->depth_write_enabled;
	if (s_canvas->cf_depth_stencil.id) {
		pip_info.attachmentInfo.depthStencilFormat = ((CF_TextureInternal*)s_canvas->cf_depth_stencil.id)->format;
	}

	// Make sure the mesh vertex format is fully compatible with the vertex shader inputs.
	SDL_GpuVertexAttribute* attributes = SDL_stack_alloc(SDL_GpuVertexAttribute, mesh->attribute_count);
	int attribute_count = 0;
	for (int i = 0; i < mesh->attribute_count; ++i) {
		SDL_GpuVertexAttribute* attr = attributes + attribute_count;
		int idx = shader->get_input_index(mesh->attributes[i].name);
		if (idx >= 0) {
			CF_ShaderInputFormat input_fmt = shader->input_formats[idx];
			CF_VertexFormat mesh_fmt = mesh->attributes[i].format;
			CF_ASSERT(s_is_compatible(input_fmt, mesh_fmt));
			attr->binding = 0;
			attr->location = shader->input_locations[idx];
			attr->format = s_wrap(mesh->attributes[i].format);
			attr->offset = mesh->attributes[i].offset;
			++attribute_count;
		}
	}
	CF_ASSERT(attribute_count == shader->input_count);
	pip_info.vertexInputState.vertexAttributeCount = attribute_count;
	pip_info.vertexInputState.vertexAttributes = attributes;
	SDL_GpuVertexBinding vertex_bindings[2];
	vertex_bindings[0].binding = 0;
	vertex_bindings[0].stride = mesh->vertices.stride;
	vertex_bindings[0].inputRate = SDL_GPU_VERTEXINPUTRATE_VERTEX;
	vertex_bindings[0].stepRate = 0;
	pip_info.vertexInputState.vertexBindings = vertex_bindings;
	//if (has_instance_data) {
	//	vertex_bindings[1].binding = 1;
	//	vertex_bindings[1].stride = mesh->instances.stride;
	//	vertex_bindings[1].inputRate = SDL_GPU_VERTEXINPUTRATE_INSTANCE;
	//	vertex_bindings[1].stepRate = 0;
	//	pip_info.vertexInputState.vertexBindingCount = 2;
	//} else {
		pip_info.vertexInputState.vertexBindingCount = 1;
	//}

	pip_info.primitiveType = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
	pip_info.rasterizerState.fillMode = SDL_GPU_FILLMODE_FILL;
	pip_info.rasterizerState.cullMode = SDL_GPU_CULLMODE_NONE;
	pip_info.rasterizerState.frontFace = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE;
	pip_info.rasterizerState.depthBiasEnable = false;
	pip_info.rasterizerState.depthBiasConstantFactor = 0;
	pip_info.rasterizerState.depthBiasClamp = 0;
	pip_info.rasterizerState.depthBiasSlopeFactor = 0;
	pip_info.multisampleState.sampleCount = SDL_GPU_SAMPLECOUNT_1;
	pip_info.multisampleState.sampleMask = 0xFFFF;

	pip_info.depthStencilState.depthTestEnable = state->depth_write_enabled;
	pip_info.depthStencilState.depthWriteEnable = state->depth_write_enabled;
	pip_info.depthStencilState.compareOp = s_wrap(state->depth_compare);
	pip_info.depthStencilState.stencilTestEnable = state->stencil.enabled;
	pip_info.depthStencilState.backStencilState.failOp = s_wrap(state->stencil.back.fail_op);
	pip_info.depthStencilState.backStencilState.passOp = s_wrap(state->stencil.back.pass_op);
	pip_info.depthStencilState.backStencilState.depthFailOp = s_wrap(state->stencil.back.depth_fail_op);
	pip_info.depthStencilState.backStencilState.compareOp = s_wrap(state->stencil.back.compare);
	pip_info.depthStencilState.frontStencilState.failOp = s_wrap(state->stencil.front.fail_op);
	pip_info.depthStencilState.frontStencilState.passOp = s_wrap(state->stencil.front.pass_op);
	pip_info.depthStencilState.frontStencilState.depthFailOp = s_wrap(state->stencil.front.depth_fail_op);
	pip_info.depthStencilState.frontStencilState.compareOp = s_wrap(state->stencil.front.compare);
	pip_info.depthStencilState.compareMask = state->stencil.read_mask;
	pip_info.depthStencilState.writeMask = state->stencil.write_mask;
	pip_info.depthStencilState.reference = state->stencil.reference;

	SDL_GpuGraphicsPipeline* pip = SDL_GpuCreateGraphicsPipeline(app->device, &pip_info);
	CF_ASSERT(pip);
	return pip;
}

void cf_apply_shader(CF_Shader shader_handle, CF_Material material_handle)
{
	CF_ASSERT(s_canvas);
	CF_ASSERT(s_canvas->mesh);
	CF_MeshInternal* mesh = s_canvas->mesh;
	CF_MaterialInternal* material = (CF_MaterialInternal*)material_handle.id;
	CF_ShaderInternal* shader = (CF_ShaderInternal*)shader_handle.id;
	CF_RenderState* state = &material->state;

	// Cache the pipeline to avoid create/release each frame.
	// ...Build a new one if the material marks itself as dirty.
	SDL_GpuGraphicsPipeline* pip = NULL;
	bool found = false;
	for (int i = 0; i < shader->pip_cache.count(); ++i) {
		CF_Pipeline pip_cache = shader->pip_cache[i];
		if (pip_cache.material == material && pip_cache.mesh == mesh) {
			found = true;
			if (material->dirty) {
				material->dirty = false;
				pip = s_build_pipeline(shader, state, mesh);
				if (pip_cache.pip) {
					SDL_GpuReleaseGraphicsPipeline(app->device, pip_cache.pip);
				}
				shader->pip_cache[i].pip = pip;
			} else {
				pip = pip_cache.pip;
			}
		}
	}
	if (!found) {
		pip = s_build_pipeline(shader, state, mesh);
		shader->pip_cache.add({ material, pip, mesh });
	}
	CF_ASSERT(pip);

	SDL_GpuCommandBuffer* cmd = app->cmd;
	s_canvas->pip = pip;

	SDL_GpuColorAttachmentInfo pass_color_info = { 0 };
	pass_color_info.textureSlice.texture = s_canvas->texture;
	pass_color_info.clearColor = { app->clear_color.r, app->clear_color.g, app->clear_color.b, app->clear_color.a };
	pass_color_info.loadOp = s_canvas->clear ? SDL_GPU_LOADOP_CLEAR : SDL_GPU_LOADOP_LOAD;
	pass_color_info.storeOp = SDL_GPU_STOREOP_STORE;
	s_canvas->clear = false;
	pass_color_info.cycle = s_canvas->clear ? true : false;
	SDL_GpuDepthStencilAttachmentInfo pass_depth_stencil_info = { 0 };
	pass_depth_stencil_info.textureSlice.texture = s_canvas->depth_stencil;
	if (s_canvas->depth_stencil) {
		pass_depth_stencil_info.loadOp = SDL_GPU_LOADOP_LOAD;
		pass_depth_stencil_info.storeOp = SDL_GPU_STOREOP_STORE;
		pass_depth_stencil_info.stencilLoadOp = SDL_GPU_LOADOP_LOAD;
		pass_depth_stencil_info.stencilStoreOp = SDL_GPU_STOREOP_DONT_CARE;
		pass_depth_stencil_info.cycle = s_canvas->clear ? true : false;
	}
	SDL_GpuRenderPass* pass = SDL_GpuBeginRenderPass(cmd, &pass_color_info, 1, NULL);
	CF_ASSERT(pass);
	s_canvas->pass = pass;
	SDL_GpuBindGraphicsPipeline(pass, pip);
	SDL_GpuBufferBinding bind;
	bind.buffer = mesh->vertices.buffer;
	bind.offset = 0;
	SDL_GpuBindVertexBuffers(pass, 0, &bind, 1);
	// @TODO SDL_GpuBindIndexBuffer
	// @TODO Storage/compute.

	// Bind images to all their respective slots.
	int sampler_count = shader->image_names.count();
	SDL_GpuTextureSamplerBinding* sampler_bindings = SDL_stack_alloc(SDL_GpuTextureSamplerBinding, sampler_count);
	int found_image_count = 0;
	for (int i = 0; found_image_count < sampler_count && i < material->fs.textures.count(); ++i) {
		const char* image_name = material->fs.textures[i].name;
		for (int i = 0; i < shader->image_names.size(); ++i) {
			if (shader->image_names[i] == image_name) {
				sampler_bindings[found_image_count].sampler = ((CF_TextureInternal*)material->fs.textures[i].handle.id)->sampler;
				sampler_bindings[found_image_count].texture = ((CF_TextureInternal*)material->fs.textures[i].handle.id)->tex;
				found_image_count++;
			}
		}
	}
	CF_ASSERT(found_image_count == sampler_count);
	SDL_GpuBindFragmentSamplers(pass, 0, sampler_bindings, (Uint32)found_image_count);

	// Copy over uniform data.
	s_copy_uniforms(cmd, &material->block_arena, shader, &material->vs, true);
	s_copy_uniforms(cmd, &material->block_arena, shader, &material->fs, false);
}

void cf_draw_elements()
{
	CF_MeshInternal* mesh = s_canvas->mesh;
	if (mesh->indices.buffer) {
		// @TODO
		//SDL_GpuDrawIndexedPrimitives(s_canvas->pass, 0, mesh->vertices.element_count);
		CF_ASSERT(false);
	} else {
		SDL_GpuDrawPrimitives(s_canvas->pass, 0, mesh->vertices.element_count);
	}
	app->draw_call_count++;
}

void cf_commit()
{
	SDL_GpuEndRenderPass(s_canvas->pass);
	CF_MeshInternal* mesh = s_canvas->mesh;
	if (mesh) {
		mesh->vertices.element_count = 0;
		mesh->indices.element_count = 0;
	}
}

#include <SPIRV-Reflect/spirv_reflect.c>
