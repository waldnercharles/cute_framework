// Holographic Radiance Cascades
//
// Demonstrates HRC 2D global illumination using the production pipeline:
//   trace -> extend -> merge -> sum_quadrants -> blur -> composite
//
// Controls:
//   D       Cycle debug modes (0=normal, 1-4=quadrant, 5=no blur, 6=emission, 7=transmittance)
//
// Reference: Freeman, Sannikov, Margel (2025) "Holographic Radiance Cascades"
// https://arxiv.org/pdf/2505.02041
// https://github.com/entropylost/amitabha

#include <cute.h>

//--------------------------------------------------------------------------------------------------
// Configuration.

#define HRC_DIM           512
#define HRC_N             9     // log2(HRC_DIM)
#define HRC_WG            16
#define HRC_BLUR_WG       8
#define HRC_TRACE_CUTOFF  3     // trace levels 0,1,2 via DDA; extend levels 3..N
#define HRC_SCALE         0.5f  // solver scale: 1.0 = full res, 0.5 = half res
#define HRC_NUM_DEBUG     8     // debug modes 0..7

//--------------------------------------------------------------------------------------------------
// Shader loading.

CF_ComputeShader load_compute_shader(const char* path)
{
	char* src = cf_fs_read_entire_file_to_memory_and_nul_terminate(path, NULL);
	CF_ComputeShader cs = cf_make_compute_shader_from_source(src);
	cf_free(src);
	return cs;
}

//--------------------------------------------------------------------------------------------------
// HRC state.

typedef struct Hrc
{
	CF_Canvas emission;
	CF_Canvas transmittance;
	CF_StorageBuffer t_rad[HRC_N + 1];
	CF_StorageBuffer t_trn[HRC_N + 1];
	CF_StorageBuffer r_rad[2];
	CF_StorageBuffer r_zero;
	CF_StorageBuffer frustum[4];
	CF_StorageBuffer merge_weights[HRC_N];
	CF_Texture radiance_preblur;
	CF_Texture radiance;
	CF_Canvas fluence;
	CF_Material mat_trace;
	CF_Material mat_extend;
	CF_Material mat_merge;
	CF_Material mat_sum_quadrants;
	CF_Material mat_blur;
	CF_Material mat_composite;
	CF_ComputeShader cs_trace;
	CF_ComputeShader cs_extend;
	CF_ComputeShader cs_merge;
	CF_ComputeShader cs_sum_quadrants;
	CF_ComputeShader cs_blur;
	CF_ComputeShader cs_composite;
	int t_w[HRC_N + 1];
	int debug_mode;
	int work_dim;
	int work_N;
	float mip_level;
} Hrc;

Hrc hrc;

//--------------------------------------------------------------------------------------------------
// Helpers.

CF_StorageBuffer hrc_make_buf(int w, int h)
{
	CF_StorageBufferParams p = cf_storage_buffer_defaults(w * h * 8);
	p.compute_readable = true;
	p.compute_writable = true;
	return cf_make_storage_buffer(p);
}

CF_Canvas hrc_make_canvas(int w, int h, CF_PixelFormat fmt, bool mipmaps)
{
	CF_CanvasParams p = cf_canvas_defaults(w, h);
	p.target.pixel_format = fmt;
	p.target.filter = CF_FILTER_LINEAR;
	p.target.usage = CF_TEXTURE_USAGE_SAMPLER_BIT | CF_TEXTURE_USAGE_COLOR_TARGET_BIT | CF_TEXTURE_USAGE_COMPUTE_STORAGE_READ_BIT | CF_TEXTURE_USAGE_COMPUTE_STORAGE_WRITE_BIT;
	p.target.wrap_u = CF_WRAP_MODE_CLAMP_TO_EDGE;
	p.target.wrap_v = CF_WRAP_MODE_CLAMP_TO_EDGE;
	if (mipmaps) {
		p.target.allocate_mipmaps = true;
		p.target.mip_filter = CF_MIP_FILTER_LINEAR;
	}
	return cf_make_canvas(p);
}

CF_Texture hrc_make_compute_texture(int w, int h)
{
	CF_TextureParams p = cf_texture_defaults(w, h);
	p.pixel_format = CF_PIXEL_FORMAT_R16G16B16A16_FLOAT;
	p.usage = CF_TEXTURE_USAGE_SAMPLER_BIT | CF_TEXTURE_USAGE_COMPUTE_STORAGE_READ_BIT | CF_TEXTURE_USAGE_COMPUTE_STORAGE_WRITE_BIT;
	p.filter = CF_FILTER_LINEAR;
	p.wrap_u = CF_WRAP_MODE_CLAMP_TO_EDGE;
	p.wrap_v = CF_WRAP_MODE_CLAMP_TO_EDGE;
	return cf_make_texture(p);
}

int hrc_div_ceil(int a, int b)
{
	return (a + b - 1) / b;
}

int hrc_probe_count(int world_w, int level)
{
	int step = 1 << level;
	return (world_w + step - 1) / step;
}

int hrc_t_width(int world_w, int level)
{
	return hrc_probe_count(world_w, level) * ((1 << level) + 1);
}

int hrc_r_width(int world_w, int level)
{
	return hrc_probe_count(world_w, level) * (1 << level);
}

float hrc_angular_span(int n, float i)
{
	float pow2n = (float)(1 << n);
	float left = atan2f(2.0f * (i - 0.5f) - pow2n, pow2n);
	float right = atan2f(2.0f * (i + 0.5f) - pow2n, pow2n);
	return right - left;
}

//--------------------------------------------------------------------------------------------------
// Init / shutdown.

void hrc_init()
{
	CF_MEMSET(&hrc, 0, sizeof(hrc));
	int dim = HRC_DIM;
	int work = (int)((float)dim * HRC_SCALE);
	if (work < 16) work = 16;
	hrc.work_dim = work;
	hrc.work_N = (int)log2f((float)work);
	if (hrc.work_N > HRC_N) hrc.work_N = HRC_N;
	hrc.mip_level = log2f((float)dim / (float)work);

	// Precompute T cascade buffer widths (solver resolution).
	for (int i = 0; i <= hrc.work_N; i++) {
		hrc.t_w[i] = hrc_t_width(work, i);
	}

	// Scene input canvases (full resolution, mipmapped for solver downsampling).
	hrc.emission = hrc_make_canvas(dim, dim, CF_PIXEL_FORMAT_R16G16B16A16_FLOAT, true);
	hrc.transmittance = hrc_make_canvas(dim, dim, CF_PIXEL_FORMAT_R16G16B16A16_FLOAT, true);

	// Per-cascade T SSBOs (solver resolution, uvec2 per texel = 8 bytes, f16-packed).
	for (int i = 0; i <= hrc.work_N; i++) {
		hrc.t_rad[i] = hrc_make_buf(hrc.t_w[i], work);
		hrc.t_trn[i] = hrc_make_buf(hrc.t_w[i], work);
	}

	// R ping-pong SSBOs + zero buffer for R_N = 0.
	for (int i = 0; i < 2; i++)
		hrc.r_rad[i] = hrc_make_buf(work, work);
	hrc.r_zero = hrc_make_buf(work, work);
	{
		int sz = work * work * 8;
		void* zeros = cf_calloc(sz, 1);
		cf_update_storage_buffer(hrc.r_zero, zeros, sz);
		cf_free(zeros);
	}

	// Per-frustum output SSBOs (4 rotations, solver resolution).
	for (int i = 0; i < 4; i++)
		hrc.frustum[i] = hrc_make_buf(work, work);

	// Precompute merge angular weights per cascade level.
	for (int level = 0; level < hrc.work_N; level++) {
		int directions = 1 << level;
		hrc.merge_weights[level] = hrc_make_buf(directions, 1);

		int float_count = directions * 2;
		float* weights = (float*)cf_calloc(float_count * (int)sizeof(float), 1);
		for (int i = 0; i < directions; i++) {
			float j_plus = 2.0f * (float)i + 1.0f;
			float j_minus = 2.0f * (float)i;
			weights[i * 2 + 0] = hrc_angular_span(level + 1, j_plus + 0.5f);
			weights[i * 2 + 1] = hrc_angular_span(level + 1, j_minus + 0.5f);
		}
		cf_update_storage_buffer(hrc.merge_weights[level], weights, float_count * (int)sizeof(float));
		cf_free(weights);
	}

	// Compute textures for post-processing (solver resolution).
	hrc.radiance_preblur = hrc_make_compute_texture(work, work);
	hrc.radiance = hrc_make_compute_texture(work, work);

	// Final output canvas (solver resolution, upscaled on display).
	hrc.fluence = hrc_make_canvas(work, work, CF_PIXEL_FORMAT_R8G8B8A8_UNORM, false);

	// Materials.
	hrc.mat_trace = cf_make_material();
	hrc.mat_extend = cf_make_material();
	hrc.mat_merge = cf_make_material();
	hrc.mat_sum_quadrants = cf_make_material();
	hrc.mat_blur = cf_make_material();
	hrc.mat_composite = cf_make_material();

	// Compute shaders.
	hrc.cs_trace = load_compute_shader("/hrc_data/hrc_trace.c_shd");
	hrc.cs_extend = load_compute_shader("/hrc_data/hrc_extend.c_shd");
	hrc.cs_merge = load_compute_shader("/hrc_data/hrc_merge.c_shd");
	hrc.cs_sum_quadrants = load_compute_shader("/hrc_data/hrc_sum_quadrants.c_shd");
	hrc.cs_blur = load_compute_shader("/hrc_data/hrc_blur.c_shd");
	hrc.cs_composite = load_compute_shader("/hrc_data/hrc_composite.c_shd");
}

void hrc_shutdown()
{
	cf_destroy_canvas(hrc.emission);
	cf_destroy_canvas(hrc.transmittance);
	for (int i = 0; i <= hrc.work_N; i++) {
		cf_destroy_storage_buffer(hrc.t_rad[i]);
		cf_destroy_storage_buffer(hrc.t_trn[i]);
	}
	for (int i = 0; i < 2; i++)
		cf_destroy_storage_buffer(hrc.r_rad[i]);
	cf_destroy_storage_buffer(hrc.r_zero);
	for (int i = 0; i < 4; i++)
		cf_destroy_storage_buffer(hrc.frustum[i]);
	for (int i = 0; i < hrc.work_N; i++)
		cf_destroy_storage_buffer(hrc.merge_weights[i]);
	cf_destroy_texture(hrc.radiance_preblur);
	cf_destroy_texture(hrc.radiance);
	cf_destroy_canvas(hrc.fluence);
	cf_destroy_material(hrc.mat_trace);
	cf_destroy_material(hrc.mat_extend);
	cf_destroy_material(hrc.mat_merge);
	cf_destroy_material(hrc.mat_sum_quadrants);
	cf_destroy_material(hrc.mat_blur);
	cf_destroy_material(hrc.mat_composite);
	cf_destroy_compute_shader(hrc.cs_trace);
	cf_destroy_compute_shader(hrc.cs_extend);
	cf_destroy_compute_shader(hrc.cs_merge);
	cf_destroy_compute_shader(hrc.cs_sum_quadrants);
	cf_destroy_compute_shader(hrc.cs_blur);
	cf_destroy_compute_shader(hrc.cs_composite);
}

//--------------------------------------------------------------------------------------------------
// Cascade compute pipeline.

void hrc_compute()
{
	CF_Texture emiss_tex = cf_canvas_get_target(hrc.emission);
	CF_Texture trans_tex = cf_canvas_get_target(hrc.transmittance);
	CF_Texture fluence_tex = cf_canvas_get_target(hrc.fluence);
	int dim = hrc.work_dim;

	for (int j = 0; j < 4; j++) {
		// Trace T_0..cutoff-1 via DDA.
		int trace_max = HRC_TRACE_CUTOFF < hrc.work_N ? HRC_TRACE_CUTOFF : hrc.work_N;
		for (int level = 0; level < trace_max; level++) {
			int rot_w = dim; // square world, so rot_w == rot_h == dim for all rotations
			int rot_h = dim;

			cf_material_set_texture_cs(hrc.mat_trace, "u_emission", emiss_tex);
			cf_material_set_texture_cs(hrc.mat_trace, "u_transmittance", trans_tex);

			int params[6] = { level, j, rot_w, rot_h, dim, dim };
			cf_material_set_uniform_cs(hrc.mat_trace, "u_cascade", params + 0, CF_UNIFORM_TYPE_INT, 1);
			cf_material_set_uniform_cs(hrc.mat_trace, "u_rotate", params + 1, CF_UNIFORM_TYPE_INT, 1);
			cf_material_set_uniform_cs(hrc.mat_trace, "u_world_w", params + 2, CF_UNIFORM_TYPE_INT, 1);
			cf_material_set_uniform_cs(hrc.mat_trace, "u_world_h", params + 3, CF_UNIFORM_TYPE_INT, 1);
			cf_material_set_uniform_cs(hrc.mat_trace, "u_work_w", params + 4, CF_UNIFORM_TYPE_INT, 1);
			cf_material_set_uniform_cs(hrc.mat_trace, "u_work_h", params + 5, CF_UNIFORM_TYPE_INT, 1);
			float mip_level = hrc.mip_level;
			cf_material_set_uniform_cs(hrc.mat_trace, "u_mip_level", &mip_level, CF_UNIFORM_TYPE_FLOAT, 1);

			CF_ComputeDispatch d = cf_compute_dispatch_defaults(
				hrc_div_ceil(hrc_t_width(rot_w, level), HRC_WG),
				hrc_div_ceil(rot_h, HRC_WG),
				1
			);
			CF_StorageBuffer rw[2] = { hrc.t_rad[level], hrc.t_trn[level] };
			d.rw_buffers = rw;
			d.rw_buffer_count = 2;
			cf_dispatch_compute(hrc.cs_trace, hrc.mat_trace, d);
		}

		// Extend T_cutoff..T_N.
		for (int level = trace_max; level <= hrc.work_N; level++) {
			int rot_w = dim;
			int rot_h = dim;
			int prev_w = hrc_t_width(rot_w, level - 1);
			int curr_w = hrc_t_width(rot_w, level);

			int params[5] = { level, rot_w, rot_h, prev_w, curr_w };
			cf_material_set_uniform_cs(hrc.mat_extend, "u_cascade", params + 0, CF_UNIFORM_TYPE_INT, 1);
			cf_material_set_uniform_cs(hrc.mat_extend, "u_world_w", params + 1, CF_UNIFORM_TYPE_INT, 1);
			cf_material_set_uniform_cs(hrc.mat_extend, "u_world_h", params + 2, CF_UNIFORM_TYPE_INT, 1);
			cf_material_set_uniform_cs(hrc.mat_extend, "u_prev_w", params + 3, CF_UNIFORM_TYPE_INT, 1);
			cf_material_set_uniform_cs(hrc.mat_extend, "u_curr_w", params + 4, CF_UNIFORM_TYPE_INT, 1);

			CF_ComputeDispatch d = cf_compute_dispatch_defaults(
				hrc_div_ceil(curr_w, HRC_WG),
				hrc_div_ceil(rot_h, HRC_WG),
				1
			);
			CF_StorageBuffer ro[2] = { hrc.t_rad[level - 1], hrc.t_trn[level - 1] };
			d.ro_buffers = ro;
			d.ro_buffer_count = 2;
			CF_StorageBuffer rw[2] = { hrc.t_rad[level], hrc.t_trn[level] };
			d.rw_buffers = rw;
			d.rw_buffer_count = 2;
			cf_dispatch_compute(hrc.cs_extend, hrc.mat_extend, d);
		}

		// Merge R_{N-1} down to R_0. Final merge writes directly to frustum[j].
		int rot_w = dim;
		int rot_h = dim;
		int r_ping = 0;
		for (int i = hrc.work_N - 1; i >= 0; i--) {
			int t_curr_w = hrc_t_width(rot_w, i);
			int t_next_w = hrc_t_width(rot_w, i + 1);
			int r_prev_w = hrc_r_width(rot_w, i + 1);
			int r_curr_w = hrc_r_width(rot_w, i);

			int params[7] = { i, rot_w, rot_h, t_curr_w, t_next_w, r_prev_w, r_curr_w };
			cf_material_set_uniform_cs(hrc.mat_merge, "u_cascade", params + 0, CF_UNIFORM_TYPE_INT, 1);
			cf_material_set_uniform_cs(hrc.mat_merge, "u_world_w", params + 1, CF_UNIFORM_TYPE_INT, 1);
			cf_material_set_uniform_cs(hrc.mat_merge, "u_world_h", params + 2, CF_UNIFORM_TYPE_INT, 1);
			cf_material_set_uniform_cs(hrc.mat_merge, "u_t_curr_w", params + 3, CF_UNIFORM_TYPE_INT, 1);
			cf_material_set_uniform_cs(hrc.mat_merge, "u_t_next_w", params + 4, CF_UNIFORM_TYPE_INT, 1);
			cf_material_set_uniform_cs(hrc.mat_merge, "u_r_prev_w", params + 5, CF_UNIFORM_TYPE_INT, 1);
			cf_material_set_uniform_cs(hrc.mat_merge, "u_r_curr_w", params + 6, CF_UNIFORM_TYPE_INT, 1);

			CF_StorageBuffer r_prev = (i == hrc.work_N - 1) ? hrc.r_zero : hrc.r_rad[1 - r_ping];
			CF_StorageBuffer r_out = (i == 0) ? hrc.frustum[j] : hrc.r_rad[r_ping];

			CF_ComputeDispatch d = cf_compute_dispatch_defaults(
				hrc_div_ceil(r_curr_w, HRC_WG),
				hrc_div_ceil(rot_h, HRC_WG),
				1
			);
			CF_StorageBuffer ro[6] = {
				hrc.t_rad[i], hrc.t_trn[i],
				hrc.t_rad[i + 1], hrc.t_trn[i + 1],
				r_prev,
				hrc.merge_weights[i]
			};
			d.ro_buffers = ro;
			d.ro_buffer_count = 6;
			CF_StorageBuffer rw[1] = { r_out };
			d.rw_buffers = rw;
			d.rw_buffer_count = 1;
			cf_dispatch_compute(hrc.cs_merge, hrc.mat_merge, d);
			r_ping = 1 - r_ping;
		}
	}

	// Sum quadrants: frustum[0..3] -> radiance_preblur.
	{
		int params[2] = { dim, dim };
		cf_material_set_uniform_cs(hrc.mat_sum_quadrants, "u_world_w", params + 0, CF_UNIFORM_TYPE_INT, 1);
		cf_material_set_uniform_cs(hrc.mat_sum_quadrants, "u_world_h", params + 1, CF_UNIFORM_TYPE_INT, 1);

		CF_ComputeDispatch d = cf_compute_dispatch_defaults(
			hrc_div_ceil(dim, HRC_WG),
			hrc_div_ceil(dim, HRC_WG),
			1
		);
		CF_StorageBuffer ro[4] = {
			hrc.frustum[0], hrc.frustum[1],
			hrc.frustum[2], hrc.frustum[3]
		};
		d.ro_buffers = ro;
		d.ro_buffer_count = 4;
		CF_Texture rw_tex[1] = { hrc.radiance_preblur };
		d.rw_textures = rw_tex;
		d.rw_texture_count = 1;
		cf_dispatch_compute(hrc.cs_sum_quadrants, hrc.mat_sum_quadrants, d);
	}

	// Blur: radiance_preblur -> radiance.
	{
		cf_material_set_texture_cs(hrc.mat_blur, "u_in_radiance", hrc.radiance_preblur);
		cf_material_set_texture_cs(hrc.mat_blur, "u_transmittance", trans_tex);
		float src_texel_size[2] = { 1.0f / (float)dim, 1.0f / (float)dim };
		cf_material_set_uniform_cs(hrc.mat_blur, "u_src_texel_size", src_texel_size, CF_UNIFORM_TYPE_FLOAT2, 1);
		float mip_level = hrc.mip_level;
		cf_material_set_uniform_cs(hrc.mat_blur, "u_mip_level", &mip_level, CF_UNIFORM_TYPE_FLOAT, 1);

		CF_ComputeDispatch d = cf_compute_dispatch_defaults(
			hrc_div_ceil(dim, HRC_BLUR_WG),
			hrc_div_ceil(dim, HRC_BLUR_WG),
			1
		);
		CF_Texture rw_tex[1] = { hrc.radiance };
		d.rw_textures = rw_tex;
		d.rw_texture_count = 1;
		cf_dispatch_compute(hrc.cs_blur, hrc.mat_blur, d);
	}

	// Composite: radiance/preblur/frustums -> fluence (rgba8).
	{
		int debug = hrc.debug_mode <= 5 ? hrc.debug_mode : 0;
		int params[3] = { dim, dim, debug };
		cf_material_set_uniform_cs(hrc.mat_composite, "u_world_w", params + 0, CF_UNIFORM_TYPE_INT, 1);
		cf_material_set_uniform_cs(hrc.mat_composite, "u_world_h", params + 1, CF_UNIFORM_TYPE_INT, 1);
		cf_material_set_uniform_cs(hrc.mat_composite, "u_debug_mode", params + 2, CF_UNIFORM_TYPE_INT, 1);

		CF_ComputeDispatch d = cf_compute_dispatch_defaults(
			hrc_div_ceil(dim, HRC_WG),
			hrc_div_ceil(dim, HRC_WG),
			1
		);
		CF_Texture ro_tex[2] = { hrc.radiance, hrc.radiance_preblur };
		d.ro_textures = ro_tex;
		d.ro_texture_count = 2;
		CF_StorageBuffer ro[4] = {
			hrc.frustum[0], hrc.frustum[1],
			hrc.frustum[2], hrc.frustum[3]
		};
		d.ro_buffers = ro;
		d.ro_buffer_count = 4;
		CF_Texture rw_tex[1] = { fluence_tex };
		d.rw_textures = rw_tex;
		d.rw_texture_count = 1;
		cf_dispatch_compute(hrc.cs_composite, hrc.mat_composite, d);
	}
}

//--------------------------------------------------------------------------------------------------
// Demo scene state.

typedef struct OrbLight
{
	float radius;
	float speed;
	float angle;
	CF_Color color;
} OrbLight;

float time_acc = 0.0f;

OrbLight orbs[4];

void scene_init()
{
	// Pre-linearize colors so the trace shader reads linear values directly.
	orbs[0] = (OrbLight){ 140.0f, 0.7f, 0.0f,
		cf_make_color_rgb_f(powf(1.0f, 2.2f), powf(0.2f, 2.2f), powf(0.1f, 2.2f)) };
	orbs[1] = (OrbLight){ 160.0f, -0.5f, CF_PI * 0.5f,
		cf_make_color_rgb_f(powf(0.1f, 2.2f), powf(1.0f, 2.2f), powf(0.2f, 2.2f)) };
	orbs[2] = (OrbLight){ 120.0f, 0.9f, CF_PI,
		cf_make_color_rgb_f(powf(0.2f, 2.2f), powf(0.3f, 2.2f), powf(1.0f, 2.2f)) };
	orbs[3] = (OrbLight){ 180.0f, -0.3f, CF_PI * 1.5f,
		cf_make_color_rgb_f(powf(1.0f, 2.2f), powf(0.9f, 2.2f), powf(0.1f, 2.2f)) };
}

//--------------------------------------------------------------------------------------------------
// Input handling.

void handle_input()
{
	if (cf_key_just_pressed(CF_KEY_D)) {
		hrc.debug_mode = (hrc.debug_mode + 1) % HRC_NUM_DEBUG;
	}
}

//--------------------------------------------------------------------------------------------------
// Drawing helpers.

void begin_canvas_draw()
{
	cf_draw_push();
	float ws = (float)HRC_DIM;
	float half = ws * 0.5f;
	cf_draw_TSR_absolute(cf_v2(0, 0), cf_v2(1, 1), 0);
	cf_draw_projection(cf_ortho_2d(0, 0, ws, ws));
	cf_draw_translate(-half, -half);
}

void end_canvas_draw()
{
	cf_draw_pop();
}

void draw_circle_at(float x, float y, float r)
{
	cf_draw_circle_fill2(cf_v2(x, y), r);
}

void push_f16_render_state()
{
	CF_RenderState rs = cf_render_state_defaults();
	rs.blend.pixel_format = CF_PIXEL_FORMAT_R16G16B16A16_FLOAT;
	cf_draw_push_render_state(rs);
}

// Per-frame light positions (computed once, drawn into both canvases).
typedef struct Light
{
	float x, y, r;
	CF_Color color;
} Light;

#define MAX_LIGHTS 16
Light frame_lights[MAX_LIGHTS];
int frame_light_count;

void update_lights()
{
	float half = (float)HRC_DIM * 0.5f;
	float ws = (float)HRC_DIM;
	float dt = CF_DELTA_TIME;
	time_acc += dt;
	frame_light_count = 0;

	// Orbiting lights.
	for (int i = 0; i < 4; i++) {
		orbs[i].angle += orbs[i].speed * dt;
		float cx = half + cosf(orbs[i].angle) * orbs[i].radius;
		float cy = half + sinf(orbs[i].angle) * orbs[i].radius;
		frame_lights[frame_light_count++] = (Light){ cx, cy, 15.0f, orbs[i].color };
	}

	// Corner accent lights (pre-linearized).
	{
		float margin = 60.0f;
		CF_Color corners[4] = {
			cf_make_color_rgb_f(powf(0.0f, 2.2f), powf(0.4f, 2.2f), powf(0.4f, 2.2f)),
			cf_make_color_rgb_f(powf(0.5f, 2.2f), powf(0.3f, 2.2f), powf(0.0f, 2.2f)),
			cf_make_color_rgb_f(powf(0.4f, 2.2f), powf(0.1f, 2.2f), powf(0.3f, 2.2f)),
			cf_make_color_rgb_f(powf(0.2f, 2.2f), powf(0.5f, 2.2f), powf(0.0f, 2.2f)),
		};
		float lx[4] = { margin, ws - margin, margin, ws - margin };
		float ly[4] = { margin, margin, ws - margin, ws - margin };
		for (int i = 0; i < 4; i++) {
			frame_lights[frame_light_count++] = (Light){ lx[i], ly[i], 8.0f, corners[i] };
		}
	}
}

// Draw all lights as colored circles (for emission canvas).
void draw_lights_emissive()
{
	for (int i = 0; i < frame_light_count; i++) {
		cf_draw_push_color(frame_lights[i].color);
		draw_circle_at(frame_lights[i].x, frame_lights[i].y, frame_lights[i].r);
		cf_draw_pop_color();
	}
}

// Draw all lights as black circles (for transmittance canvas).
// Light sources must block to emit -- they need opacity in the transmittance canvas.
void draw_lights_blocking()
{
	cf_draw_push_color(cf_make_color_rgb_f(0.0f, 0.0f, 0.0f));
	for (int i = 0; i < frame_light_count; i++) {
		draw_circle_at(frame_lights[i].x, frame_lights[i].y, frame_lights[i].r);
	}
	cf_draw_pop_color();
}

//--------------------------------------------------------------------------------------------------
// Draw emission (lights on black canvas, linear colors).

void draw_emission()
{
	begin_canvas_draw();
	push_f16_render_state();

	draw_lights_emissive();

	cf_draw_pop_render_state();
	cf_clear_color(0.0f, 0.0f, 0.0f, 0.0f);
	cf_render_to(hrc.emission, true);
	cf_generate_mipmaps(cf_canvas_get_target(hrc.emission));
	end_canvas_draw();
}

//--------------------------------------------------------------------------------------------------
// Draw transmittance (white canvas, multiplicative blend, black = opaque).
//
// Transmittance model: clear to white (fully transparent), draw black shapes
// with multiplicative blending. Black at alpha=1 -> result = 0 (fully opaque).
// This is the inverse of the old absorption model (white shapes on black canvas).

void draw_transmittance()
{
	begin_canvas_draw();

	// Multiplicative blend state: rgb_result = dst * src + dst * (1 - src_alpha)
	// Drawing black (0,0,0) at alpha=1: result = dst*0 + dst*0 = 0 (opaque)
	// Drawing nothing: result = white (transparent)
	CF_RenderState rs = cf_render_state_defaults();
	rs.blend.enabled = true;
	rs.blend.pixel_format = CF_PIXEL_FORMAT_R16G16B16A16_FLOAT;
	rs.blend.rgb_src_blend_factor = CF_BLENDFACTOR_DST_COLOR;
	rs.blend.rgb_dst_blend_factor = CF_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
	rs.blend.rgb_op = CF_BLEND_OP_ADD;
	rs.blend.alpha_src_blend_factor = CF_BLENDFACTOR_ONE;
	rs.blend.alpha_dst_blend_factor = CF_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
	rs.blend.alpha_op = CF_BLEND_OP_ADD;
	cf_draw_push_render_state(rs);

	cf_draw_push_color(cf_make_color_rgb_f(0.0f, 0.0f, 0.0f));

	float ws = (float)HRC_DIM;
	float half = ws * 0.5f;

	// 5 circular pillars in quincunx pattern.
	draw_circle_at(half, half, 25.0f);
	draw_circle_at(half - 110.0f, half - 110.0f, 20.0f);
	draw_circle_at(half + 110.0f, half - 110.0f, 20.0f);
	draw_circle_at(half - 110.0f, half + 110.0f, 20.0f);
	draw_circle_at(half + 110.0f, half + 110.0f, 20.0f);

	// 2 angled wall segments.
	cf_draw_quad_fill2(
		cf_v2(100.0f, 200.0f), cf_v2(110.0f, 200.0f),
		cf_v2(200.0f, 310.0f), cf_v2(190.0f, 310.0f), 0
	);
	cf_draw_quad_fill2(
		cf_v2(400.0f, 200.0f), cf_v2(390.0f, 200.0f),
		cf_v2(300.0f, 310.0f), cf_v2(310.0f, 310.0f), 0
	);

	// 3 small scattered square blocks.
	cf_draw_quad_fill(cf_make_aabb(cf_v2(340.0f, 400.0f), cf_v2(365.0f, 425.0f)), 0);
	cf_draw_quad_fill(cf_make_aabb(cf_v2(150.0f, 380.0f), cf_v2(175.0f, 405.0f)), 0);
	cf_draw_quad_fill(cf_make_aabb(cf_v2(370.0f, 120.0f), cf_v2(395.0f, 145.0f)), 0);

	cf_draw_pop_color();

	// Light sources must block to emit.
	draw_lights_blocking();

	cf_draw_pop_render_state();

	// Clear to white (fully transparent) before drawing.
	cf_clear_color(1.0f, 1.0f, 1.0f, 1.0f);
	cf_render_to(hrc.transmittance, true);
	cf_generate_mipmaps(cf_canvas_get_target(hrc.transmittance));
	end_canvas_draw();
}

//--------------------------------------------------------------------------------------------------
// Display fluence onto screen.

void display_fluence()
{
	CF_Canvas display = hrc.fluence;
	if (hrc.debug_mode == 6) display = hrc.emission;
	else if (hrc.debug_mode == 7) display = hrc.transmittance;

	float ws = (float)HRC_DIM;
	cf_draw_canvas(display, cf_v2(0, 0), cf_v2(ws, ws));
}

//--------------------------------------------------------------------------------------------------
// Entry point.

int main(int argc, char* argv[])
{
	cf_make_app("HRC - Holographic Radiance Cascades", 0, 0, 0, HRC_DIM, HRC_DIM, CF_APP_OPTIONS_WINDOW_POS_CENTERED_BIT, argv[0]);

	hrc_init();
	scene_init();

	while (cf_app_is_running()) {
		cf_app_update(NULL);

		char title[128];
		snprintf(title, sizeof(title), "HRC - %.0f fps", cf_app_get_smoothed_framerate());
		cf_app_set_title(title);

		handle_input();
		update_lights();
		draw_emission();
		draw_transmittance();
		hrc_compute();
		display_fluence();
		cf_app_draw_onto_screen(true);
	}

	hrc_shutdown();
	cf_destroy_app();
	return 0;
}
