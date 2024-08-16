/*
	Cute Framework
	Copyright (C) 2024 Randy Gaul https://randygaul.github.io/

	This software is dual-licensed with zlib or Unlicense, check LICENSE.txt for more info
*/

#include <cute_app.h>
#include <cute_alloc.h>
#include <cute_audio.h>
#include <cute_multithreading.h>
#include <cute_file_system.h>
#include <cute_c_runtime.h>
#include <cute_draw.h>
#include <cute_time.h>

#include <internal/cute_alloc_internal.h>
#include <internal/cute_app_internal.h>
#include <internal/cute_input_internal.h>
#include <internal/cute_graphics_internal.h>
#include <internal/cute_draw_internal.h>
#include <internal/cute_png_cache_internal.h>
#include <internal/cute_aseprite_cache_internal.h>
#include <internal/cute_imgui_internal.h>

#include <imgui/backends/imgui_impl_sdl3.h>

#include <data/fonts/calibri.h>

#include <SDL3/SDL.h>

#define CUTE_SOUND_FORCE_SDL
#include <cute/cute_sound.h>

CF_STATIC_ASSERT(sizeof(uint64_t) >= sizeof(void*), "Must be equal for opaque id implementations throughout CF.");

static void s_init_video()
{
	static bool init = false;
	if (init) return;
	init = true;
	SDL_Init(SDL_INIT_VIDEO);
}

int cf_display_count()
{
	s_init_video();
	int count = 0;
	SDL_GetDisplays(&count);
	return count;
}

int cf_display_x(int display_index)
{
	s_init_video();
	SDL_Rect rect;
	SDL_GetDisplayBounds(display_index, &rect);
	return rect.x;
}

int cf_display_y(int display_index)
{
	s_init_video();
	SDL_Rect rect;
	SDL_GetDisplayBounds(display_index, &rect);
	return rect.y;
}

int cf_display_width(int display_index)
{
	s_init_video();
	SDL_Rect rect;
	SDL_GetDisplayBounds(display_index, &rect);
	return rect.w;
}

int cf_display_height(int display_index)
{
	s_init_video();
	SDL_Rect rect;
	SDL_GetDisplayBounds(display_index, &rect);
	return rect.h;
}

float cf_display_refresh_rate(int display_index)
{
	s_init_video();
	const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(display_index);
	return mode->refresh_rate;
}

CF_Rect cf_display_bounds(int display_index)
{
	s_init_video();
	SDL_Rect rect;
	SDL_GetDisplayBounds(display_index, &rect);
	CF_Rect result = { rect.x, rect.y, rect.w, rect.h };
	return result;
}

const char* cf_display_name(int display_index)
{
	s_init_video();
	return SDL_GetDisplayName(display_index);
}

CF_DisplayOrientation cf_display_orientation(int display_index)
{
	s_init_video();
	SDL_DisplayOrientation orientation = SDL_GetCurrentDisplayOrientation(display_index);
	switch (orientation) {
	default:
	case SDL_ORIENTATION_UNKNOWN: return CF_DISPLAY_ORIENTATION_UNKNOWN;
	case SDL_ORIENTATION_LANDSCAPE: return CF_DISPLAY_ORIENTATION_LANDSCAPE;
	case SDL_ORIENTATION_LANDSCAPE_FLIPPED: return CF_DISPLAY_ORIENTATION_LANDSCAPE_FLIPPED;
	case SDL_ORIENTATION_PORTRAIT: return CF_DISPLAY_ORIENTATION_PORTRAIT;
	case SDL_ORIENTATION_PORTRAIT_FLIPPED: return CF_DISPLAY_ORIENTATION_PORTRAIT_FLIPPED;
	}
}

CF_GLOBAL CF_App* app;

using namespace Cute;

struct Vertex
{
	float x, y;
	float u, v;
};

static void s_quad(float x, float y, float sx, float sy, Vertex quad[6])
{
	quad[0].x = -0.5f; quad[0].y =  0.5f; quad[0].u = 0; quad[0].v = 1;
	quad[1].x =  0.5f; quad[1].y = -0.5f; quad[1].u = 1; quad[1].v = 0;
	quad[2].x =  0.5f; quad[2].y =  0.5f; quad[2].u = 1; quad[2].v = 1;

	quad[3].x = -0.5f; quad[3].y =  0.5f; quad[3].u = 0; quad[3].v = 1;
	quad[4].x = -0.5f; quad[4].y = -0.5f; quad[4].u = 0; quad[4].v = 0;
	quad[5].x =  0.5f; quad[5].y = -0.5f; quad[5].u = 1; quad[5].v = 0;

	for (int i = 0; i < 6; ++i) {
		quad[i].x = quad[i].x * sx + x;
		quad[i].y = quad[i].y * sy + y;
	}
}

// Embedded white_pixel
int default_png_sz = 81;
unsigned char default_png_data[81] = {
	0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a,0x00,0x00,0x00,0x0d,0x49,0x48,0x44,0x52,
	0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,0x08,0x06,0x00,0x00,0x00,0x1f,0x15,0xc4,
	0x89,0x00,0x00,0x00,0x01,0x73,0x52,0x47,0x42,0x00,0xae,0xce,0x1c,0xe9,0x00,0x00,
	0x00,0x0b,0x49,0x44,0x41,0x54,0x08,0x99,0x63,0xf8,0x0f,0x04,0x00,0x09,0xfb,0x03,
	0xfd,0xe3,0x55,0xf2,0x9c,0x00,0x00,0x00,0x00,0x49,0x45,0x4e,0x44,0xae,0x42,0x60,
	0x82
};

static void s_canvas(int w, int h)
{
	CF_CanvasParams params = cf_canvas_defaults(w, h);
	if (app->offscreen_canvas.id) {
		cf_destroy_canvas(app->offscreen_canvas);
	}
	app->offscreen_canvas = cf_make_canvas(params);
	app->canvas_w = w;
	app->canvas_h = h;
	cf_material_set_texture_fs(app->backbuffer_material, "u_image", cf_canvas_get_target(app->offscreen_canvas));

	// Create the app's canvas manually, since it requires some special APIs from SDL regarding the backbuffer.
	SDL_GpuTextureCreateInfo tex_info = SDL_GpuTextureCreateInfoDefaults(w, h);
	tex_info.format = SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT;
	tex_info.usageFlags = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET_BIT;
	SDL_GpuTexture* depth_stencil = NULL;
	if (app->use_depth_stencil) {
		depth_stencil = SDL_GpuCreateTexture(app->device, &tex_info);
		CF_ASSERT(depth_stencil);
	}

	SDL_GpuSamplerCreateInfo sampler_info = SDL_GpuSamplerCreateInfoDefaults();
	sampler_info.minFilter = SDL_GPU_FILTER_NEAREST;
	sampler_info.magFilter = SDL_GPU_FILTER_NEAREST;
	sampler_info.addressModeU = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
	sampler_info.addressModeV = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
	sampler_info.addressModeW = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
	SDL_GpuSampler* sampler = SDL_GpuCreateSampler(app->device, &sampler_info);
	CF_ASSERT(sampler);

	CF_TextureInternal* depth_stencil_internal = NULL;
	if (app->use_depth_stencil) {
		depth_stencil_internal = CF_NEW(CF_TextureInternal);
		CF_MEMSET(depth_stencil_internal, 0, sizeof(*depth_stencil_internal));
		depth_stencil_internal->w = w;
		depth_stencil_internal->h = h;
		depth_stencil_internal->filter = SDL_GPU_FILTER_NEAREST;
		depth_stencil_internal->format = SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT;
		depth_stencil_internal->sampler = sampler;
		depth_stencil_internal->tex = depth_stencil;
	}

	CF_TextureInternal* tex_internal = CF_NEW(CF_TextureInternal);
	CF_MEMSET(tex_internal, 0, sizeof(*tex_internal));
	tex_internal->w = w;
	tex_internal->h = h;
	tex_internal->filter = SDL_GPU_FILTER_NEAREST;
	tex_internal->format = SDL_GpuGetSwapchainTextureFormat(app->device, app->window);
	tex_internal->sampler = sampler;
	tex_internal->tex = NULL; // The actual SDL texture handle will be filled later via SDL_GpuAcquireSwapchainTexture.
	
	if (app->canvas.id) {
		if (app->use_depth_stencil) SDL_GpuReleaseTexture(app->device, ((CF_CanvasInternal*)app->canvas.id)->depth_stencil);
		SDL_GpuReleaseSampler(app->device, ((CF_CanvasInternal*)app->canvas.id)->sampler);
	}

	CF_CanvasInternal* canvas = CF_NEW(CF_CanvasInternal);
	CF_MEMSET(canvas, 0, sizeof(*canvas));
	canvas->cf_depth_stencil = { (uint64_t)depth_stencil_internal };
	canvas->cf_texture = { (uint64_t)tex_internal };
	canvas->depth_stencil = depth_stencil;
	canvas->texture = NULL; // The actual SDL texture handle will be filled later via SDL_GpuAcquireSwapchainTexture.
	canvas->sampler = sampler;

	app->canvas.id = { (uint64_t)canvas };
}

CF_Result cf_make_app(const char* window_title, int display_index, int x, int y, int w, int h, CF_AppOptionFlags options, const char* argv0)
{
	bool use_dx11 = options & APP_OPTIONS_GFX_D3D11_BIT;
	bool use_dx12 = options & APP_OPTIONS_GFX_D3D12_BIT;
	bool use_metal = options & APP_OPTIONS_GFX_METAL_BIT;
	bool use_vulkan = options & APP_OPTIONS_GFX_VULKAN_BIT;
	bool use_gfx = !(options & APP_OPTIONS_NO_GFX_BIT);

	// Ensure the user selected only one backend, if they selected one at all.
	if (use_dx11) {
		CF_ASSERT(!use_dx12);
		CF_ASSERT(!use_metal);
		CF_ASSERT(!use_vulkan);
	}
	if (use_dx12) {
		CF_ASSERT(!use_dx11);
		CF_ASSERT(!use_metal);
		CF_ASSERT(!use_vulkan);
	}
	if (use_metal) {
		CF_ASSERT(!use_dx11);
		CF_ASSERT(!use_dx12);
		CF_ASSERT(!use_vulkan);
	}
	if (use_vulkan) {
		CF_ASSERT(!use_dx11);
		CF_ASSERT(!use_dx12);
		CF_ASSERT(!use_metal);
	}

#ifdef CF_EMSCRIPTEN
	Uint32 sdl_options = SDL_INIT_EVENTS | SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMEPAD;
#else
	Uint32 sdl_options = SDL_INIT_EVENTS | SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMEPAD | SDL_INIT_HAPTIC;
	if (options & APP_OPTIONS_NO_GFX_BIT) {
		sdl_options &= ~SDL_INIT_VIDEO;
	}
#endif
	if (!(options & APP_OPTIONS_NO_AUDIO_BIT)) {
		SDL_Init(SDL_INIT_AUDIO);
	}

	// Turn on high DPI support for all platforms.
	options |= SDL_WINDOW_HIGH_PIXEL_DENSITY;
	
	if (SDL_Init(sdl_options)) {
		return cf_result_error("SDL_Init failed");
	}

	SDL_GpuDevice* device = NULL;
	if (use_gfx) {
		// Some backends don't support window size of zero.
		w = w <= 0 ? 1 : w;
		h = h <= 0 ? 1 : h;

		// Create the GPU device.
		SDL_PropertiesID props = SDL_CreateProperties();
		if (use_dx11) SDL_SetStringProperty(props, SDL_PROP_GPU_CREATEDEVICE_NAME_STRING, "D3D11");
		else if (use_dx12) SDL_SetStringProperty(props, SDL_PROP_GPU_CREATEDEVICE_NAME_STRING, "D3D12");
		else if (use_metal) SDL_SetStringProperty(props, SDL_PROP_GPU_CREATEDEVICE_NAME_STRING, "Metal");
		else if (use_vulkan) SDL_SetStringProperty(props, SDL_PROP_GPU_CREATEDEVICE_NAME_STRING, "Vulkan");
		device = SDL_GpuCreateDevice(true, false, props);
		SDL_DestroyProperties(props);
		if (!device) {
			return cf_result_error("Failed to create GPU Device.");
		}
	}

	Uint32 flags = 0;
	if (use_metal) flags |= SDL_WINDOW_METAL;
	if (options & APP_OPTIONS_FULLSCREEN_BIT) flags |= SDL_WINDOW_FULLSCREEN;
	if (options & APP_OPTIONS_RESIZABLE_BIT) flags |= SDL_WINDOW_RESIZABLE;
	if (options & APP_OPTIONS_HIDDEN_BIT) flags |= (SDL_WINDOW_HIDDEN | SDL_WINDOW_MINIMIZED);

	SDL_Window* window;
	SDL_PropertiesID props = SDL_CreateProperties();
	SDL_SetStringProperty(props, SDL_PROP_WINDOW_CREATE_TITLE_STRING, window_title);
	SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, w);
	SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, h);
	SDL_SetNumberProperty(props, "flags", flags);
	if (options & APP_OPTIONS_WINDOW_POS_CENTERED_BIT) {
		SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_X_NUMBER, SDL_WINDOWPOS_CENTERED_DISPLAY(display_index));
		SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_Y_NUMBER, SDL_WINDOWPOS_CENTERED_DISPLAY(display_index));
	} else {
		int x_offset = display_x(display_index);
		int y_offset = display_y(display_index);
		SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_X_NUMBER, x_offset+x);
		SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_Y_NUMBER, y_offset+y);
	}
	window = SDL_CreateWindowWithProperties(props);
	SDL_DestroyProperties(props);
	CF_App* app = (CF_App*)CF_ALLOC(sizeof(CF_App));
	CF_PLACEMENT_NEW(app) CF_App;
	app->options = options;
	app->window = window;
	app->w = w;
	app->h = h;
	SDL_GetWindowPosition(app->window, &app->x, &app->y);
	list_init(&app->joypads);
	::app = app;
	cf_make_aseprite_cache();
	cf_make_png_cache();

	if (use_gfx) {
		app->device = device;
		SDL_GpuClaimWindow(app->device, app->window, SDL_GPU_SWAPCHAINCOMPOSITION_SDR, SDL_GPU_PRESENTMODE_IMMEDIATE);
		app->cmd = SDL_GpuAcquireCommandBuffer(app->device);
		cf_load_internal_shaders();
		cf_make_draw();
		
		// Setup the backbuffer fullscreen mesh and canvas.
		{
			app->backbuffer_quad = cf_make_mesh(sizeof(Vertex) * 6);
			CF_VertexAttribute attrs[2] = { };
			attrs[0].name = "in_posH";
			attrs[0].format = CF_VERTEX_FORMAT_FLOAT2;
			attrs[0].offset = CF_OFFSET_OF(Vertex, x);
			attrs[1].name = "in_uv";
			attrs[1].format = CF_VERTEX_FORMAT_FLOAT2;
			attrs[1].offset = CF_OFFSET_OF(Vertex, u);
			cf_mesh_set_attributes(app->backbuffer_quad, attrs, CF_ARRAY_SIZE(attrs), sizeof(Vertex));
			Vertex quad[6];
			s_quad(0, 0, 2, -2, quad);
			cf_mesh_update_vertex_data(app->backbuffer_quad, quad, 6);
		}
		app->backbuffer_material = cf_make_material();
		s_canvas(app->w, app->h);

		// Load up a default image of 1x1 white pixel.
		// Used in various places as a placeholder or default.
		CF_Png img;
		cf_png_cache_load_from_memory("cf_default_png", default_png_data, (size_t)default_png_sz, &img);
		app->default_image_id = img.id;
		CF_ASSERT(app->default_image_id == CF_PNG_ID_RANGE_LO);

		// Create the default font.
		make_font_from_memory(calibri_data, calibri_sz, "Calibri");
		SDL_GpuSubmit(app->cmd);
		app->cmd = NULL;
	}

#ifdef CF_WINDOWS
	app->platform_handle = SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);;
#else
	void* hwnd = NULL;
#endif

	app->gfx_enabled = use_gfx;

	if (!(options & APP_OPTIONS_NO_AUDIO_BIT)) {
		int more_on_emscripten = 1;
	#ifdef CF_EMSCRIPTEN
		more_on_emscripten = 4;
	#endif
		cs_error_t err = cs_init(NULL, 44100, 1024 * more_on_emscripten, NULL);
		if (err == CUTE_SOUND_ERROR_NONE) {
	#ifndef CF_EMSCRIPTEN
			cs_spawn_mix_thread();
			app->spawned_mix_thread = true;
	#endif // CF_EMSCRIPTEN
			app->audio_needs_updates = true;
			//cs_cull_duplicates(true); -- https://github.com/RandyGaul/cute_framework/issues/172
		} else {
			CF_Result result;
			result.code = -1;
			result.details = cs_error_as_string(err);
			return result;
		}
	}

	int num_threads_to_spawn = cf_core_count() - 1;
	if (num_threads_to_spawn) {
		app->threadpool = cf_make_threadpool(num_threads_to_spawn);
	}

	CF_Result err = cf_fs_init(argv0);
	if (cf_is_error(err)) {
		CF_ASSERT(0);
	} else if (!(options & APP_OPTIONS_FILE_SYSTEM_DONT_DEFAULT_MOUNT_BIT)) {
		// Put the base directory (the path to the exe) onto the file system search path.
		cf_fs_mount(cf_fs_get_base_directory(), "", true);
	}

	// Initialize a default ECS world.
	app->world = cf_make_world();
	app->worlds.add(app->world);

	return cf_result_success();
}

void cf_destroy_app()
{
	if (app->using_imgui) {
		ImGui_ImplSDL3_Shutdown();
		cf_imgui_shutdown();
		app->using_imgui = false;
	}
	if (app->gfx_enabled) {
		cf_destroy_draw();
		cf_unload_shader_compiler();
		cf_destroy_canvas(app->offscreen_canvas);
		cf_destroy_mesh(app->backbuffer_quad);
		cf_destroy_material(app->backbuffer_material);
		if (app->canvas_blit_init) {
			cf_destroy_mesh(app->blit_mesh);
			cf_destroy_material(app->blit_material);
			cf_destroy_shader(app->blit_shader);
		}
		CF_CanvasInternal* canvas = (CF_CanvasInternal*)app->canvas.id;
		SDL_GpuReleaseTexture(app->device, canvas->depth_stencil);
		SDL_GpuReleaseSampler(app->device, canvas->sampler);
		if (app->use_depth_stencil) cf_free((void*)canvas->cf_depth_stencil.id);
		cf_free(canvas);
	}
	cf_destroy_aseprite_cache();
	cf_destroy_png_cache();
	// Mainly here to cleanup the default world, but, as a convenience we can just clean them all up.
	for (int i = 0; i < app->worlds.count(); ++i) {
		cf_destroy_world(app->worlds[i]);
	}
	cs_shutdown();
	destroy_mutex(&app->on_sound_finish_mutex);
	if (app->device) SDL_GpuUnclaimWindow(app->device, app->window);
	SDL_DestroyWindow(app->window);
	if (app->device) SDL_GpuDestroyDevice(app->device);
	SDL_Quit();
	destroy_threadpool(app->threadpool);
	cs_shutdown();
	CF_Image* easy_sprites = app->easy_sprites.items();
	for (int i = 0; i < app->easy_sprites.count(); ++i) {
		cf_image_free(&easy_sprites[i]);
	}
	app->~CF_App();
	CF_FREE(app);
	cf_fs_destroy();
}

bool cf_app_is_running()
{
	return app->running;
}

void cf_app_signal_shutdown()
{
	app->running = 0;
}

static void s_on_update(void* udata)
{
	cf_pump_input_msgs();
	if (app->audio_needs_updates) {
		cs_update(DELTA_TIME);
		if (app->on_sound_finish_single_threaded) {
			mutex_lock(&app->on_sound_finish_mutex);
			Array<CF_Sound> on_finish = app->on_sound_finish_queue;
			app->on_sound_finish_queue.clear();
			mutex_unlock(&app->on_sound_finish_mutex);
			for (int i = 0; i < on_finish.size(); ++i) {
				app->on_sound_finish(on_finish[i], app->on_sound_finish_udata);
			}
			if (app->on_music_finish && app->on_music_finish_signal) {
				app->on_music_finish_signal = false;
				app->on_music_finish(app->on_music_finish_udata);
			}
		}
	}
	if (app->user_on_update) app->user_on_update(udata);
}

void cf_app_update(CF_OnUpdateFn* on_update)
{
	if (app->gfx_enabled) {
		// Deal with DPI scaling.
		int pw = 0, ph = 0;
		SDL_GetWindowSizeInPixels(app->window, &pw, &ph);
		app->dpi_scale = (float)ph / (float)app->h;
		app->dpi_scale_was_changed = false;
		if (app->dpi_scale != app->dpi_scale_prev) {
			app->dpi_scale_was_changed = true;
			app->dpi_scale_prev = app->dpi_scale;
		}

		if (app->using_imgui) {
			ImGui_ImplSDL3_NewFrame();
			ImGui::NewFrame();
		}

		app->cmd = SDL_GpuAcquireCommandBuffer(app->device);
		cf_shader_watch();
	}
	app->user_on_update = on_update;
	cf_update_time(s_on_update);
}

static void s_imgui_present(SDL_GpuTexture* swapchain_texture)
{
	if (app->using_imgui) {
		ImGui::EndFrame();
		ImGui::Render();
		cf_imgui_draw(swapchain_texture);
	}
}

int cf_app_draw_onto_screen(bool clear)
{
	// Update lifteime of all text effects.
	const char** keys = app->text_effect_states.keys();
	CF_TextEffectState* effect_states = app->text_effect_states.items();
	int count = app->text_effect_states.count();
	for (int i = 0; i < count;) {
		if (!effect_states[i].alive) {
			app->text_effect_states.remove(keys[i]);
			--count;
		} else {
			effect_states[i].alive = false;
			++i;
		}
	}

	// Update the spritebatch itself.
	// This does atlas management internally.
	// All references to backend texture id's are now invalid (fetch_image or canvas_get_backend_target_handle).
	if (!draw->delay_defrag) {
		spritebatch_tick(&draw->sb);
		spritebatch_defrag(&draw->sb);
	}

	// Render any remaining geometry in the draw API.
	cf_render_to(app->offscreen_canvas, clear);

	// Stretch the app canvas onto the backbuffer canvas.
	Uint32 w, h;
	SDL_GpuTexture* swapchain_tex = SDL_GpuAcquireSwapchainTexture(app->cmd, app->window, &w, &h);
	((CF_CanvasInternal*)app->canvas.id)->texture = swapchain_tex;
	//((CF_CanvasInternal*)app->canvas.id)->depth_stencil = depth_stencil;
	if (swapchain_tex) {
		cf_apply_canvas(app->canvas, true);
		{
			cf_apply_mesh(app->backbuffer_quad);
			v2 u_texture_size = V2((float)app->w, (float)app->h);
			cf_material_set_uniform_fs(app->backbuffer_material, "u_texture_size", &u_texture_size, CF_UNIFORM_TYPE_FLOAT2, 1);
			cf_apply_shader(app->backbuffer_shader, app->backbuffer_material);
			cf_draw_elements();
			cf_commit();
		}

		// Dear ImGui draw.
		if (app->using_imgui) {
			s_imgui_present(swapchain_tex);
		}
	}

	// Do defrag down here after rendering ImGui to avoid thrashing any texture IDs. Generally we want to defrag
	// before doing final rendering to reduce draw call count, but in the case where ImGui is rendered it's acceptable
	// to have the perf-hit and delay until next frame.
	if (draw->delay_defrag) {
		spritebatch_tick(&draw->sb);
		spritebatch_defrag(&draw->sb);
		draw->delay_defrag = false;
	}

	SDL_GpuSubmit(app->cmd);
	app->cmd = NULL;

	// Clear all pushed draw parameters.
	draw->colors.set_count(1);
	draw->tints.set_count(1);
	draw->antialias.set_count(1);
	draw->antialias_scale.set_count(1);
	draw->render_states.set_count(1);
	draw->scissors.set_count(1);
	draw->viewports.set_count(1);
	draw->layers.set_count(1);
	draw->layers.set_count(1);
	draw->reset_cam();
	draw->font_sizes.set_count(1);
	draw->fonts.set_count(1);
	draw->blurs.set_count(1);
	draw->text_wrap_widths.set_count(1);
	draw->text_clip_boxes.set_count(1);
	draw->vertical.set_count(1);
	draw->user_params.set_count(1);
	draw->shaders.set_count(1);

	// Report the number of draw calls.
	// This is always user draw call count +1.
	int draw_call_count = app->draw_call_count;
	app->draw_call_count = 0;
	return draw_call_count;
}

void cf_app_get_size(int* w, int* h)
{
	if (w) *w = app->w;
	if (h) *h = app->h;
}

int cf_app_get_width()
{
	return app->w;
}

int cf_app_get_height()
{
	return app->h;
}

void cf_app_show_window()
{
	SDL_ShowWindow(app->window);
}

float cf_app_get_dpi_scale()
{
	return app->dpi_scale;
}

bool cf_app_dpi_scale_was_changed()
{
	return app->dpi_scale_was_changed;
}

void cf_app_set_size(int w, int h)
{
	SDL_SetWindowSize(app->window, w, h);
	app->w = w;
	app->h = h;
}

void cf_app_get_position(int* x, int* y)
{
	if (x) *x = app->x;
	if (y) *y = app->y;
}

void cf_app_set_position(int x, int y)
{
	SDL_SetWindowPosition(app->window, x, y);
}

bool cf_app_was_resized()
{
	return app->window_state.resized;
}

bool cf_app_was_moved()
{
	return app->window_state.moved;
}

bool cf_app_lost_focus()
{
	return !app->window_state.has_keyboard_focus && app->window_state_prev.has_keyboard_focus;
}

bool cf_app_gained_focus()
{
	return app->window_state.has_keyboard_focus && !app->window_state_prev.has_keyboard_focus;
}

bool cf_app_has_focus()
{
	return app->window_state.has_keyboard_focus;
}

void cf_app_request_attention()
{
	SDL_FlashWindow(app->window, SDL_FLASH_BRIEFLY);
}

void cf_app_request_attention_continuously()
{
	SDL_FlashWindow(app->window, SDL_FLASH_UNTIL_FOCUSED);
}

void cf_app_request_attention_cancel()
{
	SDL_FlashWindow(app->window, SDL_FLASH_CANCEL);
}

bool cf_app_was_minimized()
{
	return app->window_state.minimized && !app->window_state_prev.minimized;
}

bool cf_app_was_maximized()
{
	return app->window_state.maximized && !app->window_state_prev.maximized;
}

bool cf_app_minimized()
{
	return app->window_state.minimized;
}

bool cf_app_maximized()
{
	return app->window_state.maximized;
}

bool cf_app_was_restored()
{
	return app->window_state.restored && !app->window_state_prev.restored;
}

bool cf_app_mouse_entered()
{
	return app->window_state.mouse_inside_window && !app->window_state_prev.mouse_inside_window;
}

bool cf_app_mouse_exited()
{
	return !app->window_state.mouse_inside_window && app->window_state_prev.mouse_inside_window;
}

bool cf_app_mouse_inside()
{
	return app->window_state.mouse_inside_window;
}

CF_Canvas cf_app_get_canvas()
{
	return app->offscreen_canvas;
}

void cf_app_set_canvas_size(int w, int h)
{
	s_canvas(w, h);
}

int cf_app_get_canvas_width()
{
	return app->canvas_w;
}

int cf_app_get_canvas_height()
{
	return app->canvas_h;
}

void cf_app_set_vsync(bool true_turn_on_vsync)
{
	app->vsync = true_turn_on_vsync;
	SDL_GpuSetSwapchainParameters(app->device, app->window, SDL_GPU_SWAPCHAINCOMPOSITION_SDR, app->vsync ? SDL_GPU_PRESENTMODE_VSYNC : SDL_GPU_PRESENTMODE_IMMEDIATE);
}

bool cf_app_get_vsync()
{
	return app->vsync;
}

void cf_app_set_windowed_mode()
{
	SDL_SetWindowFullscreen(app->window, 0);
}

void cf_app_set_borderless_fullscreen_mode()
{
	SDL_SetWindowFullscreen(app->window, true);
}

void cf_app_set_fullscreen_mode()
{
	SDL_SetWindowFullscreen(app->window, SDL_WINDOW_FULLSCREEN);
}

void cf_app_set_title(const char* title)
{
	SDL_SetWindowTitle(app->window, title);
}

// Here for easy porting from SDL2.
SDL_Surface *SDL_CreateRGBSurfaceFrom(void *pixels, int width, int height, int depth, int pitch, Uint32 Rmask, Uint32 Gmask, Uint32 Bmask, Uint32 Amask)
{
	return SDL_CreateSurfaceFrom(width, height, SDL_GetPixelFormatForMasks(depth, Rmask, Gmask, Bmask, Amask), pixels, pitch);
}

void cf_app_set_icon(const char* virtual_path_to_png)
{
	CF_Image img;
	if (is_error(cf_image_load_png(virtual_path_to_png, &img))) {
		fprintf(stderr, "Unable to open icon png file %s.", virtual_path_to_png);
		return;
	}
	SDL_Surface* icon = SDL_CreateRGBSurfaceFrom(img.pix, img.w, img.h, 32, img.w * 4, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000);
	SDL_SetWindowIcon(app->window, icon);
	SDL_DestroySurface(icon);
	cf_image_free(&img);
}

ImGuiContext* cf_app_init_imgui()
{
	if (!app->gfx_enabled) return NULL;
	
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	app->using_imgui = true;

	ImGui::StyleColorsDark();
	
	SDL_GpuDriver driver = SDL_GpuGetDriver(app->device);
	switch (driver) {
	case SDL_GPU_DRIVER_VULKAN: ImGui_ImplSDL3_InitForVulkan(app->window); break;
	case SDL_GPU_DRIVER_D3D11:  // Fall-thru.
	case SDL_GPU_DRIVER_D3D12:  ImGui_ImplSDL3_InitForD3D(app->window);    break;
	case SDL_GPU_DRIVER_METAL:  ImGui_ImplSDL3_InitForMetal(app->window);  break;
	}

	cf_imgui_init();

	return ::ImGui::GetCurrentContext();
}

CF_PowerInfo cf_app_power_info()
{
	CF_PowerInfo info;
	SDL_PowerState state = SDL_GetPowerInfo(&info.seconds_left, &info.percentage_left);
	switch (state) {
	case SDL_POWERSTATE_UNKNOWN: info.state = CF_POWER_STATE_UNKNOWN;
	case SDL_POWERSTATE_ON_BATTERY: info.state = CF_POWER_STATE_ON_BATTERY;
	case SDL_POWERSTATE_NO_BATTERY: info.state = CF_POWER_STATE_NO_BATTERY;
	case SDL_POWERSTATE_CHARGING: info.state = CF_POWER_STATE_CHARGING;
	case SDL_POWERSTATE_CHARGED: info.state = CF_POWER_STATE_CHARGED;
	}
	return info;
}

void cf_default_assert(bool expr, const char* message, const char* file, int line)
{
	if (!expr) {
		fprintf(stderr, "CF_ASSERT(%s) : %s, line %d\n", message, file, line);
#ifdef _MSC_VER
		__debugbreak();
#endif
	}
}

cf_assert_fn* g_assert_fn = cf_default_assert;

void cf_set_assert_handler(cf_assert_fn* assert_fn)
{
	g_assert_fn = assert_fn;
}
