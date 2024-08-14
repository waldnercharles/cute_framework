#include <cute.h>
using namespace Cute;

int w = 480*2;
int h = 270*2;
float scale = 1;

int main(int argc, char* argv[])
{
	make_app("Metaballs", 0, 0, 0, (int)(w*scale), (int)(h*scale), APP_OPTIONS_WINDOW_POS_CENTERED_BIT, argv[0]);
	cf_shader_directory("/metaballs_data");
	CF_Canvas soft_circles = make_canvas(canvas_defaults(w, h));
	CF_Shader shd = cf_make_draw_shader("metaballs.shd");
	float t = 0;

	float frame_times[10] = { 0 };
	int frame_index = 0;
	float fps = 0;

	while (app_is_running()) {
		app_update();

		frame_times[frame_index] = DELTA_TIME;
		frame_index = (frame_index + 1) % 10;
		float sum = 0;
		for (int i = 0; i < 10; ++i) {
			sum += frame_times[i];
		}
		sum /= 10.0f;
		fps = (1.0f / sum);
		if (frame_index % 10 == 0) printf("%f\n", fps);

		draw_scale(scale, scale);
		t = t + DELTA_TIME;

		// Draw soft-circles.
		Rnd rnd = rnd_seed(0);
		for (int i = 0; i < 100; ++i)
		{
			float o = rnd_range(rnd, -10.0f,10.0f);
			float x = rnd_range(rnd, -w/scale*0.5f, w/scale*0.5f) + cosf(t+o) * 10;
			float y = rnd_range(rnd, -h/scale*0.5f, h/scale*0.5f) + sinf(t+o) * 10;
			float r = rnd_range(rnd, 10.0f,60.0f);
		
			// Render a perfect soft-circle using anti-alias scaling and zero-radius.
			draw_push_antialias_scale(r);
			draw_circle_fill(make_circle(V2(x, y), 0));
			draw_pop_antialias_scale();
		}

		static int toggle = false;
		if (key_just_pressed(KEY_SPACE)) {
			toggle = !toggle;
		}
		if (!toggle) {
			// Render soft circles onto their own canvas, to feed into the metaballs filter shader.
			render_to(soft_circles, true);
		
			// Apply the metaball filter to the soft circle render.
			render_settings_push_shader(shd);
			render_settings_push_texture("tex", canvas_get_target(soft_circles));
			draw_box_fill(make_aabb(V2(-w/scale, -h/scale), V2(w/scale, h/scale)));
			render_to(app_get_canvas());
			render_settings_pop_shader();
		}

		draw_push();
		draw_scale(5,5);
		draw_text("press space", -V2(text_width("press_space") * 0.5f, 0));
		draw_pop();

		//cf_canvas_blit(soft_circles, V2(0,0), V2(1,1), app_get_canvas(), V2(0,0), V2(1,1));

		app_draw_onto_screen();
	}

	destroy_canvas(soft_circles);
	destroy_app();
	return 0;
}
