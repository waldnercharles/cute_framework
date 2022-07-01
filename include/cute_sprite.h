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

#ifndef CUTE_SPRITE_H
#define CUTE_SPRITE_H

#include "cute_defines.h"
#include "cute_array.h"
#include "cute_dictionary.h"
#include "cute_batch.h"

/**
 * Represents one frame of animation within a sprite.
 */
typedef struct cf_frame_t
{
	uint64_t id;
	float delay;
} cf_frame_t;

/**
 * The direction a sprite plays frames.
 */
typedef enum cf_play_direction_t
{
	CF_PLAY_DIRECTION_FORWARDS,
	CF_PLAY_DIRECTION_BACKWARDS,
	CF_PLAY_DIRECTION_PINGPONG,
} cf_play_direction_t;

/**
 * A single `sprite_t` contains a set of `animation_t`. Each animation
 * can define its own frames, and a playing direction for the frames.
 */
typedef struct cf_animation_t
{
	const char* name; /*= NULL*/
	cf_play_direction_t play_direction; /*= CF_PLAY_DIRECTION_FORWARDS*/
	cf_array<cf_frame_t> frames;
} cf_animation_t;

/**
 * An animation table is a set of animations a particular sprite references.
 * Each sprite instance contains a pointer to an animation table in a "read-only" fashion.
 */
using cf_animation_table_t = cf_dictionary<const char*, const cf_animation_t*>;

/**
 * The `sprite_t` represents a set of drawable animations. Each animation is a collection
 * of frames, where each frame is one image to display on screen. The frames themselves are stored
 * elsewhere, and the sprite simply refers to them by read-only pointer.
 *
 * Switching between animations can be done by calling the `play` and passing the name of the animation
 * to the `play` method.
 */
typedef struct cf_sprite_t
{
	/**
	 * Updates the sprite's internal timer to flip through different frames.
	 */
	CUTE_INLINE void update(float dt);

	/**
	 * Switches to a new aninmation and starts playing it from the beginning.
	 */
	CUTE_INLINE void play(const char* animation);

	/**
	 * Returns true if `animation` the is currently playing animation.
	 */
	CUTE_INLINE bool is_playing(const char* animation);

	/**
	 * Resets the currently playing animation and unpauses the animation.
	 */
	CUTE_INLINE void reset();

	/**
	 * Pushes an instance of this sprite onto the `batch` member, which will be drawn the next time
	 * `batch_flush` is called on `batch`.
	 *
	 * `batch` must not be NULL.
	 */
	CUTE_INLINE void draw(cf_batch_t* batch);

	/**
	 * A lower level utility function used within the `draw` method. This is useful to prepare
	 * the sprite's drawable quad in a specific way before handing it off to a `batch`, to implement
	 * custom graphics effects.
	 */
	CUTE_INLINE cf_batch_sprite_t batch_sprite(cf_transform_t transform);
	CUTE_INLINE cf_batch_sprite_t batch_sprite();

	CUTE_INLINE void pause();
	CUTE_INLINE void unpause();
	CUTE_INLINE void toggle_pause();
	CUTE_INLINE void flip_x();
	CUTE_INLINE void flip_y();
	CUTE_INLINE int frame_count() const;
	CUTE_INLINE int current_frame() const;

	/**
	 * Returns the `delay` member of the currently playing frame, in milliseconds.
	 */
	CUTE_INLINE float frame_delay();

	/**
	 * Sums all the delays of each frame in the animation, and returns the total, in milliseconds.
	 */
	CUTE_INLINE float animation_delay();

	/**
	 * Returns a value from 0 to 1 representing how far along the animation has played. 0 means
	 * just started, while 1 means finished.
	 */
	CUTE_INLINE float animation_interpolant();

	/**
	 * Returns true if the animation will loop around and finish if `update` is called. This is useful
	 * to see if you're currently on the last frame of animation, and will finish in the particular
	 * `dt` tick.
	 */
	CUTE_INLINE bool will_finish(float dt);

	/**
	 * Returns true whenever at the very beginning of the animation sequence. This is useful for polling
	 * on when the animation restarts itself, for example, polling within an if-statement each game tick.
	 */
	CUTE_INLINE bool on_loop();

	const char* name = NULL;
	int w = 0;
	int h = 0;
	cf_v2 scale = cf_V2(1, 1);
	cf_v2 local_offset = cf_V2(0, 0);
	float opacity = 1.0f;
	int layer = 0;

	int frame_index = 0;
	int loop_count = 0;
	float play_speed_multiplier = 1.0f;
	const cf_animation_t* animation = NULL;

	bool paused = false;
	float t = 0;
	const cf_animation_table_t* animations = NULL;

	cf_transform_t transform = cf_make_transform();
} cf_sprite_t;

//--------------------------------------------------------------------------------------------------
// Easy sprite API. These functions are for loading single-frame sprites with no animations in a
// very simple way to get started. The preferred way to deal with sprites is the Aseprite sprite API,
// but the easy API is good for testing or just getting started with Cute Framework, or very simple
// games that don't require animations.

/**
 * Loads a single-frame sprite from a single png file. This function may be called many times in a row without
 * any significant performance penalties due to internal caching.
 */
CUTE_API cf_sprite_t CUTE_CALL cf_easy_sprite_make(const char* png_path);

/**
 * Unloads the sprites image resources from the internal cache. Any live `sprite_t` instances for
 * the given `png_path` will now be "dangling" and invalid.
 */
CUTE_API void CUTE_CALL cf_easy_sprite_unload(cf_sprite_t sprite);

/**
 * Gets the internal batch used for `easy_sprite_make` and `easy_sprite_unload`. The batch is used to get
 * sprites onto the screen by calling `batch_flush`.
 */
CUTE_API cf_batch_t* CUTE_CALL cf_easy_sprite_get_batch();

//--------------------------------------------------------------------------------------------------
// Aseprite sprite API. This is the preferred way to deal with sprites in Cute Framework, by loading
// .ase or .aseprite files directly. However, if you just want to get started with single-frame
// sprites loaded from a single png image, you can try the easy sprite API.

/**
 * Loads a sprite from an aseprite file. This function may be called many times in a row without
 * any significant performance penalties due to internal caching.
 */
CUTE_API cf_sprite_t CUTE_CALL cf_sprite_make(const char* aseprite_path);

/**
 * Unloads the sprites image resources from the internal cache. Any live `sprite_t` instances for
 * the given `aseprite_path` will now be "dangling" and invalid.
 */
CUTE_API void CUTE_CALL cf_sprite_unload(const char* aseprite_path);

/**
 * Gets the internal batch used for `sprite_make` and `sprite_unload`. The batch is used to get
 * sprites onto the screen by calling `batch_flush`.
 */
CUTE_API cf_batch_t* CUTE_CALL cf_sprite_get_batch();

//--------------------------------------------------------------------------------------------------
// In-line implementation of `sprite_t` member functions.

#include "cute_debug_printf.h"

void cf_sprite_t::update(float dt)
{
	if (paused) return;

	t += dt * play_speed_multiplier;
	if (t >= animation->frames[frame_index].delay) {
		frame_index++;
		if (frame_index == animation->frames.count()) {
			loop_count++;
			frame_index = 0;
		}

		t = 0;

		// TODO - Backwards and pingpong.
	}
}

void cf_sprite_t::play(const char* animation)
{
	this->animation = NULL;
	if (animations->find(animation, &this->animation).is_error()) {
		CUTE_DEBUG_PRINTF("Unable to find animation %s within sprite %s.\n", animation, name);
		CUTE_ASSERT(false);
	}
	reset();
}

bool cf_sprite_t::is_playing(const char* animation)
{
	return !CUTE_STRCMP(animation, this->animation->name);
}

void cf_sprite_t::reset()
{
	paused = false;
	frame_index = 0;
	loop_count = 0;
	t = 0;
}

void cf_sprite_t::draw(cf_batch_t* batch)
{
	CUTE_ASSERT(batch);
	cf_batch_push(batch, batch_sprite(transform));
}

cf_batch_sprite_t cf_sprite_t::batch_sprite(cf_transform_t transform)
{
	cf_batch_sprite_t q;
	q.id = animation->frames[frame_index].id;
	q.transform = transform;
	q.transform.p = cf_add_v2(q.transform.p, local_offset);
	q.w = w;
	q.h = h;
	q.scale_x = scale.x * w;
	q.scale_y = scale.y * h;
	q.sort_bits = layer;
	q.alpha = opacity;
	return q;
}

cf_batch_sprite_t cf_sprite_t::batch_sprite()
{
	cf_batch_sprite_t q;
	q.id = animation->frames[frame_index].id;
	q.transform = transform;
	q.transform.p = cf_add_v2(q.transform.p, local_offset);
	q.w = w;
	q.h = h;
	q.scale_x = scale.x * w;
	q.scale_y = scale.y * h;
	q.sort_bits = layer;
	q.alpha = opacity;
	return q;
}

void cf_sprite_t::pause()
{
	paused = true;
}

void cf_sprite_t::unpause()
{
	paused = false;
}

void cf_sprite_t::toggle_pause()
{
	paused = !paused;
}


void cf_sprite_t::flip_x()
{
	scale.x *= -1.0f;
}

void cf_sprite_t::flip_y()
{
	scale.y *= -1.0f;
}

int cf_sprite_t::frame_count() const
{
	return animation->frames.count();
}

int cf_sprite_t::current_frame() const
{
	return frame_index;
}

float cf_sprite_t::frame_delay()
{
	return animation->frames[frame_index].delay;
}

float cf_sprite_t::animation_delay()
{
	int count = frame_count();
	float delay = 0;
	for (int i = 0; i < count; ++i) {
		delay += animation->frames[i].delay;
	}
	return delay;
}

float cf_sprite_t::animation_interpolant()
{
	// TODO -- Backwards and pingpong.
	float delay = animation_delay();
	float t = this->t + animation->frames[frame_index].delay * frame_index;
	return cf_clamp(t / delay, 0.0f, 1.0f);
}

bool cf_sprite_t::will_finish(float dt)
{
	// TODO -- Backwards and pingpong.
	if (frame_index == frame_count() - 1) {
		return t + dt * play_speed_multiplier >= animation->frames[frame_index].delay;
	} else {
		return false;
	}
}

bool cf_sprite_t::on_loop()
{
	if (frame_index == 0 && t == 0) {
		return true;
	} else {
		return false;
	}
}

#ifdef  CUTE_CPP

namespace cute
{

}

#endif //  CUTE_CPP

#endif // CUTE_SPRITE_H
