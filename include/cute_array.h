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

#ifndef CUTE_ARRAY_H
#define CUTE_ARRAY_H


#include "cute_defines.h"
#include "cute_c_runtime.h"
#include "cute_alloc.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

CUTE_INLINE void cf_array_ensure_capacity(void** items, int items_count, int *items_capacity, int items_element_size, int required_capacity, void* user_allocator_context)
{
	if (required_capacity > *items_capacity) {
		int new_capacity = *items_capacity ? *items_capacity * 2 : 256;
		while (new_capacity < required_capacity) {
			new_capacity <<= 1;
			CUTE_ASSERT(new_capacity); // Detect overflow.
		}

		size_t new_size = items_element_size * new_capacity;
		void* new_items = CUTE_ALLOC(new_size, user_allocator_context);
		CUTE_ASSERT(new_items);
		CUTE_MEMCPY(new_items, *items, items_element_size * items_count);
		CUTE_FREE(*items, user_allocator_context);
		*items = new_items;
		*items_capacity = new_capacity;
	}
}

#ifdef __cplusplus
}
#endif // __cplusplus

#ifdef CUTE_CPP

/**
 * Implements a basic growable array data structure. Constructors and destructors are called, but this
 * class does *not* act as a drop-in replacement for std::vector, as there are no iterators.
 *
 * The main purpose of this class is to reduce the lines of code included compared to std::vector,
 * and also more importantly to have fast debug performance.
 */

template <typename T>
struct cf_array
{
	cf_array();
	cf_array(cf_initializer_list<T> list);
	cf_array(const cf_array<T>& other);
	cf_array(cf_array<T>&& other);
	cf_array(void* user_allocator_context);
	cf_array(int capacity, void* user_allocator_context);
	cf_array(T* data, int count, int capacity, void* user_allocator_context);
	~cf_array();

	T& add();
	T& add(const T& item);
	T& add(T&& item);
	T& insert(int index);
	T& insert(int index, const T& item);
	T& insert(int index, T&& item);
	void set(int index, const T& item);
	void remove(int index);
	T pop();
	void unordered_remove(int index);
	void clear();
	void ensure_capacity(int num_elements);
	void ensure_count(int count);
	void set_count(int count);
	void steal_from(cf_array<T>* steal_from_me);
	void steal_from(cf_array<T>& steal_from_me);
	void reverse();

	int capacity() const;
	int count() const;
	int size() const;

	T* begin();
	const T* begin() const;
	T* end();
	const T* end() const;

	T& operator[](int index);
	const T& operator[](int index) const;

	T* operator+(int index);
	const T* operator+(int index) const;

	const cf_array<T>& operator=(const cf_array<T>& rhs);
	const cf_array<T>& operator=(cf_array<T>&& rhs);

	T& last();
	const T& last() const;

	T* data();
	const T* data() const;

	//private:
	int m_capacity = 0;
	int m_count = 0;
	T* m_items = NULL;
	void* m_mem_ctx = NULL;
};

// -------------------------------------------------------------------------------------------------

template <typename T>
cf_array<T>::cf_array()
{}

template <typename T>
cf_array<T>::cf_array(cf_initializer_list<T> list)
{
	ensure_capacity((int)list.size());
	for (const T* i = list.begin(); i < list.end(); ++i) {
		add(*i);
	}
}

template <typename T>
cf_array<T>::cf_array(const cf_array<T>& other)
{
	ensure_capacity((int)other.count());
	for (int i = 0; i < other.count(); ++i) {
		add(other[i]);
	}
}

template <typename T>
cf_array<T>::cf_array(cf_array<T>&& other)
{
	steal_from(&other);
}

template <typename T>
cf_array<T>::cf_array(void* user_allocator_context)
	: m_mem_ctx(user_allocator_context)
{}

template <typename T>
cf_array<T>::cf_array(int capacity, void* user_allocator_context)
	: m_capacity(capacity)
	, m_mem_ctx(user_allocator_context)
{
	m_items = (T*)CUTE_ALLOC(sizeof(T) * capacity, m_mem_ctx);
	CUTE_ASSERT(m_items);
}


template <typename T>
cf_array<T>::cf_array(T* data, int count, int capacity, void* user_allocator_context)
	: m_items(data)
	, m_count(count)
	, m_capacity(capacity)
	, m_mem_ctx(user_allocator_context)
{
	CUTE_ASSERT(m_items);
}


template <typename T>
cf_array<T>::~cf_array()
{
	for (int i = 0; i < m_count; ++i) {
		T* slot = m_items + i;
		slot->~T();
	}
	CUTE_FREE(m_items, m_mem_ctx);
}

template <typename T>
T& cf_array<T>::add()
{
	ensure_capacity(m_count + 1);
	T* slot = m_items + m_count++;
	CUTE_PLACEMENT_NEW(slot) T;
	return *slot;
}

template <typename T>
T& cf_array<T>::add(const T& item)
{
	ensure_capacity(m_count + 1);
	T* slot = m_items + m_count++;
	CUTE_PLACEMENT_NEW(slot) T(item);
	return *slot;
}

template <typename T>
T& cf_array<T>::add(T&& item)
{
	ensure_capacity(m_count + 1);
	T* slot = m_items + m_count++;
	CUTE_PLACEMENT_NEW(slot) T(move(item));
	return *slot;
}

template <typename T>
T& cf_array<T>::insert(int index)
{
	CUTE_ASSERT(index >= 0 && index < m_count);
	add();
	int count_to_move = m_count - 1 - index;
	CUTE_MEMMOVE(m_items + index + 1, m_items + index, sizeof(T) * count_to_move);
	return m_items[index];
}

template <typename T>
T& cf_array<T>::insert(int index, const T& item)
{
	CUTE_ASSERT(index >= 0 && index < m_count);
	add();
	int count_to_move = m_count - 1 - index;
	CUTE_MEMMOVE(m_items + index + 1, m_items + index, sizeof(T) * count_to_move);
	T* slot = m_items + index;
	CUTE_PLACEMENT_NEW(slot) T(item);
	return *slot;
}

template <typename T>
T& cf_array<T>::insert(int index, T&& item)
{
	CUTE_ASSERT(index >= 0 && index < m_count);
	add();
	int count_to_move = m_count - 1 - index;
	CUTE_MEMMOVE(m_items + index + 1, m_items + index, sizeof(T) * count_to_move);
	T* slot = m_items + index;
	CUTE_PLACEMENT_NEW(slot) T(move(item));
	return *slot;
}

template <typename T>
void cf_array<T>::set(int index, const T& item)
{
	CUTE_ASSERT(index >= 0 && index < m_count);
	T* slot = m_items + index;
	*slot = item;
}

template <typename T>
void cf_array<T>::remove(int index)
{
	CUTE_ASSERT(index >= 0 && index < m_count);
	T* slot = m_items + index;
	slot->~T();
	int count_to_move = m_count - 1 - index;
	CUTE_MEMMOVE(m_items + index, m_items + index + 1, sizeof(T) * count_to_move);
	--m_count;
}

template <typename T>
T cf_array<T>::pop()
{
	CUTE_ASSERT(m_count > 0);
	T* slot = m_items + m_count - 1;
	T val = move(m_items[--m_count]);
	slot->~T();
	return val;
}

template <typename T>
void cf_array<T>::unordered_remove(int index)
{
	CUTE_ASSERT(index >= 0 && index < m_count);
	T* slot = m_items + index;
	slot->~T();
	m_items[index] = m_items[--m_count];
}

template <typename T>
void cf_array<T>::clear()
{
	for (int i = 0; i < m_count; ++i) {
		T* slot = m_items + i;
		slot->~T();
	}
	m_count = 0;
}

template <typename T>
void cf_array<T>::ensure_capacity(int num_elements)
{
	if (num_elements > m_capacity) {
		int new_capacity = m_capacity ? m_capacity * 2 : 256;
		while (new_capacity < num_elements) {
			new_capacity <<= 1;
			CUTE_ASSERT(new_capacity); // Detect overflow.
		}

		size_t new_size = sizeof(T) * new_capacity;
		T* new_items = (T*)CUTE_ALLOC(new_size, m_mem_ctx);
		CUTE_ASSERT(new_items);
		CUTE_MEMCPY(new_items, m_items, sizeof(T) * m_count);
		CUTE_FREE(m_items, m_mem_ctx);
		m_items = new_items;
		m_capacity = new_capacity;
	}
}

template <typename T>
void cf_array<T>::set_count(int count)
{
	CUTE_ASSERT(count < m_capacity || !count);
	if (m_count > count) {
		for (int i = count; i < m_count; ++i) {
			T* slot = m_items + i;
			slot->~T();
		}
	} else if (m_count < count) {
		for (int i = m_count; i < count; ++i) {
			T* slot = m_items + i;
			CUTE_PLACEMENT_NEW(slot) T;
		}
	}
	m_count = count;
}

template <typename T>
void cf_array<T>::ensure_count(int count)
{
	int old_count = m_count;
	ensure_capacity(count);
	if (m_count < count) {
		m_count = count;
		for (int i = old_count; i < count; ++i) {
			T* slot = m_items + i;
			CUTE_PLACEMENT_NEW(slot) T;
		}
	}
}

template <typename T>
void cf_array<T>::steal_from(cf_array<T>* steal_from_me)
{
	this->~cf_array<T>();
	m_capacity = steal_from_me->m_capacity;
	m_count = steal_from_me->m_count;
	m_items = steal_from_me->m_items;
	m_mem_ctx = steal_from_me->m_mem_ctx;
	CUTE_PLACEMENT_NEW(steal_from_me) cf_array<T>();
}

template <typename T>
void cf_array<T>::reverse()
{
	T* a = m_items;
	T* b = m_items + (m_count - 1);

	while (a < b) {
		T t = *a;
		*a = *b;
		*b = t;
		++a;
		--b;
	}
}

template <typename T>
void cf_array<T>::steal_from(cf_array<T>& steal_from_me)
{
	steal_from(&steal_from_me);
}

template <typename T>
int cf_array<T>::capacity() const
{
	return m_capacity;
}

template <typename T>
int cf_array<T>::count() const
{
	return m_count;
}

template <typename T>
int cf_array<T>::size() const
{
	return m_count;
}

template <typename T>
T* cf_array<T>::begin()
{
	return m_items;
}

template <typename T>
const T* cf_array<T>::begin() const
{
	return m_items;
}

template <typename T>
T* cf_array<T>::end()
{
	return m_items + m_count;
}

template <typename T>
const T* cf_array<T>::end() const
{
	return m_items + m_count;
}

template <typename T>
T& cf_array<T>::operator[](int index)
{
	CUTE_ASSERT(index >= 0 && index < m_count);
	return m_items[index];
}

template <typename T>
const T& cf_array<T>::operator[](int index) const
{
	CUTE_ASSERT(index >= 0 && index < m_count);
	return m_items[index];
}

template <typename T>
T* cf_array<T>::data()
{
	return m_items;
}

template <typename T>
const T* cf_array<T>::data() const
{
	return m_items;
}

template <typename T>
T* cf_array<T>::operator+(int index)
{
	CUTE_ASSERT(index >= 0 && index < m_count);
	return m_items + index;
}

template <typename T>
const T* cf_array<T>::operator+(int index) const
{
	CUTE_ASSERT(index >= 0 && index < m_count);
	return m_items + index;
}

template <typename T>
const cf_array<T>& cf_array<T>::operator=(const cf_array<T>& rhs)
{
	set_count(0);
	ensure_capacity((int)rhs.count());
	for (int i = 0; i < rhs.count(); ++i) {
		add(rhs[i]);
	}
	return *this;
}

template <typename T>
const cf_array<T>& cf_array<T>::operator=(cf_array<T>&& rhs)
{
	steal_from(rhs);
	return *this;
}

template <typename T>
T& cf_array<T>::last()
{
	return (*this)[m_count - 1];
}

template <typename T>
const T& cf_array<T>::last() const
{
	return (*this)[m_count - 1];
}

namespace cute
{
template <typename T> using array = cf_array<T>;
}

#endif // CUTE_CPP

#endif // CUTE_ARRAY_H
