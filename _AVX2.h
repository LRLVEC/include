#pragma once
#include <immintrin.h>

namespace AVX2
{
	//32 byte aligned
	void* mallocAVX2(size_t _size)
	{
		return _mm_malloc(_size, 32);
	}
	void freeAVX2(void* ptr)
	{
		_mm_free(ptr);
	}
}