#pragma once
#include <optix.h>
#define NOMINMAX
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix.h>
#define CloseRay 0
#define AnyRay 1

using namespace optix;
static __device__ __inline__ float random(float2 st)
{
	float a(fabsf(sin(dot(st, make_float2(12.9898f, 78.233f))) * 43758.5453123f));
	return a - floorf(a);
}
static __device__ __inline__ float2 random(float2 st, float f)
{
	float a(random(st));
	st.x += f;
	st.y -= f;
	return make_float2(a, random(st));
}
static __device__ __inline__ float2 random(uint2 st, uint2 xy, unsigned int frame)
{
	float2 a = make_float2(st) / make_float2(xy);
	a.x += float(frame);
	float b(random(a));
	a.y += float(frame);
	return make_float2(b, random(a));
}
namespace Define
{
	/*template<unsigned int N>static __host__ __device__ __inline__ unsigned int tea(unsigned int val0, unsigned int val1)
	{
		unsigned int v0 = val0;
		unsigned int v1 = val1;
		unsigned int s0 = 0;
		for (unsigned int n = 0; n < N; n++)
		{
			s0 += 0x9e3779b9;
			v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
			v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
		}
		return v0;
	}
	static __host__ __device__ __inline__ unsigned int lcg(unsigned int& prev)//[0, 2^24)
	{
		const unsigned int LCG_A = 1664525u;
		const unsigned int LCG_C = 1013904223u;
		prev = (LCG_A * prev + LCG_C);
		return prev & 0x00FFFFFF;
	}
	static __host__ __device__ __inline__ unsigned int lcg2(unsigned int& prev)
	{
		prev = (prev * 8121 + 28411) % 134456;
		return prev;
	}
	static __host__ __device__ __inline__ float random(unsigned int& prev)
	{
		return ((float)lcg(prev) / (float)0x01000000);
	}
	static __host__ __device__ __inline__ unsigned int rotSeed(unsigned int seed, unsigned int frame)
	{
		return seed ^ frame;
	}*/
	struct RayData
	{
		float3 color;
		int depth;
	};
	struct Trans
	{
		Matrix3x4 trans;
		float3 r0;
		float z0;
	};
}