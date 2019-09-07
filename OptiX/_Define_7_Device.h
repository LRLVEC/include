#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <optix_device.h>
#include <OptiX/_vector_functions.hpp>

static __forceinline__ __device__ void* uP(unsigned int i0, unsigned int i1)
{
	const unsigned long long uptr = (unsigned long long)(i0) << 32 | i1;
	void* ptr = (void*)(uptr);
	return ptr;
}
static __forceinline__ __device__ void  pP(void* ptr, unsigned int& i0, unsigned int& i1)
{
	const unsigned long long  uptr = (unsigned long long)(ptr);
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}
static __device__ __inline__ float random(float2 st)
{
	float a(fabsf(sin(dot(st, make_float2(12.9898f, 78.233f))) * 43758.5453123f));
	return a - floorf(a);
}
static __device__ __inline__ float2 random(float2 st, float f)
{
	st.x += f;
	float a(random(st));
	st.y -= f;
	return make_float2(a, random(st));
}
static __device__ __inline__ float2 random(uint2 st, uint2 xy, unsigned int frame)
{
	float2 a = make_float2(st) / make_float2(xy);
	a.x += float(frame);
	float b(random(a));
	a.y -= float(frame);
	return make_float2(b, random(a));
}
static __device__ __inline__ float2 random1(uint2 st, uint2 xy, unsigned int frame)
{
	float2 a = make_float2(st) / make_float2(xy);
	a.x += sqrtf(frame);
	float b(random(a));
	a.y -= sqrtf(frame);
	return make_float2(b, random(a));
}
static __device__ __inline__ float2 randomCircle(float2 seed)
{
	seed.x = sqrtf(seed.x);
	seed.y *= 2 * M_PIf;
	return make_float2(cosf(seed.y) * seed.x, sinf(seed.y) * seed.x);
}
static __device__ __inline__ float3 randomNormal(float3 n, float2 seed)
{
	float3 u;
	float3 v;
	seed.x = random(seed);
	seed.y = 2.0 * M_PIf * random(seed);
	if (fabsf(n.x) > 0.7f)
	{
		float s = sqrtf(1 - n.y * n.y);
		u = make_float3(-n.z, 0, n.x) / s;
		v = make_float3(n.x * n.y / s, -s, n.y * n.z / s);
	}
	else
	{
		float s = sqrtf(1 - n.x * n.x);
		u = make_float3(0, n.z, -n.y) / s;
		v = make_float3(-s, n.x * n.y / s, n.x * n.z / s);
	}
	float sinTheta = sqrtf(1 - seed.x * seed.x);
	return sinTheta * cosf(seed.y) * u + sinTheta * sin(seed.y) * v + seed.x * n;

}
static __device__ __inline__ float3 randomDirectionCosN(float3 normal, float n, float2 seed)
{
	float3 u;
	float3 v;
	seed.x = random(seed);
	seed.y = 2.0 * M_PIf * random(seed);
	if (fabsf(normal.x) > 0.7f)
	{
		float s = sqrtf(1 - normal.y * normal.y);
		u = make_float3(-normal.z, 0, normal.x) / s;
		v = make_float3(normal.x * normal.y / s, -s, normal.y * normal.z / s);
	}
	else
	{
		float s = sqrtf(1 - normal.x * normal.x);
		u = make_float3(0, normal.z, -normal.y) / s;
		v = make_float3(-s, normal.x * normal.y / s, normal.x * normal.z / s);
	}
	seed.x = powf(seed.x, 1.0f / (n + 1));
	return sqrtf(1 - seed.x * seed.x) * (cosf(seed.y) * u + sin(seed.y) * v) + seed.x * normal;
}

struct TransInfo
{
	float4 row0;
	float4 row1;
	float4 row2;
	float3 r0;
	float z0;
};